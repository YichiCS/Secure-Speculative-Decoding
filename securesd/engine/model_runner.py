import gc

import torch
from transformers import AutoConfig

from securesd.config import Config
from securesd.engine.helpers.cudagraph_helpers import (
    capture_cudagraph,
    capture_verify_cudagraph,
    run_decode_cudagraph,
    run_verify_cudagraph,
)
from securesd.engine.helpers.runner_helpers import (
    prepare_block_tables_from_seqs,
    prepare_decode_packed_from_seqs,
    prepare_prefill_tensors_from_seqs,
)
from securesd.engine.sequence import Sequence
from securesd.layers.attention import Attention
from securesd.layers.sampler import Sampler
from securesd.models.llama3 import LlamaForCausalLM
from securesd.models.qwen3 import Qwen3ForCausalLM
from securesd.sampling_params import SamplingParams
from securesd.utils.context import reset_context, set_context
from securesd.utils.loader import load_model

_MODEL_REGISTRY = {
    "llama": LlamaForCausalLM,
    "qwen3": Qwen3ForCausalLM,
}


class ModelRunner:

    def __init__(self, config: Config, is_draft: bool = False):
        self.config = config
        self.is_draft = is_draft
        self._exiting = False
        self.hf_config = config.draft_hf_config if is_draft else config.hf_config
        self.block_size = config.kvcache_block_size
        self.max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size

        self.verify_k_plus_1 = config.speculate_k + 1

        self.device = torch.device("cuda")
        self._temperature_buffer = torch.empty(
            config.max_num_seqs, dtype=torch.float32, device=self.device
        )
        self._cached_uniform_temperature: float | None = None
        self._cached_uniform_temperature_count = 0

        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(self.hf_config.dtype)
        torch.set_default_device("cuda")

        self.setup_and_warmup_model_and_cudagraphs(config, self.hf_config)

        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

    def setup_and_warmup_model_and_cudagraphs(self, config: Config, hf_config: AutoConfig):
        self.graph_vars: dict = {}
        self.graph_pools: dict = {}
        self.graphs: dict = {}
        self.graph_bs_list: dict = {}

        model_type = hf_config.model_type
        if model_type not in _MODEL_REGISTRY:
            raise ValueError(f"Unsupported model type: {model_type}")
        model_class = _MODEL_REGISTRY[model_type]

        self.model = model_class(
            config=self.hf_config,
            draft=self.is_draft,
            speculate=self.config.speculate,
        )
        load_model(self.model, config.model)
        self.sampler = Sampler()
        self.warmup_model()
        self.allocate_kv_cache()

        dv, dp, dg, dbs = capture_cudagraph(self)
        self.graph_vars["decode"] = dv
        self.graph_pools["decode"] = dp
        self.graphs["decode"] = dg
        self.graph_bs_list["decode"] = dbs

        if self.config.speculate:
            vv, vp, vg, vbs = capture_verify_cudagraph(self)
            self.graph_vars["verify"] = vv
            self.graph_pools["verify"] = vp
            self.graphs["verify"] = vg
            self.graph_bs_list["verify"] = vbs

    def exit(self, hard: bool = True):
        if self._exiting:
            return
        self._exiting = True
        for attr in ("graphs", "graph_vars", "graph_pools", "graph_bs_list",
                     "kv_cache", "sampler", "model"):
            delattr(self, attr)
        gc.collect()
        torch.cuda.empty_cache()

    def warmup_model(self):
        torch.cuda.empty_cache()
        self.run([Sequence([0], SamplingParams())], True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = self.hf_config

        free, _ = torch.cuda.mem_get_info()
        num_kv_heads = hf_config.num_key_value_heads
        block_bytes = (
            2
            * hf_config.num_hidden_layers
            * self.block_size
            * num_kv_heads
            * hf_config.head_dim
            * hf_config.dtype.itemsize
        )
        usable = int(free * config.gpu_memory_utilization)
        config.num_kvcache_blocks = usable // block_bytes
        if config.num_kvcache_blocks <= 0:
            raise RuntimeError("KV cache too big for free memory")

        self.kv_cache = torch.empty(
            2,
            hf_config.num_hidden_layers,
            config.num_kvcache_blocks,
            self.block_size,
            num_kv_heads,
            hf_config.head_dim,
        )
        layer_id = 0
        for module in self.model.modules():
            if isinstance(module, Attention):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids, positions, cu_q, cu_k, max_q, max_k, slot_mapping = \
            prepare_prefill_tensors_from_seqs(seqs, self.block_size, self.is_draft)

        block_tables = None
        if cu_k[-1] > cu_q[-1]:
            block_tables = prepare_block_tables_from_seqs(seqs, self.is_draft)

        set_context(
            is_prefill=True,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=max_q,
            max_seqlen_k=max_k,
            slot_mapping=slot_mapping,
            context_lens=None,
            block_tables=block_tables,
        )
        return input_ids, positions

    def prepare_decode(
        self,
        seqs: list[Sequence],
        verify: bool = False,
        verify_k: int | None = None,
        build_verify_cu_seqlens: bool = True,
    ):
        k_verify = self.config.speculate_k if verify_k is None else int(verify_k)
        input_ids, positions, slot_mapping, context_lens, block_tables = prepare_decode_packed_from_seqs(
            seqs,
            self.block_size,
            self.is_draft,
            verify,
            k_verify if verify else -1,
        )

        if verify:
            if build_verify_cu_seqlens:
                seqlen_q = torch.full((len(seqs),), k_verify + 1, dtype=torch.int32, device=self.device)
                cu_seqlens_q = torch.zeros(len(seqs) + 1, dtype=torch.int32, device=self.device)
                cu_seqlens_q[1:] = torch.cumsum(seqlen_q, dim=0)
            else:
                cu_seqlens_q = None
            set_context(
                is_prefill=False,
                cu_seqlens_q=cu_seqlens_q,
                max_seqlen_q=k_verify + 1,
                slot_mapping=slot_mapping,
                context_lens=context_lens,
                block_tables=block_tables,
            )
        else:
            set_context(
                is_prefill=False,
                slot_mapping=slot_mapping,
                context_lens=context_lens,
                block_tables=block_tables,
            )

        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]) -> tuple[torch.Tensor, bool]:
        bs = len(seqs)
        if bs > self._temperature_buffer.numel():
            self._temperature_buffer = torch.empty(bs, dtype=torch.float32, device=self.device)
        buf = self._temperature_buffer[:bs]

        if bs == 0:
            return buf, True

        attr = "draft_temperature" if self.is_draft else "temperature"
        first_temp = float(getattr(seqs[0], attr))
        uniform = True
        for seq in seqs[1:]:
            if float(getattr(seq, attr)) != first_temp:
                uniform = False
                break

        if uniform:
            if (
                self._cached_uniform_temperature == first_temp
                and self._cached_uniform_temperature_count >= bs
            ):
                return buf, first_temp == 0.0
            buf.fill_(first_temp)
            self._cached_uniform_temperature = first_temp
            self._cached_uniform_temperature_count = bs
            return buf, first_temp == 0.0

        buf.copy_(torch.tensor(
            [float(getattr(seq, attr)) for seq in seqs], dtype=torch.float32
        ))
        self._cached_uniform_temperature = None
        self._cached_uniform_temperature_count = 0
        return buf, all(float(getattr(seq, attr)) == 0.0 for seq in seqs)

    @torch.inference_mode()
    def run_model(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        is_prefill: bool,
        last_only: bool = True,
        verify_query_len: int | None = None,
    ):
        is_mq_kp1 = self.config.speculate and not last_only
        qlen = self.verify_k_plus_1 if verify_query_len is None else int(verify_query_len)

        if is_prefill and not last_only:
            raise ValueError("run_model: is_prefill cannot combine with last_only=False")

        if is_prefill or (is_mq_kp1 and qlen != self.verify_k_plus_1):
            outputs = self.model(input_ids, positions)
            return self.model.compute_logits(outputs, last_only)

        if is_mq_kp1:
            return run_verify_cudagraph(self, input_ids, positions, last_only, self.graph_vars["verify"])
        return run_decode_cudagraph(self, input_ids, positions, last_only, self.graph_vars["decode"])

    def run(
        self,
        seqs: list[Sequence],
        is_prefill: bool,
        last_only: bool = True,
        draft_return_logits: bool = False,
        verify_k: int | None = None,
        return_token_tensor: bool = False,
    ) -> list[int] | torch.Tensor | tuple:
        if is_prefill:
            input_ids, positions = self.prepare_prefill(seqs)
            verify_k_eff: int | None = None
        else:
            verify_k_eff = self.config.speculate_k if verify_k is None else int(verify_k)
            use_verify_graph = (not last_only) and (verify_k_eff + 1 == self.verify_k_plus_1)
            input_ids, positions = self.prepare_decode(
                seqs,
                verify=not last_only,
                verify_k=verify_k_eff,
                build_verify_cu_seqlens=not use_verify_graph,
            )
        temperatures, greedy = self.prepare_sample(seqs)

        logits = self.run_model(
            input_ids,
            positions,
            is_prefill,
            last_only,
            verify_query_len=(verify_k_eff + 1) if (not is_prefill and not last_only) else None,
        )

        if not last_only:
            reset_context()
            return logits

        token_ids = self.sampler(logits, temperatures, greedy)
        reset_context()
        token_out = token_ids if return_token_tensor else token_ids.tolist()
        return (token_out, logits) if draft_return_logits else token_out

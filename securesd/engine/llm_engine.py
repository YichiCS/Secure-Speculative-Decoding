import atexit
from time import perf_counter

from tqdm.auto import tqdm
from transformers import AutoTokenizer

from securesd.config import Config
from securesd.engine.draft_runner import DraftRunner
from securesd.engine.model_runner import ModelRunner
from securesd.engine.scheduler import Scheduler
from securesd.engine.sequence import Sequence
from securesd.engine.speculator_sync import SpeculatorSync
from securesd.engine.step import AutoRegressiveStep, InferenceStep, SpecDecodeStep
from securesd.engine.verifier import Verifier
from securesd.sampling_params import SamplingParams
from securesd.utils.misc import infer_model_family


def _new_metrics() -> dict:
    return {
        "accepted_suffix_lens_with_recovery": [],
        "prefill_total_time": 0,
        "decode_total_time": 0,
        "prefill_total_tokens": 0,
        "decode_total_tokens": 0,
        "decode_step_times": [],
        "decode_step_start_offsets_s": [],
        "decode_step_end_offsets_s": [],
        "decode_step_tokens": [],
        "decode_step_scheduled": [],
        "decode_step_seq_deltas": [],
        "per_seq_decode": [],
        "target_step_times": [],
        "target_verify_times": [],
        "distribution_diagnostics": [],
    }


def _build_per_seq_record(seq_id: int, start: float, first: float, end: float, tokens: int) -> dict:
    e2e = end - start
    decode = end - first
    decode_tokens = max(tokens - 1, 0)
    return {
        "seq_id": seq_id,
        "ttft_s": first - start,
        "e2e_latency_s": e2e,
        "decode_time_s": decode,
        "decode_tokens": tokens,
        "tpot_s": _safe_div(decode, decode_tokens) if decode_tokens > 0 else 0.0,
        "decode_tps": _safe_div(decode_tokens, decode),
        "full_span_tps": _safe_div(tokens, e2e),
    }


def _safe_div(numer: float, denom: float) -> float:
    return float(numer / denom) if denom > 0 else 0.0


class LLMEngine:
    def __init__(self, **kwargs):
        config = Config(**kwargs)
        self.config = config
        self.metrics = _new_metrics()
        self.draft_runner: DraftRunner | None = None
        self.draft_config: Config | None = None
        self._exiting = False

        Sequence.block_size = config.kvcache_block_size

        if config.kvcache_block_size < 2 * config.speculate_k + 2:
            raise ValueError(
                f"kvcache_block_size={config.kvcache_block_size} must be >= 2*speculate_k+2="
                f"{2 * config.speculate_k + 2}"
            )

        if config.speculate and infer_model_family(config.model) != infer_model_family(config.draft):
            raise ValueError(
                f"target/draft model family mismatch: target={config.model}, draft={config.draft}"
            )

        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id

        self.model_runner = ModelRunner(config=config)
        if config.speculate:
            self.draft_runner = DraftRunner(config=config)
            self.draft_config = self.draft_runner.draft_config

        self.scheduler = Scheduler(config, draft_config=self.draft_config)

        atexit.register(lambda: self.exit(hard=False))

    def exit(self, hard: bool = False):
        if self._exiting:
            return
        self._exiting = True
        self.model_runner.exit(hard)
        if self.draft_runner is not None:
            self.draft_runner.exit(hard=hard)

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        self.scheduler.add(Sequence(prompt, sampling_params))

    def is_finished(self) -> bool:
        return self.scheduler.is_finished()

    def step(self, step: InferenceStep, generate_started_at: float | None = None):
        t = perf_counter()
        seqs, is_prefill = self.scheduler.schedule()
        if not seqs:
            if self.scheduler.waiting and not self.scheduler.running:
                raise RuntimeError(
                    "Scheduler could not schedule any sequence. "
                    "Check KV cache capacity / block size / max_model_len settings."
                )
            return [], {
                "phase": "idle",
                "scheduled": 0,
                "scheduled_seq_ids": [],
                "tokens": 0,
                "finished": 0,
                "waiting": len(self.scheduler.waiting),
                "running": len(self.scheduler.running),
                "step_time_s": perf_counter() - t,
            }

        before_tokens = None if is_prefill else [seq.num_completion_tokens for seq in seqs]
        ttl_tokens = step.prefill(seqs) if is_prefill else step.decode(seqs)
        time_taken = perf_counter() - t

        if is_prefill:
            self.metrics["prefill_total_time"] += time_taken
            self.metrics["prefill_total_tokens"] += ttl_tokens
            seq_deltas = []
        else:
            self.metrics["decode_total_time"] += time_taken
            self.metrics["decode_total_tokens"] += ttl_tokens
            self.metrics["decode_step_times"].append(time_taken)
            if generate_started_at is None:
                self.metrics["decode_step_start_offsets_s"].append(0.0)
                self.metrics["decode_step_end_offsets_s"].append(time_taken)
            else:
                self.metrics["decode_step_start_offsets_s"].append(t - generate_started_at)
                self.metrics["decode_step_end_offsets_s"].append((t + time_taken) - generate_started_at)
            self.metrics["decode_step_tokens"].append(ttl_tokens)
            self.metrics["decode_step_scheduled"].append(len(seqs))
            seq_deltas = [
                (seq.seq_id, seq.num_completion_tokens - before)
                for seq, before in zip(seqs, before_tokens)
            ]
            self.metrics["decode_step_seq_deltas"].append(seq_deltas)

        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        step_info = {
            "phase": "prefill" if is_prefill else "decode",
            "scheduled": len(seqs),
            "scheduled_seq_ids": [seq.seq_id for seq in seqs],
            "seq_deltas": seq_deltas,
            "tokens": ttl_tokens,
            "finished": len(outputs),
            "waiting": len(self.scheduler.waiting),
            "running": len(self.scheduler.running),
            "step_time_s": time_taken,
        }
        return outputs, step_info

    def create_inference_step(self, config: Config) -> InferenceStep:
        if not config.speculate:
            return AutoRegressiveStep(scheduler=self.scheduler, model_runner=self.model_runner)

        speculator = SpeculatorSync(
            lookahead=config.speculate_k,
            device=config.device,
            draft_model_runner=self.draft_runner,
            method=config.method,
        )
        verifier = Verifier(
            lookahead=config.speculate_k,
            device=config.device,
            target_model_runner=self.model_runner,
            verify_method=config.method,
            method_params=config.method_params,
            eta_schedule=config.eta_schedule,
            eta_start=config.eta_start,
            eta_end=config.eta_end,
            eta_len=config.eta_len,
            eta_gamma=config.eta_gamma,
            record_distribution_diagnostics=config.record_distribution_diagnostics,
            metrics=self.metrics,
        )
        return SpecDecodeStep(scheduler=self.scheduler, speculator=speculator, verifier=verifier)

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> tuple[list[dict], dict]:
        self.metrics = _new_metrics()
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True) if use_tqdm else None

        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)

        outputs: dict[int, list[int]] = {}
        generate_started_at = perf_counter()
        seq_first_token_at: dict[int, float] = {}
        seq_decode_finished_at: dict[int, float] = {}
        seq_decode_tokens: dict[int, int] = {}
        inference_step = self.create_inference_step(self.config)
        max_steps = self.config.max_steps if self.config.max_steps is not None else float("inf")

        i = 0
        while not self.is_finished() and i < max_steps:
            i += 1
            t = perf_counter()
            step_outputs, step_info = self.step(inference_step, generate_started_at)
            time_taken = perf_counter() - t
            step_end = t + time_taken

            if step_info["phase"] != "idle":
                self.metrics["target_step_times"].append(time_taken)
            if step_info["phase"] == "decode":
                for seq_id, delta in step_info["seq_deltas"]:
                    if delta > 0 and seq_id not in seq_first_token_at:
                        seq_first_token_at[seq_id] = step_end

            for seq_id, token_ids in step_outputs:
                outputs[seq_id] = token_ids
                if step_info["phase"] == "decode":
                    seq_decode_finished_at[seq_id] = step_end
                    seq_decode_tokens[seq_id] = len(token_ids)
                if pbar is not None:
                    pbar.update(1)

        if not self.is_finished():
            raise RuntimeError(
                f"Generation terminated early at step {i} due to max_steps={max_steps}; "
                "some requests did not finish."
            )

        ordered_token_ids = [outputs[seq_id] for seq_id in sorted(outputs)]
        self.metrics["per_seq_decode"] = [
            _build_per_seq_record(
                seq_id,
                generate_started_at,
                seq_first_token_at[seq_id],
                seq_decode_finished_at[seq_id],
                seq_decode_tokens[seq_id],
            )
            for seq_id in sorted(seq_decode_finished_at)
            if seq_id in seq_first_token_at
        ]
        results = [
            {"text": self.tokenizer.decode(token_ids), "token_ids": token_ids}
            for token_ids in ordered_token_ids
        ]
        if pbar is not None:
            pbar.close()
        return results, self.metrics

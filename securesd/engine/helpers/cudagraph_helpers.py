from bisect import bisect_left

import torch

from securesd.utils.context import get_context, reset_context, set_context


def _build_graph_bs_list(max_bs: int) -> list[int]:
    if max_bs <= 0:
        return []
    buckets: set[int] = set()
    v = 1
    while v <= min(max_bs, 16):
        buckets.add(v)
        v *= 2
    buckets.update(range(24, max_bs, 8))
    buckets.add(max_bs)
    return sorted(b for b in buckets if 1 <= b <= max_bs)


def _pick_graph_bs(bs_list: list[int], want: int, kind: str) -> int:
    if not bs_list or want > bs_list[-1]:
        raise ValueError(
            f"{kind} batch size {want} exceeds captured CUDA-graph max batch "
            f"{bs_list[-1] if bs_list else 0}"
        )
    idx = bisect_left(bs_list, want)
    return bs_list[idx]


@torch.inference_mode()
def run_verify_cudagraph(model_runner, input_ids, positions, last_only, graph_vars):
    context = get_context()
    k_plus_1 = model_runner.verify_k_plus_1
    orig_bs = input_ids.size(0) // k_plus_1

    wrapper_bs = _pick_graph_bs(model_runner.graph_bs_list["verify"], orig_bs, "verify")
    graph = model_runner.graphs["verify"][wrapper_bs]
    flat = wrapper_bs * k_plus_1
    orig_flat = orig_bs * k_plus_1
    pad_bs = wrapper_bs - orig_bs
    pad_flat = flat - orig_flat

    graph_vars["input_ids"][:orig_flat] = input_ids
    graph_vars["positions"][:orig_flat] = positions
    graph_vars["slot_mapping"][:orig_flat] = context.slot_mapping
    graph_vars["context_lens"][:orig_bs] = context.context_lens

    if pad_flat > 0:
        graph_vars["input_ids"][orig_flat:flat].zero_()
        graph_vars["positions"][orig_flat:flat].zero_()
        graph_vars["slot_mapping"][orig_flat:flat].fill_(-1)
    if pad_bs > 0:
        graph_vars["context_lens"][orig_bs:wrapper_bs] = context.context_lens[orig_bs - 1:orig_bs].expand(pad_bs)

    cu = graph_vars["cu_seqlens_q"][:wrapper_bs + 1]
    cu.copy_(graph_vars["cu_seqlens_q_template"][:wrapper_bs + 1])

    if context.block_tables is not None:
        bt = graph_vars["block_tables"][:wrapper_bs]
        bt.zero_()
        width = context.block_tables.size(1)
        bt[:orig_bs, :width] = context.block_tables
        if pad_bs > 0:
            bt[orig_bs:wrapper_bs, :width] = context.block_tables[orig_bs - 1:orig_bs].expand(pad_bs, -1)

    graph.replay()

    outputs = graph_vars["outputs"][:orig_bs * k_plus_1]
    return model_runner.model.compute_logits(outputs, last_only)


@torch.inference_mode()
def run_decode_cudagraph(model_runner, input_ids, positions, last_only, graph_vars):
    context = get_context()
    flat_batch_size = input_ids.size(0)

    wrapper_bs = _pick_graph_bs(model_runner.graph_bs_list["decode"], flat_batch_size, "decode")
    graph = model_runner.graphs["decode"][wrapper_bs]
    pad = wrapper_bs - flat_batch_size

    graph_vars["input_ids"][:flat_batch_size] = input_ids
    graph_vars["positions"][:flat_batch_size] = positions
    graph_vars["slot_mapping"][:flat_batch_size] = context.slot_mapping
    graph_vars["context_lens"][:flat_batch_size] = context.context_lens
    if pad > 0:
        graph_vars["input_ids"][flat_batch_size:wrapper_bs].zero_()
        graph_vars["positions"][flat_batch_size:wrapper_bs].zero_()
        graph_vars["slot_mapping"][flat_batch_size:wrapper_bs].fill_(-1)
        graph_vars["context_lens"][flat_batch_size:wrapper_bs].zero_()

    if context.block_tables is not None:
        bt = graph_vars["block_tables"][:wrapper_bs]
        bt.zero_()
        bt[:flat_batch_size, :context.block_tables.size(1)] = context.block_tables

    graph.replay()

    outputs = graph_vars["outputs"][:flat_batch_size]
    return model_runner.model.compute_logits(outputs, last_only)


@torch.inference_mode()
def capture_cudagraph(model_runner):
    config = model_runner.config
    hf_config = config.hf_config
    max_bs = min(config.max_num_seqs, 512) + 1
    max_num_blocks = (config.max_model_len + model_runner.block_size - 1) // model_runner.block_size

    input_ids = torch.zeros(max_bs, dtype=torch.int64)
    positions = torch.zeros(max_bs, dtype=torch.int64)
    slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
    context_lens = torch.zeros(max_bs, dtype=torch.int32)
    block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
    outputs = torch.zeros(max_bs, hf_config.hidden_size)

    graph_bs_list = _build_graph_bs_list(max_bs)
    graphs: dict[int, torch.cuda.CUDAGraph] = {}
    graph_pool = None

    for bs in reversed(graph_bs_list):
        graph = torch.cuda.CUDAGraph()
        set_context(
            False,
            slot_mapping=slot_mapping[:bs],
            context_lens=context_lens[:bs],
            block_tables=block_tables[:bs],
        )

        outputs[:bs] = model_runner.model(input_ids[:bs], positions[:bs])

        with torch.cuda.graph(graph, graph_pool):
            outputs[:bs] = model_runner.model(input_ids[:bs], positions[:bs])

        if graph_pool is None:
            graph_pool = graph.pool()
        graphs[bs] = graph
        torch.cuda.synchronize()
        reset_context()

    graph_vars = {
        "input_ids": input_ids,
        "positions": positions,
        "slot_mapping": slot_mapping,
        "context_lens": context_lens,
        "block_tables": block_tables,
        "outputs": outputs,
    }
    return graph_vars, graph_pool, graphs, graph_bs_list


@torch.inference_mode()
def capture_verify_cudagraph(model_runner):
    config = model_runner.config
    hf_config = config.hf_config
    max_bs = min(config.max_num_seqs, 512)
    k_plus_1 = model_runner.verify_k_plus_1

    input_ids = torch.zeros(max_bs * k_plus_1, dtype=torch.int64)
    positions = torch.zeros(max_bs * k_plus_1, dtype=torch.int64)
    slot_mapping = torch.zeros(max_bs * k_plus_1, dtype=torch.int32)
    context_lens = torch.zeros(max_bs, dtype=torch.int32)
    block_tables = torch.zeros(max_bs, model_runner.max_num_blocks, dtype=torch.int32)
    outputs = torch.zeros(max_bs * k_plus_1, hf_config.hidden_size)
    cu_seqlens_q = torch.zeros(max_bs + 1, dtype=torch.int32)

    graph_bs_list = _build_graph_bs_list(max_bs)
    graphs: dict[int, torch.cuda.CUDAGraph] = {}
    graph_pool = None

    for bs in reversed(graph_bs_list):
        graph = torch.cuda.CUDAGraph()
        seqlen_q = torch.full((bs,), k_plus_1, dtype=torch.int32)
        cu = cu_seqlens_q[:bs + 1]
        cu.zero_()
        cu[1:].copy_(torch.cumsum(seqlen_q, 0))
        context_lens[:bs] = seqlen_q

        set_context(
            is_prefill=False,
            slot_mapping=slot_mapping[:bs * k_plus_1],
            context_lens=context_lens[:bs],
            block_tables=block_tables[:bs],
            cu_seqlens_q=cu,
            max_seqlen_q=k_plus_1,
        )

        outputs[:bs * k_plus_1] = model_runner.model(input_ids[:bs * k_plus_1], positions[:bs * k_plus_1])

        with torch.cuda.graph(graph, graph_pool):
            outputs[:bs * k_plus_1] = model_runner.model(input_ids[:bs * k_plus_1], positions[:bs * k_plus_1])

        if graph_pool is None:
            graph_pool = graph.pool()
        graphs[bs] = graph
        torch.cuda.synchronize()
        reset_context()

    graph_vars = {
        "input_ids": input_ids,
        "positions": positions,
        "slot_mapping": slot_mapping,
        "context_lens": context_lens,
        "block_tables": block_tables,
        "cu_seqlens_q": cu_seqlens_q,
        "cu_seqlens_q_template": torch.arange(
            0,
            (max_bs + 1) * k_plus_1,
            k_plus_1,
            dtype=torch.int32,
            device=cu_seqlens_q.device,
        ),
        "outputs": outputs,
    }
    return graph_vars, graph_pool, graphs, graph_bs_list

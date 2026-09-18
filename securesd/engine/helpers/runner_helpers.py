import torch

from securesd.engine.sequence import Sequence

def _to_cuda(data: list[int], dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor(data, dtype=dtype, pin_memory=True).cuda(non_blocking=True)


def prepare_decode_packed_from_seqs(
    seqs: list[Sequence],
    block_size: int,
    is_draft: bool,
    verify: bool = False,
    k: int = -1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    input_ids: list[int] = []
    positions: list[int] = []
    slot_mapping: list[int] = []
    context_lens: list[int] = []

    if not verify:
        if k != -1:
            raise ValueError("k must be -1 for non-verify decode prep")
        for seq in seqs:
            block_table = seq.draft_block_table if is_draft else seq.block_table
            if len(seq) // block_size > len(block_table):
                raise RuntimeError("sync-spec draft decode: not enough blocks allocated")
            num_cached = seq.num_draft_cached_tokens if is_draft else seq.num_cached_tokens
            if num_cached != len(seq) - 1:
                raise RuntimeError(
                    f"num_cached_tokens={num_cached} must equal len(seq)-1 in single-query decode"
                )
            input_ids.append(seq.last_token)
            pos = seq.num_tokens - 1
            positions.append(pos)
            context_lens.append(len(seq))
            slot_mapping.append(block_table[pos // block_size] * block_size + pos % block_size)
    else:
        if is_draft:
            raise ValueError("verify path is only supported on target model")
        if k <= 0:
            raise ValueError(f"k must be > 0 for verify prep, got {k}")
        for seq in seqs:
            if (seq.num_tokens - 1) // block_size > len(seq.block_table):
                raise RuntimeError("sync-spec target verify: not enough blocks allocated")
            pos0 = seq.num_tokens - (k + 1)
            if seq.num_cached_tokens != pos0:
                raise RuntimeError(
                    f"num_cached_tokens={seq.num_cached_tokens} != pos0={pos0} "
                    f"(num_tokens={seq.num_tokens}, k={k})"
                )
            input_ids.extend(seq[pos0:])
            positions.extend(range(pos0, pos0 + k + 1))
            context_lens.append(len(seq))
            for j in range(k + 1):
                pos = pos0 + j
                slot_mapping.append(seq.block_table[pos // block_size] * block_size + pos % block_size)

    tables = [seq.draft_block_table if is_draft else seq.block_table for seq in seqs]
    max_len = max(len(t) for t in tables)
    padded = [t + [-1] * (max_len - len(t)) for t in tables]
    block_flat: list[int] = []
    for row in padded:
        block_flat.extend(row)

    n_tok = len(input_ids)
    if len(positions) != n_tok:
        raise RuntimeError("internal: input_ids and positions length mismatch")
    n_slot = len(slot_mapping)
    n_ctx = len(context_lens)
    if n_ctx != len(seqs):
        raise RuntimeError("internal: context_lens batch mismatch")
    if n_slot != n_tok:
        raise RuntimeError("internal: slot_mapping vs token row length mismatch")
    i64_buf = input_ids + positions
    dev_i64 = torch.tensor(i64_buf, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
    input_ids_cuda = dev_i64[:n_tok]
    positions_cuda = dev_i64[n_tok : 2 * n_tok]

    i32_buf = slot_mapping + context_lens + block_flat
    dev_i32 = torch.tensor(i32_buf, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
    slot_cuda = dev_i32[:n_slot]
    context_cuda = dev_i32[n_slot : n_slot + n_ctx]
    block_cuda = dev_i32[n_slot + n_ctx :].view(n_ctx, max_len)
    return input_ids_cuda, positions_cuda, slot_cuda, context_cuda, block_cuda


def prepare_block_tables_from_seqs(
    seqs: list[Sequence],
    is_draft: bool = False,
) -> torch.Tensor:
    tables = [seq.draft_block_table if is_draft else seq.block_table for seq in seqs]
    max_len = max(len(t) for t in tables)
    padded = [t + [-1] * (max_len - len(t)) for t in tables]
    return torch.tensor(padded, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)


def prepare_prefill_tensors_from_seqs(
    seqs: list[Sequence],
    block_size: int,
    is_draft: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int, torch.Tensor]:
    input_ids: list[int] = []
    positions: list[int] = []
    cu_seqlens_q: list[int] = [0]
    cu_seqlens_k: list[int] = [0]
    max_seqlen_q = 0
    max_seqlen_k = 0
    slot_mapping: list[int] = []

    for seq in seqs:
        seqlen = len(seq)
        if is_draft:
            start = seq.num_draft_cached_tokens
            block_table = seq.draft_block_table
        else:
            start = seq.num_cached_tokens
            block_table = seq.block_table

        input_ids.extend(seq[start:])
        positions.extend(range(start, seqlen))
        seqlen_q = seqlen - start
        cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
        cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen)
        max_seqlen_q = max(seqlen_q, max_seqlen_q)
        max_seqlen_k = max(seqlen, max_seqlen_k)

        if not block_table:
            continue
        for pos in range(start, seq.num_tokens):
            slot_mapping.append(block_table[pos // block_size] * block_size + pos % block_size)

    return (
        _to_cuda(input_ids, torch.int64),
        _to_cuda(positions, torch.int64),
        _to_cuda(cu_seqlens_q, torch.int32),
        _to_cuda(cu_seqlens_k, torch.int32),
        max_seqlen_q,
        max_seqlen_k,
        _to_cuda(slot_mapping, torch.int32),
    )

import math

import torch

from .common import finalize_native
from .registry import register
from .types import VerifierMethodOutput, VerifierMethodRequest


def _normalized_entropy(scaled: torch.Tensor, vocab: int) -> torch.Tensor:
    max_v = scaled.amax(dim=-1, keepdim=True)
    exp_c = (scaled - max_v).exp()
    sum_exp = exp_c.sum(dim=-1)
    probs = exp_c / sum_exp.unsqueeze(-1)
    log_z = max_v.squeeze(-1) + sum_exp.log()
    return (log_z - (probs * scaled).sum(dim=-1)) / math.log(vocab)


@register("fly")
def verify_fly(
    request: VerifierMethodRequest,
    probs_only: bool = False,
    all_greedy: bool | None = None,
) -> VerifierMethodOutput:
    logits_p = request.logits_p
    theta = request.param("fly_entropy_threshold")
    window = request.param("fly_window_size")

    device = logits_p.device
    B, _, V = logits_p.shape
    K = request.num_spec
    temps_t = request.temperatures_target.to(device=device, dtype=torch.float32)

    step_logits = logits_p[:, :K, :]
    draft_tokens = request.speculations[:, 1:]
    mismatches = draft_tokens != step_logits.argmax(dim=-1)

    needs_entropy = window <= K
    entropies = torch.zeros((B, K), dtype=torch.float32, device=device)
    if needs_entropy:
        mismatch_idx = mismatches.nonzero(as_tuple=False)
        if mismatch_idx.numel() > 0:
            rows, cols = mismatch_idx[:, 0], mismatch_idx[:, 1]
            sub_temps = temps_t.index_select(0, rows)
            scales = torch.where(sub_temps > 0, sub_temps, torch.ones_like(sub_temps))
            scaled = step_logits[rows, cols, :].to(torch.float32) / scales.view(-1, 1)
            entropies[rows, cols] = _normalized_entropy(scaled, V)

    steps = torch.arange(K, device=device, dtype=torch.long).expand(B, -1)
    mismatch_pos = torch.where(mismatches, steps, torch.full_like(steps, K))
    future_mismatch = torch.full_like(mismatch_pos, K)
    if K > 1:
        future_mismatch[:, :-1] = torch.flip(
            torch.cummin(torch.flip(mismatch_pos[:, 1:], dims=[1]), dim=1).values,
            dims=[1],
        )

    window_branch = ((steps + window) > K) | (
        (future_mismatch < K) & (future_mismatch <= (steps + window))
    )
    reject_mask = mismatches & (
        ((entropies < theta) | window_branch) if needs_entropy else window_branch
    )
    accept_probs = (~reject_mask).to(torch.float32)

    if probs_only:
        sd_probs = (~mismatches).to(torch.float32) if all_greedy else None
        return VerifierMethodOutput([], [], accept_probs, sd_accept_probs=sd_probs)

    return finalize_native(request, accept_probs, all_greedy=all_greedy)

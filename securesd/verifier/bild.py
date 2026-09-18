import math

import torch

from .common import finalize_native, greedy_sd_accept_probs
from .registry import register
from .types import VerifierMethodOutput, VerifierMethodRequest


@register("bild")
def verify_bild(
    request: VerifierMethodRequest,
    probs_only: bool = False,
    all_greedy: bool | None = None,
) -> VerifierMethodOutput:
    logits_p = request.logits_p
    logits_q = request.logits_q
    fb = float(request.param("bild_fallback_threshold"))
    rb = float(request.param("bild_rollback_threshold"))

    device = logits_p.device
    B = logits_p.shape[0]
    K = request.num_spec
    draft_tokens = request.speculations[:, 1:]
    batch_idx = torch.arange(B, device=device)

    if fb > 0:
        below_fb = (logits_q.amax(dim=2) - torch.logsumexp(logits_q, dim=2)) < math.log(fb)
    else:
        below_fb = torch.zeros(logits_q.shape[:2], dtype=torch.bool, device=device)
    first_fb = below_fb.int().argmax(dim=1)
    fallback_idx = torch.where(below_fb.any(dim=1), first_fb, torch.full_like(first_fb, K))

    logits_p_steps = logits_p[:, :K, :]
    logp_at_draft = logits_p_steps.gather(2, draft_tokens.unsqueeze(2)).squeeze(2)
    above_rb = (logp_at_draft - torch.logsumexp(logits_p_steps, dim=2)) < -rb
    first_rb = above_rb.int().argmax(dim=1)

    preds_p_all = logits_p.argmax(dim=-1)
    rollback_active = above_rb.any(dim=1) & (
        preds_p_all[batch_idx, first_rb] != draft_tokens[batch_idx, first_rb]
    )
    rollback_idx = torch.where(rollback_active, first_rb, torch.full_like(first_rb, K))

    accept_until = torch.minimum(fallback_idx, rollback_idx)

    positions = torch.arange(K, device=device).unsqueeze(0)
    accept_probs = (positions < accept_until.unsqueeze(1)).to(torch.float32)

    if probs_only:
        sd_probs = (
            greedy_sd_accept_probs(logits_p, request.speculations) if all_greedy else None
        )
        return VerifierMethodOutput([], [], accept_probs, sd_accept_probs=sd_probs)

    return finalize_native(request, accept_probs, all_greedy=all_greedy)

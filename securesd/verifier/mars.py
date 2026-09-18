import torch

from .common import finalize_native
from .registry import register
from .types import VerifierMethodOutput, VerifierMethodRequest


@register("mars")
def verify_mars(
    request: VerifierMethodRequest,
    probs_only: bool = False,
    all_greedy: bool | None = None,
) -> VerifierMethodOutput:
    logits_p = request.logits_p
    K = request.num_spec

    top2_vals, top2_idx = torch.topk(logits_p[:, :K, :], k=2, dim=2)
    top1_idx, top2_idx = top2_idx[:, :, 0], top2_idx[:, :, 1]
    top1_vals = top2_vals[:, :, 0].to(torch.float32)
    top2_vals = top2_vals[:, :, 1].to(torch.float32)

    draft_tokens = request.speculations[:, 1:]
    is_top1 = draft_tokens == top1_idx
    accepts = is_top1 | (
        (draft_tokens == top2_idx) & ((top2_vals / top1_vals) > request.param("mars_theta"))
    )

    if probs_only:
        sd_probs = is_top1.to(torch.float32) if all_greedy else None
        return VerifierMethodOutput(
            [], [], accepts.to(torch.float32), sd_accept_probs=sd_probs
        )

    return finalize_native(request, accepts.to(torch.float32), all_greedy=all_greedy)

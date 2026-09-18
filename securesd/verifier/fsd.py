import math

import torch

from .common import finalize_native
from .registry import register
from .types import VerifierMethodOutput, VerifierMethodRequest


def _divergences(
    logits_p: torch.Tensor,
    logits_q: torch.Tensor,
    div_type: str,
) -> torch.Tensor:
    if div_type == "tv_div":
        p = torch.softmax(logits_p, dim=-1, dtype=torch.float32)
        q = torch.softmax(logits_q, dim=-1, dtype=torch.float32)
        return 0.5 * torch.abs(p - q).sum(dim=-1)

    logp = torch.log_softmax(logits_p, dim=-1, dtype=torch.float32)
    logq = torch.log_softmax(logits_q, dim=-1, dtype=torch.float32)
    p = logp.exp()
    if div_type == "kl_div":
        return (p * (logp - logq)).sum(dim=-1)
    if div_type == "js_div":
        q = logq.exp()
        logm = torch.logaddexp(logp, logq) - math.log(2.0)
        return 0.5 * (
            (p * (logp - logm)).sum(dim=-1)
            + (q * (logq - logm)).sum(dim=-1)
        )
    raise ValueError(f"Unsupported fsd_div_type: {div_type}")


@register("fsd")
def verify_fsd(
    request: VerifierMethodRequest,
    probs_only: bool = False,
    all_greedy: bool | None = None,
) -> VerifierMethodOutput:
    K = request.num_spec
    divs = _divergences(
        request.logits_p[:, :K, :], request.logits_q, request.param("fsd_div_type")
    )
    accept_probs = (~(divs > request.param("fsd_threshold"))).to(torch.float32)

    if probs_only:
        return VerifierMethodOutput([], [], accept_probs)
    return finalize_native(request, accept_probs, all_greedy=all_greedy)

import torch

from securesd.methods import get_spec

from .common import (
    alpha_sd_for,
    materialize_outputs,
    mixed_accept_until,
    temp_aware_probs,
)
from .types import VerifierMethodOutput, VerifierMethodRequest


def _fused_recovery(
    request: VerifierMethodRequest,
    accept_until: torch.Tensor,
    recovery_kind: str,
    all_greedy: bool,
) -> torch.Tensor:
    logits_p = request.logits_p
    B = logits_p.shape[0]
    K = request.num_spec
    device = logits_p.device
    batch = torch.arange(B, device=device)
    sel_p_logits = logits_p[batch, accept_until, :]

    if all_greedy:
        return sel_p_logits.argmax(dim=-1)

    temps_t = request.temperatures_target.to(device=device, dtype=torch.float32)
    temps_q = request.temperatures_draft.to(device=device, dtype=torch.float32)
    all_positive = request.temp_mode()[1]
    p_target = temp_aware_probs(sel_p_logits, temps_t, all_positive)

    if recovery_kind == "sd":
        need_sd, need_method = True, False
    else:
        use_method = torch.rand(B, device=device) < request.eta[batch, accept_until]
        n_method = int(use_method.sum())
        need_sd, need_method = n_method < B, n_method > 0

    r_sd = None
    if need_sd:
        r_sd = torch.multinomial(p_target, 1).squeeze(1)
        reject = accept_until < K
        if reject.any():
            idx = reject.nonzero(as_tuple=False).squeeze(1)
            q_at = temp_aware_probs(
                request.logits_q[idx, accept_until[idx], :], temps_q[idx], all_positive
            )
            resid = torch.clamp(p_target[idx] - q_at, min=0.0)
            resid_sum = resid.sum(dim=1, keepdim=True)
            safe = torch.where(
                resid_sum > 0, resid / resid_sum.clamp_min(1e-12), p_target[idx]
            )
            r_sd[idx] = torch.multinomial(safe, 1).squeeze(1)

    if not need_method:
        return r_sd

    if recovery_kind == "argmax":
        r_method = sel_p_logits.argmax(dim=-1)
    elif recovery_kind == "target":
        r_method = torch.multinomial(p_target, 1).squeeze(1)
    else:
        raise ValueError(f"Unsupported recovery_kind: {recovery_kind}")

    return r_method if not need_sd else torch.where(use_method, r_method, r_sd)


def apply_eta_fusion(
    request: VerifierMethodRequest,
    native: VerifierMethodOutput,
    all_greedy: bool,
) -> VerifierMethodOutput:
    if request.eta is None or native.accept_probs is None:
        return native
    K = request.num_spec

    alpha_sd = native.sd_accept_probs
    if alpha_sd is None:
        alpha_sd = alpha_sd_for(request, all_greedy)

    accept_until = mixed_accept_until(native.accept_probs, alpha_sd, request.eta[:, :K])
    recovery = _fused_recovery(
        request, accept_until, get_spec(request.method).recovery_kind, all_greedy
    )
    new_suffixes, recovery_tokens = materialize_outputs(
        request.speculations, accept_until, recovery
    )
    return VerifierMethodOutput(
        new_suffixes=new_suffixes,
        recovery_tokens=recovery_tokens,
        accept_probs=native.accept_probs,
    )

import torch

from .common import materialize_outputs, first_reject_index, scaled_logits, temp_aware_probs
from .registry import register
from .types import VerifierMethodOutput, VerifierMethodRequest


def _token_and_max_probs(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    temperatures: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    temps = temperatures.to(device=logits.device, dtype=torch.float32)
    scaled = scaled_logits(logits, temps)
    lse = torch.logsumexp(scaled, dim=-1)
    p_softmax_tok = (scaled.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1) - lse).exp()
    max_softmax = (scaled.amax(dim=-1) - lse).exp()

    pos_mask = (temps > 0).view((-1,) + (1,) * (token_ids.ndim - 1))
    p_argmax_tok = (token_ids == logits.argmax(dim=-1)).to(torch.float32)
    return torch.where(pos_mask, p_softmax_tok, p_argmax_tok), max_softmax


def _max_probs(logits: torch.Tensor, temperatures: torch.Tensor) -> torch.Tensor:
    scaled = scaled_logits(logits, temperatures)
    return (scaled.amax(dim=-1) - torch.logsumexp(scaled, dim=-1)).exp()


def _defer(
    max_p: torch.Tensor,
    max_q: torch.Tensor,
    logits_p: torch.Tensor,
    logits_q: torch.Tensor,
    temps_t: torch.Tensor,
    temps_q: torch.Tensor,
    alpha: float,
    rule: str,
) -> torch.Tensor:
    if rule == "chow":
        return max_q < (1.0 - alpha)
    if rule == "diff":
        return max_q < (max_p - alpha)
    if rule == "opt":
        p_probs = torch.softmax(scaled_logits(logits_p, temps_t), dim=-1)
        q_probs = torch.softmax(scaled_logits(logits_q, temps_q), dim=-1)
        tv = torch.clamp(p_probs - q_probs, min=0.0).sum(dim=-1)
        return max_q < (max_p - alpha * tv)
    raise ValueError(f"Unsupported sc_rule: {rule}")


@register("sc")
def verify_sc(
    request: VerifierMethodRequest,
    probs_only: bool = False,
    all_greedy: bool | None = None,
) -> VerifierMethodOutput:
    logits_p = request.logits_p
    logits_q = request.logits_q
    logits_q_next = request.logits_q_next
    speculations = request.speculations
    alpha = request.param("sc_alpha")
    rule = request.param("sc_rule")

    device = logits_p.device
    B = logits_p.shape[0]
    K = request.num_spec
    if logits_q_next is None:
        raise ValueError("sc verifier requires logits_q_next")

    temps_t = request.temperatures_target.to(device=device, dtype=torch.float32)
    temps_q = request.temperatures_draft.to(device=device, dtype=torch.float32)
    all_positive = request.temp_mode()[1]
    draft_tokens = speculations[:, 1:]
    logits_p_steps = logits_p[:, :K, :]

    q_tok, max_q = _token_and_max_probs(logits_q, draft_tokens, temps_q)
    p_tok, max_p = _token_and_max_probs(logits_p_steps, draft_tokens, temps_t)

    delta_steps = _defer(
        max_p, max_q, logits_p_steps, logits_q, temps_t, temps_q, alpha, rule,
    ).to(torch.float32)
    pi_tok = q_tok + delta_steps * (p_tok - q_tok)

    if (q_tok <= 0).any():
        raise ValueError(
            "sc verifier encountered a draft token with zero q-probability; "
            "the speculation is inconsistent with the draft distribution"
        )

    accept_probs = torch.minimum(torch.ones_like(pi_tok), pi_tok / q_tok)
    sd_probs = torch.minimum(torch.ones_like(p_tok), p_tok / q_tok)

    if probs_only:
        return VerifierMethodOutput([], [], accept_probs, sd_accept_probs=sd_probs)

    accept_until = first_reject_index(torch.rand_like(accept_probs) <= accept_probs)
    recovery = torch.empty(B, dtype=torch.long, device=device)
    accept_all = accept_until == K

    if accept_all.any():
        idx = torch.nonzero(accept_all, as_tuple=False).squeeze(1)
        p_next_logits = logits_p[idx, K, :]
        q_next_logits = logits_q_next[idx]
        p_next = temp_aware_probs(p_next_logits, temps_t[idx], all_positive)
        q_next = temp_aware_probs(q_next_logits, temps_q[idx], all_positive)
        delta_next = _defer(
            _max_probs(p_next_logits, temps_t[idx]),
            _max_probs(q_next_logits, temps_q[idx]),
            p_next_logits, q_next_logits, temps_t[idx], temps_q[idx], alpha, rule,
        ).to(torch.float32)
        pi_next = q_next + delta_next.unsqueeze(1) * (p_next - q_next)
        recovery[idx] = torch.multinomial(pi_next, 1).squeeze(1)

    if (~accept_all).any():
        idx = torch.nonzero(~accept_all, as_tuple=False).squeeze(1)
        rej_pos = accept_until[idx]
        p_sel = temp_aware_probs(logits_p_steps[idx, rej_pos, :], temps_t[idx], all_positive)
        q_sel = temp_aware_probs(logits_q[idx, rej_pos, :], temps_q[idx], all_positive)
        pi_sel = q_sel + delta_steps[idx, rej_pos].unsqueeze(1) * (p_sel - q_sel)
        residual = torch.clamp(pi_sel - q_sel, min=0.0)
        total = residual.sum(dim=1, keepdim=True)
        if (total <= 0).any():
            raise ValueError(
                "sc verifier encountered a zero-mass residual normalize((pi-q)_+), "
                "which is inconsistent with an Algorithm 4 rejection."
            )
        recovery[idx] = torch.multinomial(residual / total, 1).squeeze(1)

    new_suffixes, recovery_tokens = materialize_outputs(
        speculations, accept_until, recovery
    )
    return VerifierMethodOutput(
        new_suffixes, recovery_tokens, accept_probs, sd_accept_probs=sd_probs
    )

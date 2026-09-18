import torch

from securesd.methods import get_spec

from .registry import register
from .types import VerifierMethodOutput, VerifierMethodRequest


def scaled_logits(logits: torch.Tensor, temps: torch.Tensor) -> torch.Tensor:
    temps = temps.to(device=logits.device, dtype=torch.float32)
    safe = torch.where(temps > 0, temps, torch.ones_like(temps))
    return logits.to(torch.float32) / safe.view((-1,) + (1,) * (logits.ndim - 1))


def temp_aware_logprobs(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    temps: torch.Tensor,
    all_positive: bool | None = None,
) -> torch.Tensor:
    temps = temps.to(device=logits.device, dtype=torch.float32)
    scaled = scaled_logits(logits, temps)
    soft = scaled.gather(2, token_ids.unsqueeze(2)).squeeze(2) - torch.logsumexp(scaled, dim=2)
    if all_positive is None:
        all_positive = bool((temps > 0).all())
    if all_positive:
        return soft
    greedy = torch.where(token_ids == logits.argmax(dim=2), 0.0, -torch.inf).to(torch.float32)
    return torch.where((temps > 0).view(-1, 1), soft, greedy)


def temp_aware_probs(
    logits: torch.Tensor,
    temps: torch.Tensor,
    all_positive: bool | None = None,
) -> torch.Tensor:
    temps = temps.to(device=logits.device, dtype=torch.float32)
    soft = torch.softmax(scaled_logits(logits, temps), dim=-1)
    if all_positive is None:
        all_positive = bool((temps > 0).all())
    if all_positive:
        return soft
    onehot = torch.zeros_like(soft)
    onehot.scatter_(-1, logits.argmax(dim=-1, keepdim=True), 1.0)
    return torch.where((temps > 0).view((-1,) + (1,) * (logits.ndim - 1)), soft, onehot)


def materialize_outputs(
    speculations: torch.Tensor,
    accept_until: torch.Tensor,
    recovery_tokens: torch.Tensor,
) -> tuple[list[list[int]], list[int]]:
    pack = torch.cat(
        [accept_until.unsqueeze(1), recovery_tokens.unsqueeze(1), speculations],
        dim=1,
    )
    rows = pack.tolist()
    return (
        [row[2 : 3 + row[0]] for row in rows],
        [row[1] for row in rows],
    )


def first_reject_index(accepts: torch.Tensor) -> torch.Tensor:
    rejects = ~accepts
    K = accepts.shape[1]
    first = rejects.int().argmax(dim=1)
    return torch.where(rejects.any(dim=1), first, torch.full_like(first, K))


def resolve_temp_mode(request: VerifierMethodRequest) -> tuple[bool, bool]:
    tt = request.temperatures_target
    tq = request.temperatures_draft
    greedy, positive = torch.stack([
        (tt == 0).all() & (tq == 0).all(),
        (tt > 0).all() & (tq > 0).all(),
    ]).tolist()
    return bool(greedy), bool(positive)


def resolve_all_greedy(request: VerifierMethodRequest, all_greedy: bool | None) -> bool:
    if all_greedy is not None:
        return all_greedy
    return resolve_temp_mode(request)[0]


def greedy_sd_accept_probs(
    logits_p: torch.Tensor,
    speculations: torch.Tensor,
) -> torch.Tensor:
    K = logits_p.shape[1] - 1
    return (logits_p[:, :K, :].argmax(dim=2) == speculations[:, 1:]).to(torch.float32)


def sd_accept_probs(
    logits_p_steps: torch.Tensor,
    logits_q: torch.Tensor,
    draft_tokens: torch.Tensor,
    temperatures_target: torch.Tensor,
    temperatures_draft: torch.Tensor,
    all_positive: bool | None = None,
) -> torch.Tensor:
    logp = temp_aware_logprobs(logits_p_steps, draft_tokens, temperatures_target, all_positive)
    logq = temp_aware_logprobs(logits_q, draft_tokens, temperatures_draft, all_positive)
    return logp.sub(logq).exp_().clamp_(max=1.0)


def alpha_sd_for(request: VerifierMethodRequest, all_greedy: bool) -> torch.Tensor:
    if all_greedy:
        return greedy_sd_accept_probs(request.logits_p, request.speculations)
    K = request.num_spec
    return sd_accept_probs(
        request.logits_p[:, :K, :], request.logits_q, request.speculations[:, 1:],
        request.temperatures_target, request.temperatures_draft,
        all_positive=request.temp_mode()[1],
    )


def mixed_accept_until(
    alpha_method: torch.Tensor,
    alpha_sd: torch.Tensor,
    eta_steps: torch.Tensor,
) -> torch.Tensor:
    alpha = alpha_sd.add(eta_steps.mul(alpha_method.sub(alpha_sd)))
    return first_reject_index(torch.rand_like(alpha) <= alpha)


def finalize_native(
    request: VerifierMethodRequest,
    accept_probs: torch.Tensor,
    all_greedy: bool | None = None,
) -> VerifierMethodOutput:
    recovery_kind = get_spec(request.method).recovery_kind
    logits_p = request.logits_p
    device = logits_p.device
    B = logits_p.shape[0]

    accept_until = first_reject_index(accept_probs != 0)
    sel = logits_p[torch.arange(B, device=device), accept_until, :]

    if recovery_kind == "argmax" or resolve_all_greedy(request, all_greedy):
        recovery = sel.argmax(dim=-1)
    elif recovery_kind == "target":
        temps_t = request.temperatures_target.to(device=device, dtype=torch.float32)
        recovery = torch.multinomial(
            temp_aware_probs(sel, temps_t, request.temp_mode()[1]), 1
        ).squeeze(1)
    else:
        raise ValueError(
            f"finalize_native: unsupported recovery_kind {recovery_kind!r}"
        )

    new_suffixes, recovery_tokens = materialize_outputs(
        request.speculations, accept_until, recovery
    )
    return VerifierMethodOutput(
        new_suffixes=new_suffixes,
        recovery_tokens=recovery_tokens,
        accept_probs=accept_probs,
    )


def _sd_family(
    request: VerifierMethodRequest,
    epsilon: float,
    probs_only: bool,
    all_greedy: bool | None,
) -> VerifierMethodOutput:
    logits_p = request.logits_p
    speculations = request.speculations
    device = logits_p.device
    K = request.num_spec
    all_greedy = resolve_all_greedy(request, all_greedy)

    if all_greedy:
        target_argmax = logits_p.argmax(dim=2)
        sd_probs = (target_argmax[:, :K] == speculations[:, 1:]).to(torch.float32)
        accept_probs = sd_probs.add(epsilon).clamp_(max=1.0) if epsilon else sd_probs
        if probs_only:
            return VerifierMethodOutput([], [], accept_probs, sd_accept_probs=sd_probs)
        accepts = (
            torch.rand_like(accept_probs) <= accept_probs if epsilon else sd_probs > 0
        )
        accept_until = first_reject_index(accepts)
        recovery = target_argmax.gather(1, accept_until.unsqueeze(1)).squeeze(1)
        new_suffixes, recovery_tokens = materialize_outputs(
            speculations, accept_until, recovery
        )
        return VerifierMethodOutput(
            new_suffixes, recovery_tokens, accept_probs, sd_accept_probs=sd_probs
        )

    logits_q = request.logits_q
    temps_t = request.temperatures_target.to(device=device, dtype=torch.float32)
    temps_q = request.temperatures_draft.to(device=device, dtype=torch.float32)
    draft_tokens = speculations[:, 1:]

    all_positive = request.temp_mode()[1]
    logp = temp_aware_logprobs(logits_p[:, :K, :], draft_tokens, temps_t, all_positive)
    logq = temp_aware_logprobs(logits_q, draft_tokens, temps_q, all_positive)
    ratio = logp.sub(logq).exp_()
    sd_probs = ratio.clamp(max=1.0)
    accept_probs = ratio.add(epsilon).clamp_(max=1.0) if epsilon else sd_probs

    if probs_only:
        return VerifierMethodOutput([], [], accept_probs, sd_accept_probs=sd_probs)

    accept_until = first_reject_index(torch.rand_like(accept_probs) <= accept_probs)

    gather_idx = accept_until.view(-1, 1, 1).expand(-1, 1, logits_p.size(-1))
    p_next = temp_aware_probs(logits_p.gather(1, gather_idx).squeeze(1), temps_t, all_positive)
    recovery = torch.multinomial(p_next, 1).squeeze(1)

    use_residual = (accept_until < K) & (temps_t > 0)
    if use_residual.any():
        idx = use_residual.nonzero(as_tuple=False).squeeze(1)
        q_at = temp_aware_probs(
            logits_q[idx, accept_until.index_select(0, idx), :],
            temps_q.index_select(0, idx),
        )
        residual = p_next.index_select(0, idx).sub(q_at).clamp_min_(0.0)
        total = residual.sum(dim=1, keepdim=True)
        if (total <= 0).any():
            raise ValueError(
                f"{request.method} verifier hit a zero-mass residual "
                "normalize((P-Q)_+) at a rejected position"
            )
        residual.div_(total)
        recovery = recovery.clone()
        recovery.index_copy_(0, idx, torch.multinomial(residual, 1).squeeze(1))

    new_suffixes, recovery_tokens = materialize_outputs(
        speculations, accept_until, recovery
    )
    return VerifierMethodOutput(
        new_suffixes, recovery_tokens, accept_probs, sd_accept_probs=sd_probs
    )


@register("sd")
def verify_sd(
    request: VerifierMethodRequest,
    probs_only: bool = False,
    all_greedy: bool | None = None,
) -> VerifierMethodOutput:
    return _sd_family(request, 0.0, probs_only, all_greedy)


@register("lossy_sd")
def verify_lossy_sd(
    request: VerifierMethodRequest,
    probs_only: bool = False,
    all_greedy: bool | None = None,
) -> VerifierMethodOutput:
    return _sd_family(
        request, float(request.param("lossy_epsilon") or 0.0), probs_only, all_greedy
    )

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class VerifierMethodRequest:

    method: str
    logits_p: torch.Tensor
    logits_q: torch.Tensor
    speculations: torch.Tensor
    temperatures_target: torch.Tensor
    temperatures_draft: torch.Tensor
    params: dict[str, Any] = field(default_factory=dict)
    logits_q_next: torch.Tensor | None = None
    eta: torch.Tensor | None = None
    eta_bypass: float | None = None

    @property
    def num_spec(self) -> int:
        return self.logits_p.shape[1] - 1

    def param(self, name: str) -> Any:
        return self.params[name]

    def temp_mode(self) -> tuple[bool, bool]:
        cached = getattr(self, "_temp_mode", None)
        if cached is None:
            from .common import resolve_temp_mode
            cached = resolve_temp_mode(self)
            object.__setattr__(self, "_temp_mode", cached)
        return cached

    def __post_init__(self) -> None:
        B, Kp1, V = self.logits_p.shape
        K = Kp1 - 1
        if K < 1:
            raise ValueError(
                f"speculation length must be >= 1, got logits_p {tuple(self.logits_p.shape)}"
            )
        if tuple(self.logits_q.shape) != (B, K, V):
            raise ValueError(
                f"logits_q shape mismatch: expected {(B, K, V)}, got {tuple(self.logits_q.shape)}"
            )
        if tuple(self.speculations.shape) != (B, K + 1):
            raise ValueError(
                f"speculations shape mismatch: expected {(B, K + 1)}, "
                f"got {tuple(self.speculations.shape)}"
            )
        if self.logits_q_next is not None and tuple(self.logits_q_next.shape) != (B, V):
            raise ValueError(
                f"logits_q_next shape mismatch: expected {(B, V)}, "
                f"got {tuple(self.logits_q_next.shape)}"
            )
        if self.eta is not None and tuple(self.eta.shape) != (B, K + 1):
            raise ValueError(
                f"eta shape mismatch: expected {(B, K + 1)}, got {tuple(self.eta.shape)}"
            )


@dataclass
class VerifierMethodOutput:
    new_suffixes: list[list[int]]
    recovery_tokens: list[int]
    accept_probs: torch.Tensor | None = None
    sd_accept_probs: torch.Tensor | None = None

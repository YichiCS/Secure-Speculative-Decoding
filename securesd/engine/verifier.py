from time import perf_counter

import torch

from securesd.constants import ETA_SCHEDULE_CHOICES
from securesd.engine.helpers.speculate_types import SpeculateResult, VerifierBase, VerifyResult
from securesd.engine.model_runner import ModelRunner
from securesd.engine.sequence import Sequence
from securesd.methods import get_spec
from securesd.verifier import run_verifier_method
from securesd.verifier.eta import window_eta, window_eta_bypass
from securesd.verifier.types import VerifierMethodRequest


class Verifier(VerifierBase):

    def __init__(
        self,
        lookahead: int,
        device: torch.device,
        target_model_runner: ModelRunner,
        verify_method: str,
        method_params: dict,
        eta_schedule: str | None,
        eta_start: float,
        eta_end: float,
        eta_len: float,
        eta_gamma: float,
        metrics: dict,
    ):
        super().__init__(lookahead, device)
        self.target_model_runner = target_model_runner
        self.verify_method = verify_method
        self.method_params = method_params
        self.eta_schedule = eta_schedule
        self.eta_start = eta_start
        self.eta_end = eta_end
        self.eta_len = eta_len
        self.eta_gamma = eta_gamma
        self.metrics = metrics

        self._validate_config()

        max_num_seqs = self.target_model_runner.config.max_num_seqs
        self._temperatures_target_buf = torch.empty(max_num_seqs, dtype=torch.float32, device=self.device)
        self._temperatures_draft_buf = torch.empty(max_num_seqs, dtype=torch.float32, device=self.device)

    def _validate_config(self) -> None:
        get_spec(self.verify_method).validate(self.method_params)

        if self.eta_schedule is None:
            return
        if self.eta_schedule not in ETA_SCHEDULE_CHOICES:
            raise ValueError(
                f"Unsupported eta_schedule={self.eta_schedule}. "
                f"Supported: {list(ETA_SCHEDULE_CHOICES)}"
            )
        for name, value in (("eta_start", self.eta_start), ("eta_end", self.eta_end)):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        if self.eta_schedule in ("linear", "power"):
            if self.eta_len <= 0:
                raise ValueError(
                    f"eta {self.eta_schedule} schedule requires eta_len > 0, got {self.eta_len}"
                )
            if self.eta_schedule == "power" and self.eta_gamma <= 0:
                raise ValueError(
                    f"eta power schedule requires eta_gamma > 0, got {self.eta_gamma}"
                )
        elif self.eta_len < 0:
            raise ValueError(f"eta_len must be >= 0, got {self.eta_len}")

    def _apply_method(
        self,
        logits_p: torch.Tensor,
        speculate_result: SpeculateResult,
        temperatures_target: torch.Tensor,
        temperatures_draft: torch.Tensor,
        eta: torch.Tensor | None,
        eta_bypass: float | None,
    ) -> VerifyResult:
        out = run_verifier_method(
            VerifierMethodRequest(
                method=self.verify_method,
                logits_p=logits_p,
                logits_q=speculate_result.logits_q,
                speculations=speculate_result.speculations,
                temperatures_target=temperatures_target,
                temperatures_draft=temperatures_draft,
                params=self.method_params,
                logits_q_next=speculate_result.logits_q_next,
                eta=eta,
                eta_bypass=eta_bypass,
            )
        )
        return VerifyResult(out.new_suffixes, out.recovery_tokens)

    def _build_eta(self, seqs: list[Sequence]) -> tuple[torch.Tensor | None, float | None]:
        if self.eta_schedule is None:
            return None, None
        offsets = [seq.num_accepted_completion_tokens for seq in seqs]
        num_cols = self.lookahead + 1
        bypass = window_eta_bypass(
            self.eta_schedule,
            self.eta_start,
            self.eta_end,
            self.eta_len,
            min(offsets),
            max(offsets),
            num_cols,
        )
        base_offsets = torch.tensor(offsets, dtype=torch.int64, device=self.device)
        eta = window_eta(
            self.eta_schedule,
            self.eta_start,
            self.eta_end,
            self.eta_len,
            base_offsets,
            num_cols,
            self.eta_gamma,
        )
        return eta, bypass

    def _run_target_forward(
        self,
        seqs: list[Sequence],
    ) -> tuple[torch.Tensor, int]:
        batch_size = len(seqs)
        logits_flat = self.target_model_runner.run(seqs, False, False, True)
        logits_p = logits_flat.view(batch_size, self.lookahead + 1, -1)
        return logits_p, self.lookahead + 1

    def prefill(self, seqs: list[Sequence]) -> VerifyResult:
        self.target_model_runner.run(seqs, True)
        return VerifyResult([], [])

    def verify(self, seqs: list[Sequence], speculate_result: SpeculateResult) -> VerifyResult:
        batch_size = len(seqs)
        t0 = perf_counter()

        temperatures_target = self._temperatures_target_buf[:batch_size]
        temperatures_draft = self._temperatures_draft_buf[:batch_size]
        temperatures_target.copy_(
            torch.tensor([seq.temperature for seq in seqs], dtype=torch.float32)
        )
        temperatures_draft.copy_(
            torch.tensor([seq.draft_temperature for seq in seqs], dtype=torch.float32)
        )

        logits_p, verify_query_len = self._run_target_forward(seqs)
        for s in seqs:
            s.num_cached_tokens += verify_query_len

        eta, eta_bypass = self._build_eta(seqs)
        result = self._apply_method(
            logits_p,
            speculate_result,
            temperatures_target,
            temperatures_draft,
            eta,
            eta_bypass,
        )
        final_new_suffixes = result.new_suffixes
        final_recovery_tokens = result.recovery_tokens

        self.metrics["target_verify_times"].append(perf_counter() - t0)
        self.metrics["accepted_suffix_lens_with_recovery"].extend(len(s) for s in final_new_suffixes)

        return VerifyResult(new_suffixes=final_new_suffixes, recovery_tokens=final_recovery_tokens)

import torch

from securesd.engine.sequence import Sequence
from securesd.engine.model_runner import ModelRunner
from securesd.engine.helpers.speculate_types import SpeculateResult, VerifyResult, SpeculatorBase

class SpeculatorSync(SpeculatorBase):

    def __init__(
        self,
        lookahead: int,
        device: torch.device,
        draft_model_runner: ModelRunner,
        method: str = "sd",
    ):
        super().__init__(lookahead, device)
        self.draft_model_runner = draft_model_runner
        self.method = method
        self._speculations_buf = torch.empty(
            draft_model_runner.config.max_num_seqs,
            lookahead + 1,
            dtype=torch.int64,
            device=device,
        )

    @staticmethod
    def _advance_draft_state(
        seqs: list[Sequence],
        token_ids: list[int] | None = None,
    ):
        if token_ids is None:
            for seq in seqs:
                seq.num_draft_cached_tokens += 1
            return

        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            seq.num_draft_cached_tokens += 1

    def prefill(self, seqs: list[Sequence], verify_result: VerifyResult) -> SpeculateResult:
        self.draft_model_runner.run(seqs, True)
        return SpeculateResult([], [], None)

    def speculate(self, seqs: list[Sequence], verify_result: VerifyResult) -> SpeculateResult:
        batch_size = len(seqs)
        speculations = self._speculations_buf[:batch_size]
        logits_q: list[torch.Tensor | None] = [None] * self.lookahead
        logits_q_next: torch.Tensor | None = None
        is_sc = self.method == "sc"

        seed_tokens = [0] * batch_size
        for i, seq in enumerate(seqs):
            if seq.recovery_token_id is None:
                seed_tokens[i] = seq.last_token
            else:
                seed_tokens[i] = seq.recovery_token_id
                seq.append_token(seq.recovery_token_id)
        speculations[:, 0] = torch.tensor(seed_tokens, dtype=torch.int64, device=self.device)

        for k in range(self.lookahead + 1):
            need_step_logits = (k < self.lookahead) or is_sc
            out = self.draft_model_runner.run(
                seqs,
                is_prefill=False,
                last_only=True,
                draft_return_logits=need_step_logits,
                return_token_tensor=True,
            )
            token_ids_t, step_logits_q = out if need_step_logits else (out, None)

            if k == self.lookahead:
                self._advance_draft_state(seqs)
                if is_sc:
                    logits_q_next = step_logits_q
                break

            logits_q[k] = step_logits_q
            self._advance_draft_state(seqs, token_ids_t.tolist())
            speculations[:, k + 1] = token_ids_t

        return SpeculateResult(speculations, torch.stack(logits_q, dim=1), logits_q_next)

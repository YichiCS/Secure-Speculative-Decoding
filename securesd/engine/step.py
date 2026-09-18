from abc import ABC, abstractmethod

from securesd.engine.helpers.speculate_types import SpeculatorBase, VerifierBase, VerifyResult
from securesd.engine.model_runner import ModelRunner
from securesd.engine.scheduler import Scheduler
from securesd.engine.sequence import Sequence


class InferenceStep(ABC):

    def __init__(self, scheduler: Scheduler):
        self.scheduler = scheduler

    @abstractmethod
    def prefill(self, seqs: list[Sequence]) -> int:
        raise NotImplementedError

    @abstractmethod
    def decode(self, seqs: list[Sequence]) -> int:
        raise NotImplementedError


class AutoRegressiveStep(InferenceStep):

    def __init__(self, scheduler: Scheduler, model_runner: ModelRunner):
        super().__init__(scheduler)
        self.model_runner = model_runner

    def prefill(self, seqs: list[Sequence]) -> int:
        token_ids = self.model_runner.run(seqs, True)
        self.scheduler.postprocess(seqs, token_ids, is_prefill=True)
        return sum(len(seq) for seq in seqs)

    def decode(self, seqs: list[Sequence]) -> int:
        token_ids = self.model_runner.run(seqs, False)
        self.scheduler.postprocess(seqs, token_ids, is_prefill=False)
        return len(seqs)


class SpecDecodeStep(InferenceStep):

    def __init__(
        self,
        scheduler: Scheduler,
        speculator: SpeculatorBase,
        verifier: VerifierBase,
    ):
        super().__init__(scheduler)
        self.speculator = speculator
        self.verifier = verifier

    def prefill(self, seqs: list[Sequence]) -> int:
        verify_result = self.verifier.prefill(seqs)
        self.speculator.prefill(seqs, verify_result)
        for seq in seqs:
            seq.recovery_token_id = None
            seq.num_cached_tokens = seq.num_prompt_tokens - 1
            seq.num_draft_cached_tokens = seq.num_prompt_tokens - 1
        return sum(len(seq) for seq in seqs)

    def decode(self, seqs: list[Sequence]) -> int:
        saved = [
            (len(seq.token_ids), seq.num_tokens, seq.last_token,
             seq.num_draft_cached_tokens, seq.num_cached_tokens)
            for seq in seqs
        ]
        is_first_step = [seq.recovery_token_id is None for seq in seqs]

        speculate_result = self.speculator.speculate(seqs, VerifyResult([], []))
        verify_result = self.verifier.verify(seqs, speculate_result)

        for seq, (orig_len, orig_nt, orig_lt, orig_ndc, orig_nct) in zip(seqs, saved):
            del seq.token_ids[orig_len:]
            seq.num_tokens = orig_nt
            seq.last_token = orig_lt
            seq.num_draft_cached_tokens = orig_ndc
            seq.num_cached_tokens = orig_nct

        new_suffixes = verify_result.new_suffixes
        if any(is_first_step):
            new_suffixes = list(new_suffixes)
            for i, first_step in enumerate(is_first_step):
                if first_step:
                    seqs[i].num_cached_tokens += 1
                    seqs[i].num_draft_cached_tokens += 1
                    new_suffixes[i] = new_suffixes[i][1:]

        self.scheduler.postprocess_speculate(
            seqs,
            new_suffixes,
            verify_result.recovery_tokens,
        )
        return sum(len(s) for s in new_suffixes)

from copy import copy
from enum import Enum, auto
from itertools import count

from securesd.sampling_params import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    counter = count()
    block_size: int = 0

    def __init__(self, token_ids: list[int], sampling_params: SamplingParams):
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.block_table: list[int] = []
        self.last_spec_step_accepted_len = -1

        self.draft_block_table: list[int] = []
        self.num_draft_cached_tokens = 0

        self.temperature = sampling_params.temperature
        self.draft_temperature = sampling_params.draft_temperature
        self.max_new_tokens = sampling_params.max_new_tokens
        self.ignore_eos = sampling_params.ignore_eos

        self.recovery_token_id: int | None = None
        self.target_finalized_blocks = 0
        self.draft_finalized_blocks = 0
        self.num_accepted_tokens = len(token_ids)

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def num_accepted_completion_tokens(self):
        return self.num_accepted_tokens - self.num_prompt_tokens

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_cached_blocks(self):
        return (self.num_cached_tokens + self.block_size - 1) // self.block_size

    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def num_draft_cached_blocks(self):
        return (self.num_draft_cached_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_cached_blocks - 1) * self.block_size

    @property
    def last_block_num_tokens_draft(self):
        return self.num_tokens - (self.num_draft_cached_blocks - 1) * self.block_size

    def block(self, i: int) -> list[int]:
        if not 0 <= i < self.num_blocks:
            raise IndexError(f"block index {i} out of range [0, {self.num_blocks})")
        return self.token_ids[i * self.block_size:(i + 1) * self.block_size]

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

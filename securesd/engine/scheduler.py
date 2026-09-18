from collections import deque

from securesd.config import Config
from securesd.engine.block_manager import BlockManager
from securesd.engine.sequence import Sequence, SequenceStatus


class Scheduler:

    def __init__(self, config: Config, draft_config: Config | None = None):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.max_model_len = config.max_model_len
        self.eos = config.eos
        self.speculate = config.speculate
        self.K = config.speculate_k
        self.target_verify_lookahead = self.K + 1
        self.block_size = config.kvcache_block_size

        self.block_manager = BlockManager(
            config.num_kvcache_blocks,
            config.kvcache_block_size,
            is_draft=False,
            max_model_len=self.max_model_len,
        )
        self.draft_block_manager: BlockManager | None = None
        if self.speculate:
            if draft_config is None:
                raise ValueError("draft_config required when speculate=True")
            self.draft_block_manager = BlockManager(
                draft_config.num_kvcache_blocks,
                draft_config.kvcache_block_size,
                is_draft=True,
                speculate_k=self.K,
                max_model_len=self.max_model_len,
            )

        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self) -> bool:
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def _next_decode_lookahead_len(self) -> int:
        return self.target_verify_lookahead if self.speculate else 1

    def bms_can_append(
        self,
        seq: Sequence,
        target_lookahead_len: int,
        draft_lookahead_len: int | None = None,
    ) -> bool:
        if not self.block_manager.can_append(seq, target_lookahead_len):
            return False
        if not self.speculate:
            if draft_lookahead_len is not None:
                raise ValueError("draft_lookahead_len must be None when speculate=False")
            return True
        return self.draft_block_manager.can_append(seq, draft_lookahead_len)

    def bms_can_allocate(self, seq: Sequence) -> bool:
        if not self.block_manager.can_allocate(seq):
            return False
        return not self.speculate or self.draft_block_manager.can_allocate(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        scheduled_seqs: list[Sequence] = []
        num_batched_tokens = 0

        while self.waiting:
            seq = self.waiting[0]
            remain = len(seq) - seq.num_cached_tokens
            if num_batched_tokens + remain > self.max_num_batched_tokens or not self.bms_can_allocate(seq):
                break

            self.block_manager.allocate(seq)
            if self.speculate:
                self.draft_block_manager.allocate(seq)

            num_batched_tokens += remain
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)

        if scheduled_seqs:
            return scheduled_seqs, True

        num_seqs_decoded = 0
        if self.speculate:
            target_lookahead_len = self.target_verify_lookahead
            draft_lookahead_len = self.K + 1
        else:
            target_lookahead_len = 1
            draft_lookahead_len = None

        while self.running and num_seqs_decoded < self.max_num_seqs:
            seq = self.running.popleft()

            while not self.bms_can_append(seq, target_lookahead_len, draft_lookahead_len):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                num_seqs_decoded += 1
                self.block_manager.may_append(seq, target_lookahead_len)
                if self.speculate:
                    self.draft_block_manager.may_append(seq, draft_lookahead_len)
                scheduled_seqs.append(seq)

        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.recovery_token_id = None
        self.block_manager.deallocate(seq)
        if self.speculate:
            self.draft_block_manager.deallocate(seq)
        self.waiting.appendleft(seq)
        seq.num_prompt_tokens = seq.num_tokens
        seq.last_spec_step_accepted_len = -1

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        finished_ids: set[int] = set()
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            seq.num_cached_tokens = seq.num_prompt_tokens if is_prefill else seq.num_cached_tokens + 1

            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_new_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                finished_ids.add(seq.seq_id)
                continue

            if seq.last_block_num_tokens == self.block_size:
                block_table = seq.block_table
                token_ids_block = seq.block(seq.num_blocks - 1)
                prefix = self.block_manager.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
                h = self.block_manager.compute_hash(token_ids_block, prefix)
                last_block = self.block_manager.blocks[block_table[-1]]
                last_block.update(h, token_ids_block)
                self.block_manager.hash_to_block_id[h] = last_block.block_id

        if finished_ids:
            self.running = deque(s for s in self.running if s.seq_id not in finished_ids)

    def _truncate_suffix(self, seq: Sequence, new_suffix: list[int]) -> tuple[list[int], bool]:
        if not seq.ignore_eos and self.eos in new_suffix:
            new_suffix = new_suffix[:new_suffix.index(self.eos) + 1]

        over_cap = seq.num_completion_tokens + len(new_suffix) - seq.max_new_tokens
        if over_cap > 0:
            new_suffix = new_suffix[:-over_cap] if over_cap <= len(new_suffix) else []

        overflow = (seq.num_tokens + len(new_suffix)) - self.max_model_len
        if overflow > 0:
            new_suffix = new_suffix[:max(0, len(new_suffix) - overflow)]

        final_total = seq.num_tokens + len(new_suffix)
        finished = (
            (not seq.ignore_eos and new_suffix and new_suffix[-1] == self.eos)
            or (seq.num_completion_tokens + len(new_suffix) == seq.max_new_tokens)
            or (final_total + self._next_decode_lookahead_len() > self.max_model_len)
        )
        if seq.num_completion_tokens > seq.max_new_tokens:
            raise RuntimeError(
                f"num_completion_tokens={seq.num_completion_tokens} > max_new_tokens={seq.max_new_tokens}"
            )
        return new_suffix, finished

    def _update_kv_caches(self, seq: Sequence, new_suffix: list[int]):
        required = (seq.num_tokens + len(new_suffix) + self.block_size - 1) // self.block_size
        self._shrink_blocks(seq.block_table, self.block_manager, required)
        self._shrink_blocks(seq.draft_block_table, self.draft_block_manager, required)
        seq.target_finalized_blocks = min(seq.target_finalized_blocks, len(seq.block_table))
        seq.draft_finalized_blocks = min(seq.draft_finalized_blocks, len(seq.draft_block_table))

    @staticmethod
    def _shrink_blocks(block_table: list[int], manager: BlockManager, required: int):
        excess = len(block_table) - required
        if excess <= 0:
            return
        for block_id in block_table[-excess:]:
            block = manager.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                manager._deallocate_block(block_id)
        del block_table[-excess:]

    def _finalize_block(self, manager: BlockManager, seq: Sequence, block_table: list[int], index: int):
        token_ids = seq.block(index)
        prefix = manager.blocks[block_table[index - 1]].hash if index > 0 else -1
        h = manager.compute_hash(token_ids, prefix)
        block = manager.blocks[block_table[index]]
        block.update(h, token_ids)
        manager.hash_to_block_id[h] = block.block_id

    def _finalize_complete_blocks(self, seq: Sequence):
        complete = seq.num_tokens // self.block_size

        def _run(manager: BlockManager, block_table: list[int], start_attr: str):
            idx = getattr(seq, start_attr)
            limit = min(complete, len(block_table))
            while idx < limit:
                block = manager.blocks[block_table[idx]]
                if block.hash == -1:
                    self._finalize_block(manager, seq, block_table, idx)
                idx += 1
            setattr(seq, start_attr, idx)

        _run(self.block_manager, seq.block_table, "target_finalized_blocks")
        _run(self.draft_block_manager, seq.draft_block_table, "draft_finalized_blocks")

    def _update_sequence_metadata(self, seq: Sequence, new_suffix: list[int], recovery_token: int):
        seq.token_ids.extend(new_suffix)
        seq.num_tokens += len(new_suffix)
        seq.num_accepted_tokens += len(new_suffix)
        if new_suffix:
            seq.last_token = new_suffix[-1]
        seq.num_cached_tokens += len(new_suffix)
        seq.num_draft_cached_tokens += len(new_suffix)
        seq.last_spec_step_accepted_len = len(new_suffix)
        seq.recovery_token_id = recovery_token

        if seq.last_block_num_tokens != seq.last_block_num_tokens_draft:
            raise RuntimeError(
                f"target last-block tokens {seq.last_block_num_tokens} != "
                f"draft last-block tokens {seq.last_block_num_tokens_draft}"
            )
        if not seq.block_table or not seq.draft_block_table:
            raise RuntimeError("block tables must be populated after speculative step")

        self._finalize_complete_blocks(seq)

    def postprocess_speculate(
        self,
        seqs: list[Sequence],
        new_suffixes: list[list[int]],
        next_recovery_tokens: list[int],
    ):
        finished_ids: set[int] = set()
        for seq, new_suffix, recovery in zip(seqs, new_suffixes, next_recovery_tokens):
            new_suffix, finished = self._truncate_suffix(seq, new_suffix)
            self._update_kv_caches(seq, new_suffix)
            self._update_sequence_metadata(seq, new_suffix, recovery)
            if finished:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.draft_block_manager.deallocate(seq)
                finished_ids.add(seq.seq_id)
        if finished_ids:
            self.running = deque(s for s in self.running if s.seq_id not in finished_ids)

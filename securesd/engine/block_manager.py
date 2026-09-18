from collections import deque

import numpy as np
import xxhash

from securesd.engine.sequence import Sequence


class Block:

    def __init__(self, block_id: int):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids: list[int] = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def acquire(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []

    def release(self):
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []


class BlockManager:

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        is_draft: bool = False,
        speculate_k: int = -1,
        max_model_len: int = -1,
    ):
        if num_blocks <= 0:
            raise ValueError(f"num_blocks must be > 0, got {num_blocks}")
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = {}
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()
        self.is_draft = is_draft
        self.speculate_k = speculate_k
        self.max_model_len = max_model_len

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1) -> int:
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _take_free_block(self, block_id: int, from_head: bool) -> Block:
        block = self.blocks[block_id]
        if block.ref_count != 0:
            raise RuntimeError(f"attempted to take in-use block {block_id}")
        if from_head:
            popped = self.free_block_ids.popleft()
            if popped != block_id:
                raise RuntimeError(
                    f"free-queue corruption: expected head={block_id}, got {popped}"
                )
        else:
            self.free_block_ids.remove(block_id)
        block.acquire()
        self.used_block_ids.add(block_id)
        return block

    def _deallocate_block(self, block_id: int) -> None:
        block = self.blocks[block_id]
        if block.ref_count != 0:
            raise RuntimeError(f"refcount != 0 on dealloc of block {block_id}")
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]
        self.used_block_ids.remove(block_id)
        block.release()
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> bool:
        h = -1
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                return len(self.free_block_ids) >= seq.num_blocks - i
        return True

    def allocate(self, seq: Sequence):
        block_table = seq.draft_block_table if self.is_draft else seq.block_table
        if block_table:
            raise RuntimeError("block_table already populated before allocate()")

        h = -1
        cache_miss = False
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True

            if cache_miss:
                block_id = self.free_block_ids[0]
                block = self._take_free_block(block_id, from_head=True)
            else:
                if self.is_draft:
                    seq.num_draft_cached_tokens += self.block_size
                else:
                    seq.num_cached_tokens += self.block_size
                if block_id in self.used_block_ids:
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    block = self._take_free_block(block_id, from_head=False)

            if h != -1:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id
            block_table.append(block_id)

        finalized = 0
        for block_id in block_table:
            if self.blocks[block_id].hash == -1:
                break
            finalized += 1
        if self.is_draft:
            seq.draft_finalized_blocks = finalized
        else:
            seq.target_finalized_blocks = finalized

    def deallocate(self, seq: Sequence):
        block_table = seq.draft_block_table if self.is_draft else seq.block_table
        for block_id in reversed(block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)

        if self.is_draft:
            seq.num_draft_cached_tokens = 0
            seq.draft_finalized_blocks = 0
        else:
            seq.num_cached_tokens = 0
            seq.target_finalized_blocks = 0

        block_table.clear()

    def _required_blocks(self, seq: Sequence, lookahead_num_tokens: int) -> int:
        return (seq.num_tokens + lookahead_num_tokens + self.block_size - 1) // self.block_size

    def can_append(self, seq: Sequence, lookahead_num_tokens: int = 1) -> bool:
        if seq.num_tokens + lookahead_num_tokens > self.max_model_len:
            return False
        block_table = seq.draft_block_table if self.is_draft else seq.block_table
        needed = self._required_blocks(seq, lookahead_num_tokens) - len(block_table)
        return needed <= 0 or len(self.free_block_ids) >= needed

    def may_append(self, seq: Sequence, lookahead_num_tokens: int = 1):
        block_table = seq.draft_block_table if self.is_draft else seq.block_table
        needed = self._required_blocks(seq, lookahead_num_tokens) - len(block_table)
        if needed <= 0:
            return
        if len(self.free_block_ids) < needed:
            raise RuntimeError(
                f"may_append: need {needed} blocks but only {len(self.free_block_ids)} free"
            )
        for _ in range(needed):
            block_id = self.free_block_ids.popleft()
            block = self.blocks[block_id]
            if block.ref_count != 0:
                raise RuntimeError(f"attempted to take in-use block {block_id}")
            block.acquire()
            self.used_block_ids.add(block_id)
            block_table.append(block_id)

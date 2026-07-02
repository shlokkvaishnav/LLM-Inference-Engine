"""
Paged KV-cache block manager — Milestone 4.

Core idea: instead of allocating a contiguous GPU tensor per sequence
(which fragments badly and caps batch size to however many fit), we divide
the entire KV-cache into fixed-size blocks and give each sequence a
logical→physical block table — the same idea as virtual memory paging.

Benefits:
  - Zero internal fragmentation (unused slots are only in the last block)
  - Sequences of wildly different lengths share the same pool
  - Eviction is O(num_blocks) not O(sequence_length) in memory terms
  - Prefix caching (future): blocks for shared prefixes can be ref-counted

This module is pure bookkeeping — no tensors, no GPU. It decides WHICH
block IDs belong to a sequence. The actual paged-attention kernel (M4b)
uses those IDs to gather KV tensors scattered across the pool.
"""
from __future__ import annotations
import math


class PhysicalBlock:
    """One allocatable unit of KV-cache GPU memory."""

    def __init__(self, block_id: int, block_size: int) -> None:
        self.block_id = block_id
        self.block_size = block_size  # number of tokens this block can hold
        self.ref_count = 0            # >1 means shared (prefix caching, Milestone 4+)

    def __repr__(self) -> str:
        return f"Block(id={self.block_id}, refs={self.ref_count})"


class BlockManager:
    """
    Allocates/frees fixed-size KV-cache blocks and tracks each sequence's
    logical→physical mapping via seq.block_table (a flat list[int] of
    physical block IDs, in logical order — index 0 is the first block).

    Public contract:
      allocate(seq)      — assign physical blocks for the full prompt
      free(seq)          — return all of seq's blocks to the free pool
      can_allocate(seq)  — check headroom before scheduling a sequence
      append_slot(seq)   — ensure space for one more decode token;
                            returns True if a new block was allocated
      num_free_blocks    — property: current free pool size
    """

    def __init__(self, num_blocks: int, block_size: int) -> None:
        self.num_blocks = num_blocks
        self.block_size = block_size
        self._free_blocks: list[PhysicalBlock] = [
            PhysicalBlock(i, block_size) for i in range(num_blocks)
        ]
        # O(1) reverse lookup for free(): block_id -> PhysicalBlock, regardless
        # of whether the block is currently free or allocated.
        self._id_to_block: dict[int, PhysicalBlock] = {
            b.block_id: b for b in self._free_blocks
        }

    @property
    def num_free_blocks(self) -> int:
        return len(self._free_blocks)

    def _blocks_needed(self, num_tokens: int) -> int:
        """ceil(num_tokens / block_size) — how many blocks to hold num_tokens."""
        if num_tokens <= 0:
            return 0
        return math.ceil(num_tokens / self.block_size)

    def can_allocate(self, seq: object) -> bool:
        """
        Check whether enough free blocks exist to cover seq's current length,
        beyond whatever blocks it already holds (block_table may be non-empty
        if this is called for append_slot planning rather than fresh prefill).
        """
        needed = self._blocks_needed(seq.length) - len(seq.block_table)
        return needed <= self.num_free_blocks

    def allocate(self, seq: object) -> None:
        """
        Assign physical blocks to cover seq's full current length (typically
        called once, right after prefill, when seq.block_table is empty).

        Raises if the pool can't satisfy the request — callers must check
        can_allocate() first; the scheduler uses this to decide preemption.
        """
        needed = self._blocks_needed(seq.length) - len(seq.block_table)
        if needed > self.num_free_blocks:
            raise RuntimeError(
                f"Cannot allocate {needed} blocks: only "
                f"{self.num_free_blocks} free (pool size {self.num_blocks})."
            )
        for _ in range(needed):
            block = self._free_blocks.pop()
            block.ref_count = 1
            seq.block_table.append(block.block_id)

    def append_slot(self, seq: object) -> bool:
        """
        Ensure space exists for the token just appended to seq (call this
        AFTER seq.append_token()). Allocates a new block only when the
        current blocks' total capacity has been exceeded — i.e. exactly at
        a block boundary, not on every token.

        Returns True if a new block was allocated.
        """
        capacity = len(seq.block_table) * self.block_size
        if seq.length <= capacity:
            return False   # still room in the last block

        if self.num_free_blocks == 0:
            raise RuntimeError("Out of KV-cache blocks — caller must preempt.")
        block = self._free_blocks.pop()
        block.ref_count = 1
        seq.block_table.append(block.block_id)
        return True

    def free(self, seq: object) -> None:
        """
        Return all of seq's blocks to the free pool and clear its block_table.
        Decrements ref_count first — a block with ref_count > 1 (shared via
        prefix caching, not yet implemented) would stay allocated until the
        last owner frees it.
        """
        for block_id in seq.block_table:
            block = self._id_to_block[block_id]
            block.ref_count -= 1
            if block.ref_count <= 0:
                block.ref_count = 0
                self._free_blocks.append(block)
        seq.block_table.clear()

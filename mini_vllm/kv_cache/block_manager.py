"""
Paged KV-cache block manager — Milestone 4 (built interactively).

Core idea: instead of allocating a contiguous GPU tensor per sequence
(which fragments badly and caps batch size to however many fit), we divide
the entire KV-cache into fixed-size blocks and give each sequence a
logical→physical block table — the same idea as virtual memory paging.

Benefits:
  - Zero internal fragmentation (unused slots are only in the last block)
  - Sequences of wildly different lengths share the same pool
  - Eviction is O(num_blocks) not O(sequence_length) in memory terms
  - Prefix caching (future): blocks for shared prefixes can be ref-counted

We implement this together in Milestone 4 so you can explain block allocation,
append-slot semantics, and eviction policy under memory pressure.
"""
from __future__ import annotations


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
    Placeholder — implemented in Milestone 4.

    Public contract:
      allocate(seq)         — assign physical blocks for the full prompt
      free(seq)             — return all of seq's blocks to the free pool
      can_allocate(seq)     — check headroom before scheduling a sequence
      append_slot(seq)      — ensure space for one more decode token;
                              returns True if a new block was allocated
      num_free_blocks       — property: current free pool size
    """

    def __init__(self, num_blocks: int, block_size: int) -> None:
        self.num_blocks = num_blocks
        self.block_size = block_size
        self._free_blocks: list[PhysicalBlock] = [
            PhysicalBlock(i, block_size) for i in range(num_blocks)
        ]

    @property
    def num_free_blocks(self) -> int:
        return len(self._free_blocks)

    def allocate(self, seq: object) -> None:
        raise NotImplementedError("Milestone 4")

    def free(self, seq: object) -> None:
        raise NotImplementedError("Milestone 4")

    def can_allocate(self, seq: object) -> bool:
        raise NotImplementedError("Milestone 4")

    def append_slot(self, seq: object) -> bool:
        """Return True if a new block was allocated (useful for stats)."""
        raise NotImplementedError("Milestone 4")

"""Per-sequence logical → physical block mapping (Milestone 4)."""
from __future__ import annotations


class BlockTable:
    """
    Maps logical block indices (0, 1, 2, ...) to physical block IDs in the
    KV-cache pool.

    A sequence with 512 tokens and block_size=16 has 32 logical blocks.
    Each maps to one physical block that holds 16 tokens × (num_heads × head_dim)
    key + value tensors.

    Think of it as an OS page table, but for attention key/value tensors.
    """

    def __init__(self) -> None:
        self._table: list[int] = []  # logical_idx → physical_block_id

    def append(self, physical_block_id: int) -> None:
        self._table.append(physical_block_id)

    def get_physical(self, logical_idx: int) -> int:
        return self._table[logical_idx]

    def pop(self) -> int:
        return self._table.pop()

    def __len__(self) -> int:
        return len(self._table)

    def as_list(self) -> list[int]:
        """Flat list of physical block IDs, in logical order."""
        return list(self._table)

    def __repr__(self) -> str:
        return f"BlockTable({self._table})"

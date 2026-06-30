"""Block manager unit tests — Milestone 4."""
import pytest

from mini_vllm.kv_cache.block_manager import BlockManager
from mini_vllm.engine.sequence import Sequence, SamplingParams


@pytest.mark.skip(reason="Milestone 4 — not yet implemented")
def test_allocate_and_free_returns_blocks():
    """Blocks allocated for a sequence are fully returned on free()."""
    ...


@pytest.mark.skip(reason="Milestone 4 — not yet implemented")
def test_cannot_allocate_beyond_pool():
    """can_allocate() returns False when the free pool is exhausted."""
    ...


@pytest.mark.skip(reason="Milestone 4 — not yet implemented")
def test_append_slot_allocates_new_block_at_boundary():
    """append_slot() allocates a new block exactly when a block fills up."""
    ...


@pytest.mark.skip(reason="Milestone 4 — not yet implemented")
def test_block_table_length_matches_token_count():
    """Number of blocks in the block table == ceil(num_tokens / block_size)."""
    ...

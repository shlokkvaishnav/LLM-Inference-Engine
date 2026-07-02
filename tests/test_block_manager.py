"""Block manager unit tests — Milestone 4. Pure Python, no GPU/model needed."""
import pytest

from mini_vllm.kv_cache.block_manager import BlockManager
from mini_vllm.engine.sequence import Sequence, SamplingParams


def _make_seq(num_prompt_tokens: int) -> Sequence:
    return Sequence(list(range(num_prompt_tokens)), SamplingParams())


def test_allocate_and_free_returns_blocks():
    """Blocks allocated for a sequence are fully returned on free()."""
    bm = BlockManager(num_blocks=8, block_size=4)
    seq = _make_seq(10)   # needs ceil(10/4) = 3 blocks

    bm.allocate(seq)
    assert len(seq.block_table) == 3
    assert bm.num_free_blocks == 5

    bm.free(seq)
    assert seq.block_table == []
    assert bm.num_free_blocks == 8


def test_cannot_allocate_beyond_pool():
    """can_allocate() returns False when the free pool is exhausted."""
    bm = BlockManager(num_blocks=2, block_size=4)
    small = _make_seq(4)    # needs 1 block
    big = _make_seq(20)     # needs 5 blocks — more than the whole pool

    assert bm.can_allocate(small) is True
    bm.allocate(small)

    assert bm.can_allocate(big) is False
    with pytest.raises(RuntimeError):
        bm.allocate(big)

    # Failed allocation must not have partially consumed the pool.
    assert bm.num_free_blocks == 1


def test_append_slot_allocates_new_block_at_boundary():
    """append_slot() allocates a new block exactly when a block fills up."""
    bm = BlockManager(num_blocks=8, block_size=4)
    seq = _make_seq(4)   # exactly fills one block (capacity 4, length 4)
    bm.allocate(seq)
    assert len(seq.block_table) == 1

    # Token 5: length=5 > capacity(1*4=4) -> allocate 2nd block.
    seq.append_token(999)
    assert bm.append_slot(seq) is True
    assert len(seq.block_table) == 2

    # Tokens 6, 7, 8: length 6,7,8 <= capacity(2*4=8) -> no new block.
    for _ in range(3):
        seq.append_token(999)
        assert bm.append_slot(seq) is False
    assert len(seq.block_table) == 2

    # Token 9: length=9 > capacity(2*4=8) -> allocate 3rd block.
    seq.append_token(999)
    assert bm.append_slot(seq) is True
    assert len(seq.block_table) == 3


def test_block_table_length_matches_token_count():
    """Number of blocks in the block table == ceil(num_tokens / block_size)."""
    bm = BlockManager(num_blocks=16, block_size=4)

    for num_tokens, expected_blocks in [(1, 1), (4, 1), (5, 2), (16, 4), (17, 5)]:
        seq = _make_seq(num_tokens)
        bm.allocate(seq)
        assert len(seq.block_table) == expected_blocks, (
            f"{num_tokens} tokens should need {expected_blocks} blocks, "
            f"got {len(seq.block_table)}"
        )
        bm.free(seq)

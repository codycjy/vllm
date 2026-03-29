# SPDX-License-Identifier: Apache-2.0
"""Tests for adaptive eviction policy in FreeKVCacheBlockQueue.

These tests verify the non-invasive eviction policy switch:
- "lru" mode behaves identically to the original vLLM implementation.
- "adaptive" mode evicts blocks with the lowest reuse count.
- Blocks without cached data (block_hash=None) are always evicted first.

No GPU required — pure Python unit tests.
"""

import pytest

from vllm.v1.core.kv_cache_utils import (
    FreeKVCacheBlockQueue,
    KVCacheBlock,
    make_block_hash_with_group_id,
)
from vllm.utils.hashing import sha256

pytestmark = pytest.mark.cpu_test


def _make_block_hash(token_id: int, group_id: int = 0):
    """Create a deterministic block hash for testing."""
    raw = sha256(token_id.to_bytes(8, "big"))
    return make_block_hash_with_group_id(raw, group_id)


# ─── LRU baseline tests ────────────────────────────────────────────

class TestLRUEviction:
    """Verify that eviction_policy='lru' preserves original behavior."""

    def test_popleft_returns_front(self):
        blocks = [KVCacheBlock(block_id=i) for i in range(5)]
        queue = FreeKVCacheBlockQueue(blocks, eviction_policy="lru")
        assert queue.popleft().block_id == 0
        assert queue.popleft().block_id == 1
        assert queue.num_free_blocks == 3

    def test_popleft_n_returns_front_n(self):
        blocks = [KVCacheBlock(block_id=i) for i in range(5)]
        queue = FreeKVCacheBlockQueue(blocks, eviction_policy="lru")
        popped = queue.popleft_n(3)
        assert [b.block_id for b in popped] == [0, 1, 2]
        assert queue.num_free_blocks == 2

    def test_reuse_counter_not_populated(self):
        blocks = [KVCacheBlock(block_id=i) for i in range(3)]
        queue = FreeKVCacheBlockQueue(blocks, eviction_policy="lru")
        # increment_reuse should be a no-op for LRU
        blocks[0]._block_hash = _make_block_hash(42)
        queue.increment_reuse(blocks[0])
        assert len(queue.reuse_counter) == 0

    def test_default_policy_is_lru(self):
        blocks = [KVCacheBlock(block_id=i) for i in range(3)]
        queue = FreeKVCacheBlockQueue(blocks)
        assert queue.eviction_policy == "lru"


# ─── Adaptive eviction tests ───────────────────────────────────────

class TestAdaptiveEviction:
    """Verify that eviction_policy='adaptive' uses reuse counting."""

    def test_no_hash_blocks_evicted_first(self):
        """Blocks without block_hash should be evicted immediately."""
        blocks = [KVCacheBlock(block_id=i) for i in range(4)]
        queue = FreeKVCacheBlockQueue(blocks, eviction_policy="adaptive")

        # Give hashes to blocks 0, 1, 3 — but NOT block 2
        blocks[0]._block_hash = _make_block_hash(10)
        blocks[1]._block_hash = _make_block_hash(20)
        # blocks[2] has no hash (None)
        blocks[3]._block_hash = _make_block_hash(30)

        # Adaptive should pick block 2 first (no cached data = free eviction)
        victim = queue.popleft()
        assert victim.block_id == 2
        assert queue.num_free_blocks == 3

    def test_lowest_reuse_count_evicted(self):
        """Block with lowest reuse count should be evicted."""
        blocks = [KVCacheBlock(block_id=i) for i in range(3)]
        queue = FreeKVCacheBlockQueue(blocks, eviction_policy="adaptive")

        # Assign hashes
        h0 = _make_block_hash(100)
        h1 = _make_block_hash(200)
        h2 = _make_block_hash(300)
        blocks[0]._block_hash = h0
        blocks[1]._block_hash = h1
        blocks[2]._block_hash = h2

        # Simulate reuse: block 0 hit 5 times, block 1 hit 1 time,
        # block 2 hit 3 times
        for _ in range(5):
            queue.increment_reuse(blocks[0])
        for _ in range(1):
            queue.increment_reuse(blocks[1])
        for _ in range(3):
            queue.increment_reuse(blocks[2])

        # Should evict block 1 (lowest reuse count = 1)
        victim = queue.popleft()
        assert victim.block_id == 1

        # Next should evict block 2 (reuse count = 3)
        victim = queue.popleft()
        assert victim.block_id == 2

        # Finally block 0 (reuse count = 5)
        victim = queue.popleft()
        assert victim.block_id == 0

    def test_zero_reuse_count_evicted_before_nonzero(self):
        """Blocks with 0 reuse count (but with hash) should be evicted
        before blocks with higher reuse counts."""
        blocks = [KVCacheBlock(block_id=i) for i in range(3)]
        queue = FreeKVCacheBlockQueue(blocks, eviction_policy="adaptive")

        blocks[0]._block_hash = _make_block_hash(10)
        blocks[1]._block_hash = _make_block_hash(20)
        blocks[2]._block_hash = _make_block_hash(30)

        # Only increment reuse for blocks 0 and 2
        queue.increment_reuse(blocks[0])
        queue.increment_reuse(blocks[2])

        # Block 1 has reuse count 0 → evicted first
        victim = queue.popleft()
        assert victim.block_id == 1

    def test_popleft_n_adaptive(self):
        """popleft_n should respect adaptive ordering."""
        blocks = [KVCacheBlock(block_id=i) for i in range(4)]
        queue = FreeKVCacheBlockQueue(blocks, eviction_policy="adaptive")

        blocks[0]._block_hash = _make_block_hash(10)
        blocks[1]._block_hash = _make_block_hash(20)
        blocks[2]._block_hash = _make_block_hash(30)
        blocks[3]._block_hash = _make_block_hash(40)

        # Reuse counts: block0=3, block1=0, block2=1, block3=2
        for _ in range(3):
            queue.increment_reuse(blocks[0])
        for _ in range(1):
            queue.increment_reuse(blocks[2])
        for _ in range(2):
            queue.increment_reuse(blocks[3])

        # Pop 3 blocks — should be in order: block1(0), block2(1), block3(2)
        victims = queue.popleft_n(3)
        assert [v.block_id for v in victims] == [1, 2, 3]
        assert queue.num_free_blocks == 1

    def test_empty_queue_raises(self):
        """Popping from empty adaptive queue should raise ValueError."""
        queue = FreeKVCacheBlockQueue([], eviction_policy="adaptive")
        with pytest.raises(ValueError, match="No free blocks available"):
            queue.popleft()

    def test_append_after_adaptive_pop(self):
        """Blocks can be re-appended after adaptive eviction."""
        blocks = [KVCacheBlock(block_id=i) for i in range(3)]
        queue = FreeKVCacheBlockQueue(blocks, eviction_policy="adaptive")

        blocks[0]._block_hash = _make_block_hash(10)
        blocks[1]._block_hash = _make_block_hash(20)
        blocks[2]._block_hash = _make_block_hash(30)

        queue.increment_reuse(blocks[0])
        queue.increment_reuse(blocks[0])

        # Pop the lowest (block 1 or 2, both have 0 reuse)
        victim = queue.popleft()
        assert victim.block_id in (1, 2)
        assert queue.num_free_blocks == 2

        # Re-append it
        victim._block_hash = None  # reset hash as eviction would
        queue.append(victim)
        assert queue.num_free_blocks == 3

    def test_increment_reuse_tracks_per_hash(self):
        """Reuse counter is keyed by block hash, not block id."""
        blocks = [KVCacheBlock(block_id=i) for i in range(2)]
        queue = FreeKVCacheBlockQueue(blocks, eviction_policy="adaptive")

        same_hash = _make_block_hash(42)
        blocks[0]._block_hash = same_hash
        blocks[1]._block_hash = same_hash

        # Increment via block 0
        queue.increment_reuse(blocks[0])

        # Both blocks share the same hash → same reuse count
        assert queue.reuse_counter[same_hash] == 1

        # Increment via block 1 (same hash)
        queue.increment_reuse(blocks[1])
        assert queue.reuse_counter[same_hash] == 2

    def test_mixed_hashed_and_unhashed(self):
        """Mix of hashed and unhashed blocks: unhashed always go first."""
        blocks = [KVCacheBlock(block_id=i) for i in range(5)]
        queue = FreeKVCacheBlockQueue(blocks, eviction_policy="adaptive")

        # blocks 0, 2, 4 have hashes; blocks 1, 3 have no hash
        blocks[0]._block_hash = _make_block_hash(10)
        blocks[2]._block_hash = _make_block_hash(30)
        blocks[4]._block_hash = _make_block_hash(50)

        # First two pops should get unhashed blocks (1 and 3)
        v1 = queue.popleft()
        v2 = queue.popleft()
        unhashed_ids = {v1.block_id, v2.block_id}
        assert unhashed_ids == {1, 3}
        assert queue.num_free_blocks == 3

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
LRU-K eviction policy for KV cache offloading.

LRU-K is an improvement over standard LRU that tracks the last K access times
instead of just the most recent access. This provides better handling of
scan-resistant workloads and temporal locality patterns.

The algorithm works by:
1. Maintaining the last K access timestamps for each block
2. Evicting blocks with the oldest K-th access time
3. Blocks with fewer than K accesses are treated as having infinite K-th distance

Common configurations:
- K=2 (LRU-2): Best for most workloads, balances recency and frequency
- K=3 (LRU-3): More resistant to scans but higher memory overhead
- K=1 (LRU-1): Equivalent to standard LRU

Reference: O'Neil et al., "The LRU-K Page Replacement Algorithm For Database
Disk Buffering" (SIGMOD 1993)
"""

import time
from collections import deque
from collections.abc import Iterable
from typing import Optional

from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_offload.abstract import (
    LoadStoreSpec,
    OffloadingEvent,
    OffloadingManager,
    OffloadingStats,
    PrepareStoreOutput,
)
from vllm.v1.kv_offload.backend import Backend, BlockStatus


class LRUKOffloadingManager(OffloadingManager):
    """
    LRU-K eviction policy implementation.

    Tracks the last K access times for each block and evicts based on the
    K-th most recent access (the oldest of the K accesses).

    Args:
        backend: The offloading backend
        k: Number of historical accesses to track (default: 2)
        enable_events: Whether to enable event publishing
    """

    def __init__(
        self,
        backend: Backend,
        k: int = 2,
        enable_events: bool = False,
    ):
        self.backend = backend
        self.k = k
        self.blocks: dict[BlockHash, BlockStatus] = {}

        # Track last K access times for each block
        # Each entry is a deque of timestamps (newest on right)
        self.access_history: dict[BlockHash, deque[float]] = {}

        # Event tracking
        self.events: list[OffloadingEvent] | None = (
            [] if enable_events else None
        )
        self._stats = OffloadingStats()

    def lookup(self, block_hashes: Iterable[BlockHash]) -> int:
        """
        Check how many blocks are ready to use.

        Returns:
            Number of consecutive blocks that are ready, starting from the beginning
        """
        hit_count = 0
        total = 0
        for block_hash in block_hashes:
            total += 1
            block = self.blocks.get(block_hash)
            if block is None or not block.is_ready:
                break
            hit_count += 1
        self._stats.lookup_count += 1
        self._stats.hit_blocks += hit_count
        self._stats.miss_blocks += total - hit_count
        return hit_count

    def prepare_load(self, block_hashes: Iterable[BlockHash]) -> LoadStoreSpec:
        """
        Prepare to load blocks from offload storage.

        Args:
            block_hashes: List of block hashes to load

        Returns:
            LoadStoreSpec describing how to perform the load
        """
        blocks = []
        for block_hash in block_hashes:
            block = self.blocks[block_hash]
            assert block.is_ready
            block.ref_cnt += 1
            blocks.append(block)

        return self.backend.get_load_store_spec(block_hashes, blocks)

    def prepare_store(
        self, block_hashes: Iterable[BlockHash]
    ) -> Optional[PrepareStoreOutput]:
        """
        Prepare to store blocks, evicting if necessary.

        Evicts blocks with the oldest K-th access time. Blocks with fewer
        than K accesses are prioritized for eviction (treated as infinite distance).

        Args:
            block_hashes: List of block hashes to store

        Returns:
            PrepareStoreOutput with blocks to evict and store, or None if no space
        """
        block_hashes_list = list(block_hashes)

        # Filter out blocks that are already stored
        block_hashes_to_store = [
            block_hash
            for block_hash in block_hashes_list
            if block_hash not in self.blocks
        ]

        num_blocks_to_evict = (
            len(block_hashes_to_store) - self.backend.get_num_free_blocks()
        )

        # Build list of blocks to evict using LRU-K policy
        to_evict = []
        if num_blocks_to_evict > 0:
            # Blocks from the original input are excluded from eviction candidates
            protected = set(block_hashes_list)

            # Get eviction candidates sorted by LRU-K score
            candidates = self._get_eviction_candidates(protected)

            for block_hash in candidates:
                block = self.blocks[block_hash]
                if block.ref_cnt == 0:
                    to_evict.append(block_hash)
                    num_blocks_to_evict -= 1
                    if num_blocks_to_evict == 0:
                        break
            else:
                # Could not evict enough blocks
                return None

        # Evict blocks
        for block_hash in to_evict:
            self.backend.free(self.blocks.pop(block_hash))
            self.access_history.pop(block_hash, None)

        if to_evict and self.events is not None:
            self.events.append(
                OffloadingEvent(
                    block_hashes=to_evict,
                    block_size=self.backend.block_size,
                    medium=self.backend.medium,
                    removed=True,
                )
            )

        # Allocate blocks
        blocks = self.backend.allocate_blocks(block_hashes_to_store)
        assert len(blocks) == len(block_hashes_to_store)

        for block_hash, block in zip(block_hashes_to_store, blocks):
            self.blocks[block_hash] = block
            # Initialize access history with current time
            self.access_history[block_hash] = deque([time.time()], maxlen=self.k)

        # Build store specs for allocated blocks
        store_spec = self.backend.get_load_store_spec(block_hashes_to_store, blocks)

        self._stats.eviction_count += len(to_evict)
        self._stats.store_count += len(block_hashes_to_store)

        return PrepareStoreOutput(
            block_hashes_to_store=block_hashes_to_store,
            store_spec=store_spec,
            block_hashes_evicted=to_evict,
        )

    def _get_eviction_candidates(
        self, protected: set[BlockHash]
    ) -> list[BlockHash]:
        """
        Get blocks sorted by eviction priority (LRU-K).

        Returns blocks sorted with highest priority (should evict first) at the front.
        """
        candidates = []

        for block_hash in self.blocks.keys():
            if block_hash in protected:
                continue

            history = self.access_history.get(block_hash)
            if history is None or len(history) == 0:
                # No history, treat as oldest possible
                kth_time = 0.0
            elif len(history) < self.k:
                # Insufficient history, use oldest access time
                # These are prioritized for eviction
                kth_time = history[0]
            else:
                # Use K-th most recent access (oldest in the history)
                kth_time = history[0]

            candidates.append((kth_time, block_hash))

        # Sort by K-th access time (oldest first)
        candidates.sort()

        return [block_hash for _, block_hash in candidates]

    def touch(self, block_hashes: Iterable[BlockHash]) -> None:
        """
        Update access history for blocks.

        Records the current timestamp and maintains the last K accesses.

        Args:
            block_hashes: List of block hashes that were accessed
        """
        current_time = time.time()

        for block_hash in block_hashes:
            if block_hash not in self.blocks:
                continue

            # Update access history
            if block_hash not in self.access_history:
                self.access_history[block_hash] = deque(maxlen=self.k)

            # Append new access time (deque automatically evicts oldest if full)
            self.access_history[block_hash].append(current_time)

    def complete_store(
        self, block_hashes: Iterable[BlockHash], success: bool = True
    ) -> None:
        """
        Complete a store operation.

        Args:
            block_hashes: Blocks that were stored
            success: Whether the store succeeded
        """
        stored_block_hashes: list[BlockHash] = []
        if success:
            for block_hash in block_hashes:
                block = self.blocks[block_hash]
                if not block.is_ready:
                    block.ref_cnt = 0
                    stored_block_hashes.append(block_hash)
        else:
            for block_hash in block_hashes:
                block = self.blocks[block_hash]
                if not block.is_ready:
                    self.backend.free(block)
                    del self.blocks[block_hash]
                    self.access_history.pop(block_hash, None)

        if stored_block_hashes and self.events is not None:
            self.events.append(
                OffloadingEvent(
                    block_hashes=stored_block_hashes,
                    block_size=self.backend.block_size,
                    medium=self.backend.medium,
                    removed=False,
                )
            )

    def complete_load(self, block_hashes: Iterable[BlockHash]) -> None:
        """
        Complete a load operation.

        Args:
            block_hashes: Blocks that were loaded
        """
        for block_hash in block_hashes:
            block = self.blocks[block_hash]
            assert block.ref_cnt > 0
            block.ref_cnt -= 1

    def take_events(self) -> Iterable[OffloadingEvent]:
        """
        Retrieve and clear pending events.

        Returns:
            Iterator over OffloadingEvents
        """
        if self.events is not None:
            yield from self.events
            self.events.clear()

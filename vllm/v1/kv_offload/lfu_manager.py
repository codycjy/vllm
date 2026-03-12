# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections import OrderedDict
from collections.abc import Iterable

from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_offload.abstract import (
    LoadStoreSpec,
    OffloadingEvent,
    OffloadingManager,
    PrepareStoreOutput,
)
from vllm.v1.kv_offload.backend import Backend, BlockStatus


class LFUOffloadingManager(OffloadingManager):
    """
    An OffloadingManager with a pluggable backend, which evicts blocks by LFU
    (Least Frequently Used).

    This implementation tracks access frequency for each block and evicts
    the least frequently accessed block when space is needed.

    When multiple blocks have the same frequency, it uses LRU as a tie-breaker
    (evicts the least recently used among blocks with the same frequency).
    """

    def __init__(self, backend: Backend, enable_events: bool = False):
        self.backend: Backend = backend
        # block_hash -> BlockStatus
        self.blocks: dict[BlockHash, BlockStatus] = {}
        # block_hash -> access frequency count
        self.frequencies: dict[BlockHash, int] = {}
        # frequency -> OrderedDict[block_hash -> None] (for LRU tie-breaking)
        self.freq_lists: dict[int, OrderedDict[BlockHash, None]] = {}
        self.min_freq: int = 0
        self.events: list[OffloadingEvent] | None = [] if enable_events else None

    def lookup(self, block_hashes: Iterable[BlockHash]) -> int | None:
        hit_count = 0
        for block_hash in block_hashes:
            block = self.blocks.get(block_hash)
            if block is None or not block.is_ready:
                break
            hit_count += 1
        return hit_count

    def prepare_load(self, block_hashes: Iterable[BlockHash]) -> LoadStoreSpec:
        blocks = []
        for block_hash in block_hashes:
            block = self.blocks[block_hash]
            assert block.is_ready
            block.ref_cnt += 1
            blocks.append(block)

        return self.backend.get_load_store_spec(block_hashes, blocks)

    def _update_freq(self, block_hash: BlockHash):
        """Update the frequency of a block and move it to the appropriate freq list."""
        if block_hash not in self.frequencies:
            return

        old_freq = self.frequencies[block_hash]
        new_freq = old_freq + 1

        # Remove from old frequency list
        if old_freq in self.freq_lists:
            self.freq_lists[old_freq].pop(block_hash, None)
            # If old freq list is empty and it was the min_freq, update min_freq
            if not self.freq_lists[old_freq] and old_freq == self.min_freq:
                self.min_freq = new_freq
            # Clean up empty frequency lists
            if not self.freq_lists[old_freq]:
                del self.freq_lists[old_freq]

        # Update frequency
        self.frequencies[block_hash] = new_freq

        # Add to new frequency list
        if new_freq not in self.freq_lists:
            self.freq_lists[new_freq] = OrderedDict()
        self.freq_lists[new_freq][block_hash] = None

    def touch(self, block_hashes: Iterable[BlockHash]):
        """Update access frequency for touched blocks."""
        for block_hash in reversed(list(block_hashes)):
            if block_hash in self.blocks:
                self._update_freq(block_hash)

    def complete_load(self, block_hashes: Iterable[BlockHash]):
        for block_hash in block_hashes:
            block = self.blocks[block_hash]
            assert block.ref_cnt > 0
            block.ref_cnt -= 1

    def prepare_store(
        self, block_hashes: Iterable[BlockHash]
    ) -> PrepareStoreOutput | None:
        block_hashes_list = list(block_hashes)

        # filter out blocks that are already stored
        block_hashes_to_store = [
            block_hash
            for block_hash in block_hashes_list
            if block_hash not in self.blocks
        ]

        num_blocks_to_evict = (
            len(block_hashes_to_store) - self.backend.get_num_free_blocks()
        )

        # build list of blocks to evict
        to_evict = []
        if num_blocks_to_evict > 0:
            # Blocks from the original input are excluded from eviction candidates:
            # a block that was already stored must remain in the cache after this call.
            protected = set(block_hashes_list)

            # Find blocks to evict using LFU strategy
            # Start from the minimum frequency and work upwards
            for freq in sorted(self.freq_lists.keys()):
                if num_blocks_to_evict <= 0:
                    break

                # Within same frequency, evict LRU (oldest first)
                for block_hash in list(self.freq_lists[freq].keys()):
                    if num_blocks_to_evict <= 0:
                        break

                    block = self.blocks.get(block_hash)
                    if block and block.ref_cnt == 0 and block_hash not in protected:
                        to_evict.append(block_hash)
                        num_blocks_to_evict -= 1

            # If we couldn't evict enough blocks, return None
            if num_blocks_to_evict > 0:
                return None

        # evict blocks
        for block_hash in to_evict:
            block = self.blocks.pop(block_hash)
            self.backend.free(block)

            # Remove from frequency tracking
            freq = self.frequencies.pop(block_hash, None)
            if freq is not None and freq in self.freq_lists:
                self.freq_lists[freq].pop(block_hash, None)
                if not self.freq_lists[freq]:
                    del self.freq_lists[freq]

        if to_evict and self.events is not None:
            self.events.append(
                OffloadingEvent(
                    block_hashes=to_evict,
                    block_size=self.backend.block_size,
                    medium=self.backend.medium,
                    removed=True,
                )
            )

        blocks = self.backend.allocate_blocks(block_hashes_to_store)
        assert len(blocks) == len(block_hashes_to_store)

        # Add new blocks with initial frequency of 1
        for block_hash, block in zip(block_hashes_to_store, blocks):
            self.blocks[block_hash] = block
            self.frequencies[block_hash] = 1

            # Add to frequency list for freq=1
            if 1 not in self.freq_lists:
                self.freq_lists[1] = OrderedDict()
            self.freq_lists[1][block_hash] = None

            # Update min_freq if needed
            self.min_freq = 1

        # build store specs for allocated blocks
        store_spec = self.backend.get_load_store_spec(block_hashes_to_store, blocks)

        return PrepareStoreOutput(
            block_hashes_to_store=block_hashes_to_store,
            store_spec=store_spec,
            block_hashes_evicted=to_evict,
        )

    def complete_store(self, block_hashes: Iterable[BlockHash], success: bool = True):
        stored_block_hashes: list[BlockHash] = []
        if success:
            for block_hash in block_hashes:
                block = self.blocks.get(block_hash)
                if block and not block.is_ready:
                    block.ref_cnt = 0
                    stored_block_hashes.append(block_hash)
        else:
            for block_hash in block_hashes:
                block = self.blocks.get(block_hash)
                if block and not block.is_ready:
                    self.backend.free(block)
                    del self.blocks[block_hash]
                    # Clean up frequency tracking
                    freq = self.frequencies.pop(block_hash, None)
                    if freq is not None and freq in self.freq_lists:
                        self.freq_lists[freq].pop(block_hash, None)
                        if not self.freq_lists[freq]:
                            del self.freq_lists[freq]

        if stored_block_hashes and self.events is not None:
            self.events.append(
                OffloadingEvent(
                    block_hashes=stored_block_hashes,
                    block_size=self.backend.block_size,
                    medium=self.backend.medium,
                    removed=False,
                )
            )

    def take_events(self) -> Iterable[OffloadingEvent]:
        if self.events is not None:
            yield from self.events
            self.events.clear()

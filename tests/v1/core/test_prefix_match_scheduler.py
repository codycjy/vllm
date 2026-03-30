# SPDX-License-Identifier: Apache-2.0
"""Tests for PrefixMatchRequestQueue scheduling policy.

These tests verify:
- Requests with higher prefix cache hit ratio are scheduled first.
- Aging mechanism prevents starvation of low-match requests.
- Edge cases: empty queue, remove, prepend, all-zero matches.

No GPU required — pure Python unit tests with mocked KVCacheManager.
"""

import time
from unittest.mock import MagicMock, patch

import pytest

from vllm.v1.core.sched.request_queue import (
    PrefixMatchRequestQueue,
    SchedulingPolicy,
    create_request_queue,
)

pytestmark = pytest.mark.cpu_test


def _make_request(request_id: str, num_tokens: int = 100,
                  arrival_time: float | None = None):
    """Create a mock Request with the fields needed by PrefixMatchRequestQueue."""
    req = MagicMock()
    req.request_id = request_id
    req.num_tokens = num_tokens
    req.arrival_time = arrival_time if arrival_time is not None else time.monotonic()
    return req


def _make_kv_cache_manager(hit_map: dict[str, int] | None = None):
    """Create a mock KVCacheManager.

    Args:
        hit_map: mapping from request_id -> number of cached tokens.
                 Defaults to 0 for unknown requests.
    """
    if hit_map is None:
        hit_map = {}
    manager = MagicMock()

    def get_computed_blocks(request):
        num_cached = hit_map.get(request.request_id, 0)
        return MagicMock(), num_cached  # (KVCacheBlocks, int)

    manager.get_computed_blocks = MagicMock(side_effect=get_computed_blocks)
    return manager


# ─── Prefix match ordering tests ─────────────────────────────────────


class TestPrefixMatchOrdering:
    """Verify that higher cache hit requests are scheduled first."""

    def test_high_hit_scheduled_first(self):
        """Request with more cached tokens should come first."""
        hit_map = {"A": 80, "B": 10, "C": 50}
        mgr = _make_kv_cache_manager(hit_map)
        queue = PrefixMatchRequestQueue(mgr, max_wait_seconds=9999)

        now = time.monotonic()
        req_a = _make_request("A", num_tokens=100, arrival_time=now)
        req_b = _make_request("B", num_tokens=100, arrival_time=now)
        req_c = _make_request("C", num_tokens=100, arrival_time=now)

        queue.add_request(req_b)
        queue.add_request(req_a)
        queue.add_request(req_c)

        # Should be ordered: A(0.8), C(0.5), B(0.1)
        assert queue.pop_request().request_id == "A"
        assert queue.pop_request().request_id == "C"
        assert queue.pop_request().request_id == "B"

    def test_peek_returns_best(self):
        """peek_request should return the highest-priority request."""
        hit_map = {"X": 90, "Y": 10}
        mgr = _make_kv_cache_manager(hit_map)
        queue = PrefixMatchRequestQueue(mgr, max_wait_seconds=9999)

        now = time.monotonic()
        queue.add_request(_make_request("Y", num_tokens=100, arrival_time=now))
        queue.add_request(_make_request("X", num_tokens=100, arrival_time=now))

        assert queue.peek_request().request_id == "X"
        # peek should not remove
        assert len(queue) == 2


class TestAgingMechanism:
    """Verify that aging prevents starvation."""

    def test_aging_boosts_old_request(self):
        """A request waiting beyond max_wait should be boosted above
        a high-hit request that arrived recently."""
        hit_map = {"old": 0, "new": 90}
        mgr = _make_kv_cache_manager(hit_map)
        queue = PrefixMatchRequestQueue(mgr, max_wait_seconds=5.0)

        now = time.monotonic()
        # "old" arrived 20 seconds ago: aging_bonus = (20 - 5) * 0.1 = 1.5
        # score = 0/100 + 1.5 = 1.5
        req_old = _make_request("old", num_tokens=100, arrival_time=now - 20)
        # "new" arrived just now: score = 90/100 + 0 = 0.9
        req_new = _make_request("new", num_tokens=100, arrival_time=now)

        queue.add_request(req_new)
        queue.add_request(req_old)

        # old should win due to aging bonus
        assert queue.pop_request().request_id == "old"

    def test_no_aging_within_threshold(self):
        """Requests within max_wait should be ordered purely by cache hit."""
        hit_map = {"A": 80, "B": 20}
        mgr = _make_kv_cache_manager(hit_map)
        queue = PrefixMatchRequestQueue(mgr, max_wait_seconds=60.0)

        now = time.monotonic()
        # Both arrived recently (within threshold)
        req_a = _make_request("A", num_tokens=100, arrival_time=now - 10)
        req_b = _make_request("B", num_tokens=100, arrival_time=now - 15)

        queue.add_request(req_b)
        queue.add_request(req_a)

        # A has higher hit ratio, no aging yet
        assert queue.pop_request().request_id == "A"


class TestDegradesToFCFS:
    """When all cache hits are 0, behavior should approximate FCFS."""

    def test_zero_hits_preserves_order(self):
        """With no cache hits and no aging, original insertion order is kept."""
        mgr = _make_kv_cache_manager({})
        queue = PrefixMatchRequestQueue(mgr, max_wait_seconds=9999)

        now = time.monotonic()
        reqs = [_make_request(f"R{i}", num_tokens=100, arrival_time=now)
                for i in range(5)]
        for r in reqs:
            queue.add_request(r)

        # All scores are 0, tie-break by original insertion order
        for i in range(5):
            assert queue.pop_request().request_id == f"R{i}"


class TestQueueOperations:
    """Verify standard queue operations."""

    def test_empty_queue_peek_raises(self):
        mgr = _make_kv_cache_manager({})
        queue = PrefixMatchRequestQueue(mgr)
        with pytest.raises(IndexError):
            queue.peek_request()

    def test_empty_queue_bool(self):
        mgr = _make_kv_cache_manager({})
        queue = PrefixMatchRequestQueue(mgr)
        assert not queue
        queue.add_request(_make_request("X"))
        assert queue

    def test_len(self):
        mgr = _make_kv_cache_manager({})
        queue = PrefixMatchRequestQueue(mgr)
        assert len(queue) == 0
        queue.add_request(_make_request("A"))
        queue.add_request(_make_request("B"))
        assert len(queue) == 2

    def test_remove_request(self):
        mgr = _make_kv_cache_manager({})
        queue = PrefixMatchRequestQueue(mgr)
        req_a = _make_request("A")
        req_b = _make_request("B")
        queue.add_request(req_a)
        queue.add_request(req_b)

        queue.remove_request(req_a)
        assert len(queue) == 1
        assert queue.pop_request().request_id == "B"

    def test_remove_requests_batch(self):
        mgr = _make_kv_cache_manager({})
        queue = PrefixMatchRequestQueue(mgr)
        reqs = [_make_request(f"R{i}") for i in range(5)]
        for r in reqs:
            queue.add_request(r)

        queue.remove_requests([reqs[1], reqs[3]])
        assert len(queue) == 3

    def test_prepend_triggers_resort(self):
        """After prepend, queue should re-sort on next peek."""
        hit_map = {"A": 10, "B": 50, "C": 90}
        mgr = _make_kv_cache_manager(hit_map)
        queue = PrefixMatchRequestQueue(mgr, max_wait_seconds=9999)

        now = time.monotonic()
        queue.add_request(_make_request("A", num_tokens=100, arrival_time=now))
        queue.add_request(_make_request("B", num_tokens=100, arrival_time=now))

        # Force sort
        _ = queue.peek_request()

        # Prepend a high-hit request
        queue.prepend_request(
            _make_request("C", num_tokens=100, arrival_time=now))

        # Should re-sort: C(0.9) > B(0.5) > A(0.1)
        assert queue.peek_request().request_id == "C"

    def test_prepend_requests_from_another_queue(self):
        """prepend_requests should accept another RequestQueue."""
        hit_map = {"A": 10, "B": 90}
        mgr = _make_kv_cache_manager(hit_map)
        queue = PrefixMatchRequestQueue(mgr, max_wait_seconds=9999)

        from vllm.v1.core.sched.request_queue import FCFSRequestQueue
        other = FCFSRequestQueue()

        now = time.monotonic()
        queue.add_request(_make_request("A", num_tokens=100, arrival_time=now))
        other.add_request(_make_request("B", num_tokens=100, arrival_time=now))

        queue.prepend_requests(other)
        assert len(queue) == 2
        # B has higher hit, should come first after sort
        assert queue.peek_request().request_id == "B"

    def test_iter_sorted(self):
        """Iteration should yield requests in sorted order."""
        hit_map = {"X": 30, "Y": 70, "Z": 50}
        mgr = _make_kv_cache_manager(hit_map)
        queue = PrefixMatchRequestQueue(mgr, max_wait_seconds=9999)

        now = time.monotonic()
        for name in ["X", "Y", "Z"]:
            queue.add_request(
                _make_request(name, num_tokens=100, arrival_time=now))

        ids = [r.request_id for r in queue]
        assert ids == ["Y", "Z", "X"]


class TestFactoryFunction:
    """Verify create_request_queue supports prefix_match."""

    def test_create_prefix_match_queue(self):
        mgr = _make_kv_cache_manager({})
        queue = create_request_queue(
            SchedulingPolicy.PREFIX_MATCH,
            kv_cache_manager=mgr,
            max_wait_seconds=10.0,
        )
        assert isinstance(queue, PrefixMatchRequestQueue)

    def test_create_prefix_match_without_manager_raises(self):
        with pytest.raises(AssertionError):
            create_request_queue(SchedulingPolicy.PREFIX_MATCH)

    def test_create_fcfs_still_works(self):
        from vllm.v1.core.sched.request_queue import FCFSRequestQueue
        queue = create_request_queue(SchedulingPolicy.FCFS)
        assert isinstance(queue, FCFSRequestQueue)


class TestSchedulerConfigIntegration:
    """Verify config changes are correct."""

    def test_scheduling_policy_literal_includes_prefix_match(self):
        from typing import get_args
        from vllm.config.scheduler import SchedulerPolicy
        assert "prefix_match" in get_args(SchedulerPolicy)

    def test_scheduling_max_wait_default(self):
        from vllm.config.scheduler import SchedulerConfig
        cfg = SchedulerConfig.default_factory()
        assert cfg.scheduling_max_wait == 30.0

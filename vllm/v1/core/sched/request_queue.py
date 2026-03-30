# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import heapq
import time
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterable, Iterator
from enum import Enum
from typing import TYPE_CHECKING

from vllm.v1.request import Request

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheManager


class SchedulingPolicy(Enum):
    """Enum for scheduling policies."""

    FCFS = "fcfs"
    PRIORITY = "priority"
    PREFIX_MATCH = "prefix_match"


class RequestQueue(ABC):
    """Abstract base class for request queues."""

    @abstractmethod
    def add_request(self, request: Request) -> None:
        """Add a request to the queue according to the policy."""
        pass

    @abstractmethod
    def pop_request(self) -> Request:
        """Pop a request from the queue according to the policy."""
        pass

    @abstractmethod
    def peek_request(self) -> Request:
        """Peek at the request at the front of the queue without removing it."""
        pass

    @abstractmethod
    def prepend_request(self, request: Request) -> None:
        """Prepend a request to the front of the queue."""
        pass

    @abstractmethod
    def prepend_requests(self, requests: "RequestQueue") -> None:
        """Prepend all requests from another queue to the front of this
        queue."""
        pass

    @abstractmethod
    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        pass

    @abstractmethod
    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue."""
        pass

    @abstractmethod
    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Get number of requests in queue."""
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[Request]:
        """Iterate over the queue according to the policy."""
        pass


class FCFSRequestQueue(deque[Request], RequestQueue):
    """A first-come-first-served queue that supports deque operations."""

    def add_request(self, request: Request) -> None:
        """Add a request to the queue according to FCFS policy."""
        self.append(request)

    def pop_request(self) -> Request:
        """Pop a request from the queue according to FCFS policy."""
        return self.popleft()

    def peek_request(self) -> Request:
        """Peek at the next request in the queue without removing it."""
        if not self:
            raise IndexError("peek from an empty queue")
        return self[0]

    def prepend_request(self, request: Request) -> None:
        """Prepend a request to the front of the queue."""
        self.appendleft(request)

    def prepend_requests(self, requests: RequestQueue) -> None:
        """Prepend all requests from another queue to the front of this
        queue.

        Note: The requests will be prepended in reverse order of their
        appearance in the `requests` queue.
        """
        self.extendleft(requests)

    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        self.remove(request)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue."""
        requests_to_remove = set(requests)
        filtered_requests = [req for req in self if req not in requests_to_remove]
        # deque does not support in-place filtering, so we need to clear
        # and extend
        self.clear()
        self.extend(filtered_requests)

    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        return len(self) > 0

    def __len__(self) -> int:
        """Get number of requests in queue."""
        return super().__len__()

    def __iter__(self) -> Iterator[Request]:
        """Iterate over the queue according to FCFS policy."""
        return super().__iter__()


class PriorityRequestQueue(RequestQueue):
    """
    A priority queue that supports heap operations.

    Respects the ordering defined in the Request class, where
    requests with a smaller value of `priority` are processed first.
    If multiple requests have the same priority, the one with the earlier
    `arrival_time` is processed first.
    """

    def __init__(self) -> None:
        self._heap: list[Request] = []

    def add_request(self, request: Request) -> None:
        """Add a request to the queue according to priority policy."""
        heapq.heappush(self._heap, request)

    def pop_request(self) -> Request:
        """Pop a request from the queue according to priority policy."""
        if not self._heap:
            raise IndexError("pop from empty heap")
        return heapq.heappop(self._heap)

    def peek_request(self) -> Request:
        """Peek at the next request in the queue without removing it."""
        if not self._heap:
            raise IndexError("peek from empty heap")
        return self._heap[0]

    def prepend_request(self, request: Request) -> None:
        """Add a request to the queue according to priority policy.

        Note: In a priority queue, there is no concept of prepending to the
        front. Requests are ordered by (priority, arrival_time)."""
        self.add_request(request)

    def prepend_requests(self, requests: RequestQueue) -> None:
        """Add all requests from another queue according to priority policy.

        Note: In a priority queue, there is no concept of prepending to the
        front. Requests are ordered by (priority, arrival_time)."""
        for request in requests:
            self.add_request(request)

    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        self._heap.remove(request)
        heapq.heapify(self._heap)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue."""
        requests_to_remove = requests if isinstance(requests, set) else set(requests)
        self._heap = [r for r in self._heap if r not in requests_to_remove]
        heapq.heapify(self._heap)

    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        return bool(self._heap)

    def __len__(self) -> int:
        """Get number of requests in queue."""
        return len(self._heap)

    def __iter__(self) -> Iterator[Request]:
        """Iterate over the queue according to priority policy."""
        heap_copy = self._heap[:]
        while heap_copy:
            yield heapq.heappop(heap_copy)


class PrefixMatchRequestQueue(RequestQueue):
    """A request queue that prioritizes requests with higher prefix cache
    hit ratios. Includes aging mechanism to prevent starvation of requests
    with low cache hits.

    Requests are sorted by a score combining prefix cache match ratio and
    an aging bonus for requests that have waited beyond max_wait_seconds.
    """

    def __init__(
        self,
        kv_cache_manager: KVCacheManager,
        max_wait_seconds: float = 30.0,
    ) -> None:
        self._queue: deque[Request] = deque()
        self._kv_cache_manager = kv_cache_manager
        self._max_wait_seconds = max_wait_seconds
        self._sorted = False

    def _compute_schedule_score(self, request: Request,
                                now: float) -> float:
        """Compute scheduling priority score (higher = schedule sooner).

        score = prefix_match_ratio + aging_bonus
        - prefix_match_ratio: cached tokens / total tokens (0~1)
        - aging_bonus: after max_wait, +0.1 per extra second
        """
        _, num_cached_tokens = (
            self._kv_cache_manager.get_computed_blocks(request))
        prefix_match_ratio = num_cached_tokens / max(request.num_tokens, 1)

        wait_time = now - request.arrival_time
        aging_bonus = 0.0
        if wait_time > self._max_wait_seconds:
            aging_bonus = (wait_time - self._max_wait_seconds) * 0.1

        return prefix_match_ratio + aging_bonus

    def _sort_queue(self) -> None:
        """Sort the queue by schedule score (descending)."""
        now = time.monotonic()
        scored = [(self._compute_schedule_score(req, now), i, req)
                  for i, req in enumerate(self._queue)]
        # Higher score first; tie-break by original order (stable)
        scored.sort(key=lambda x: (-x[0], x[1]))

        self._queue.clear()
        self._queue.extend(item[2] for item in scored)
        self._sorted = True

    def add_request(self, request: Request) -> None:
        """Add a request; mark queue as needing re-sort."""
        self._queue.append(request)
        self._sorted = False

    def pop_request(self) -> Request:
        """Sort if needed, then pop the highest-priority request."""
        if not self._sorted:
            self._sort_queue()
        return self._queue.popleft()

    def peek_request(self) -> Request:
        """Sort if needed, then peek at the highest-priority request."""
        if not self._queue:
            raise IndexError("peek from an empty queue")
        if not self._sorted:
            self._sort_queue()
        return self._queue[0]

    def prepend_request(self, request: Request) -> None:
        """Add a request back; will be re-sorted on next peek/pop."""
        self._queue.appendleft(request)
        self._sorted = False

    def prepend_requests(self, requests: RequestQueue) -> None:
        """Add all requests back; will be re-sorted on next peek/pop."""
        self._queue.extendleft(requests)
        self._sorted = False

    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        self._queue.remove(request)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple requests from the queue."""
        requests_to_remove = set(requests)
        filtered = [r for r in self._queue if r not in requests_to_remove]
        self._queue.clear()
        self._queue.extend(filtered)

    def __bool__(self) -> bool:
        return len(self._queue) > 0

    def __len__(self) -> int:
        return len(self._queue)

    def __iter__(self) -> Iterator[Request]:
        """Iterate in sorted order."""
        if not self._sorted:
            self._sort_queue()
        return iter(self._queue)


def create_request_queue(
    policy: SchedulingPolicy,
    kv_cache_manager: KVCacheManager | None = None,
    max_wait_seconds: float = 30.0,
) -> RequestQueue:
    """Create request queue based on scheduling policy."""
    if policy == SchedulingPolicy.PRIORITY:
        return PriorityRequestQueue()
    elif policy == SchedulingPolicy.FCFS:
        return FCFSRequestQueue()
    elif policy == SchedulingPolicy.PREFIX_MATCH:
        assert kv_cache_manager is not None, (
            "kv_cache_manager is required for prefix_match scheduling policy")
        return PrefixMatchRequestQueue(kv_cache_manager, max_wait_seconds)
    else:
        raise ValueError(f"Unknown scheduling policy: {policy}")

#!/usr/bin/env python3
"""Standalone test for PrefixMatchRequestQueue — no pytest needed."""

import sys
import time
from unittest.mock import MagicMock

# Ensure the vllm package is importable
sys.path.insert(0, "/ocean/projects/cis250265p/xli45/opensource/dev/vllm")

from vllm.v1.core.sched.request_queue import (
    FCFSRequestQueue,
    PrefixMatchRequestQueue,
    SchedulingPolicy,
    create_request_queue,
)


def _make_request(request_id, num_tokens=100, arrival_time=None):
    req = MagicMock()
    req.request_id = request_id
    req.num_tokens = num_tokens
    req.arrival_time = arrival_time if arrival_time is not None else time.monotonic()
    return req


def _make_mgr(hit_map=None):
    if hit_map is None:
        hit_map = {}
    mgr = MagicMock()
    def get_computed_blocks(request):
        return MagicMock(), hit_map.get(request.request_id, 0)
    mgr.get_computed_blocks = MagicMock(side_effect=get_computed_blocks)
    return mgr


passed = 0
failed = 0

def check(name, condition):
    global passed, failed
    if condition:
        print(f"  PASS: {name}")
        passed += 1
    else:
        print(f"  FAIL: {name}")
        failed += 1


# ─── Test 1: High hit scheduled first ─────────────────────────────
print("\n[Test 1] High hit scheduled first")
mgr = _make_mgr({"A": 80, "B": 10, "C": 50})
q = PrefixMatchRequestQueue(mgr, max_wait_seconds=9999)
now = time.monotonic()
q.add_request(_make_request("B", 100, now))
q.add_request(_make_request("A", 100, now))
q.add_request(_make_request("C", 100, now))
check("A first", q.pop_request().request_id == "A")
check("C second", q.pop_request().request_id == "C")
check("B third", q.pop_request().request_id == "B")

# ─── Test 2: Peek returns best without removing ───────────────────
print("\n[Test 2] Peek returns best")
mgr = _make_mgr({"X": 90, "Y": 10})
q = PrefixMatchRequestQueue(mgr, max_wait_seconds=9999)
now = time.monotonic()
q.add_request(_make_request("Y", 100, now))
q.add_request(_make_request("X", 100, now))
check("peek is X", q.peek_request().request_id == "X")
check("len still 2", len(q) == 2)

# ─── Test 3: Aging boosts old request ─────────────────────────────
print("\n[Test 3] Aging boosts old request")
mgr = _make_mgr({"old": 0, "new": 90})
q = PrefixMatchRequestQueue(mgr, max_wait_seconds=5.0)
now = time.monotonic()
q.add_request(_make_request("new", 100, now))
q.add_request(_make_request("old", 100, now - 20))  # waited 20s, bonus=1.5
check("old wins via aging", q.pop_request().request_id == "old")

# ─── Test 4: No aging within threshold ────────────────────────────
print("\n[Test 4] No aging within threshold")
mgr = _make_mgr({"A": 80, "B": 20})
q = PrefixMatchRequestQueue(mgr, max_wait_seconds=60.0)
now = time.monotonic()
q.add_request(_make_request("B", 100, now - 15))
q.add_request(_make_request("A", 100, now - 10))
check("A wins (higher hit, no aging)", q.pop_request().request_id == "A")

# ─── Test 5: Zero hits preserves insertion order ──────────────────
print("\n[Test 5] Zero hits preserves order")
mgr = _make_mgr({})
q = PrefixMatchRequestQueue(mgr, max_wait_seconds=9999)
now = time.monotonic()
for i in range(5):
    q.add_request(_make_request(f"R{i}", 100, now))
ids = [q.pop_request().request_id for _ in range(5)]
check("order preserved", ids == ["R0", "R1", "R2", "R3", "R4"])

# ─── Test 6: Empty queue ──────────────────────────────────────────
print("\n[Test 6] Empty queue operations")
mgr = _make_mgr({})
q = PrefixMatchRequestQueue(mgr)
check("empty is falsy", not q)
try:
    q.peek_request()
    check("peek raises IndexError", False)
except IndexError:
    check("peek raises IndexError", True)

# ─── Test 7: Remove request ──────────────────────────────────────
print("\n[Test 7] Remove request")
mgr = _make_mgr({})
q = PrefixMatchRequestQueue(mgr)
a, b = _make_request("A"), _make_request("B")
q.add_request(a)
q.add_request(b)
q.remove_request(a)
check("len after remove", len(q) == 1)
check("remaining is B", q.pop_request().request_id == "B")

# ─── Test 8: Prepend triggers resort ─────────────────────────────
print("\n[Test 8] Prepend triggers resort")
mgr = _make_mgr({"A": 10, "B": 50, "C": 90})
q = PrefixMatchRequestQueue(mgr, max_wait_seconds=9999)
now = time.monotonic()
q.add_request(_make_request("A", 100, now))
q.add_request(_make_request("B", 100, now))
_ = q.peek_request()  # force sort
q.prepend_request(_make_request("C", 100, now))
check("C on top after prepend", q.peek_request().request_id == "C")

# ─── Test 9: Factory function ────────────────────────────────────
print("\n[Test 9] Factory function")
mgr = _make_mgr({})
q = create_request_queue(SchedulingPolicy.PREFIX_MATCH, kv_cache_manager=mgr)
check("creates PrefixMatchRequestQueue", isinstance(q, PrefixMatchRequestQueue))
q2 = create_request_queue(SchedulingPolicy.FCFS)
check("FCFS still works", isinstance(q2, FCFSRequestQueue))
try:
    create_request_queue(SchedulingPolicy.PREFIX_MATCH)
    check("no manager raises", False)
except AssertionError:
    check("no manager raises", True)

# ─── Test 10: Iter sorted ────────────────────────────────────────
print("\n[Test 10] Iter sorted")
mgr = _make_mgr({"X": 30, "Y": 70, "Z": 50})
q = PrefixMatchRequestQueue(mgr, max_wait_seconds=9999)
now = time.monotonic()
for n in ["X", "Y", "Z"]:
    q.add_request(_make_request(n, 100, now))
ids = [r.request_id for r in q]
check("iter order Y,Z,X", ids == ["Y", "Z", "X"])

# ─── Test 11: Config ─────────────────────────────────────────────
print("\n[Test 11] Config integration")
from typing import get_args
from vllm.config.scheduler import SchedulerPolicy
check("prefix_match in SchedulerPolicy", "prefix_match" in get_args(SchedulerPolicy))

# ─── Summary ──────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"Results: {passed} passed, {failed} failed, {passed+failed} total")
if failed:
    sys.exit(1)
else:
    print("All tests PASSED!")
    sys.exit(0)

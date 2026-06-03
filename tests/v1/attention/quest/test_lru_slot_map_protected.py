# SPDX-License-Identifier: Apache-2.0
"""Stage 3: _LRUSlotMap protected-victim eviction (mixture policy core)."""
from __future__ import annotations

from vllm.v1.attention.backends.quest.cache.tier_manager import _LRUSlotMap


def test_protected_none_is_pure_lru():
    # Regression anchor: protected=None must behave exactly as before.
    m = _LRUSlotMap(capacity=2)
    s0, e0 = m.add((0, 0))
    s1, e1 = m.add((0, 1))
    assert (e0, e1) == (None, None)
    # full; next add evicts the LRU tail = (0, 0)
    s2, e2 = m.add((0, 2), protected=None)
    assert e2 == (0, 0)


def test_protected_skips_protected_block():
    m = _LRUSlotMap(capacity=2)
    m.add((0, 0))  # oldest
    m.add((0, 1))  # newer
    # (0,0) is LRU tail, but it's protected -> victim must be (0,1) instead.
    slot, evicted = m.add((0, 2), protected={(0, 0)})
    assert evicted == (0, 1)


def test_protected_falls_back_when_all_protected():
    m = _LRUSlotMap(capacity=2)
    m.add((0, 0))
    m.add((0, 1))
    # both protected -> must still evict (no free slot); pick protected LRU tail
    slot, evicted = m.add((0, 2), protected={(0, 0), (0, 1)})
    assert evicted == (0, 0)  # LRU tail among the protected set


def test_protected_picks_oldest_nonprotected():
    m = _LRUSlotMap(capacity=3)
    m.add((0, 0))  # oldest
    m.add((0, 1))
    m.add((0, 2))  # newest
    # protect the two oldest; only (0,2) is evictable
    slot, evicted = m.add((0, 3), protected={(0, 0), (0, 1)})
    assert evicted == (0, 2)


def test_protected_ignored_when_free_slot_exists():
    m = _LRUSlotMap(capacity=4)
    m.add((0, 0))
    # free slots remain -> no eviction regardless of protected
    slot, evicted = m.add((0, 1), protected={(0, 0)})
    assert evicted is None


def test_existing_key_returns_without_eviction():
    m = _LRUSlotMap(capacity=2)
    s0, _ = m.add((0, 0))
    m.add((0, 1))
    slot, evicted = m.add((0, 0), protected={(0, 1)})
    assert evicted is None and slot == s0

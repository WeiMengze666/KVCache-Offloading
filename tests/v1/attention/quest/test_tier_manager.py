# SPDX-License-Identifier: Apache-2.0
"""TierManager: per-layer GPU/CPU coordination."""
from __future__ import annotations

import pytest
import torch


def _build(layer_idx=0, gpu_budget=4, cpu_budget=8, **kw):
    """Build a TierManager + dependencies wired into a fake GPU paged cache."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    from vllm.v1.attention.backends.quest.cache.block_summary import (
        BlockSummaryStore,
    )
    from vllm.v1.attention.backends.quest.cache.cpu_backing_store import (
        CpuKvBackingStore,
    )
    from vllm.v1.attention.backends.quest.cache.residency import (
        BlockResidency,
    )
    from vllm.v1.attention.backends.quest.cache.tier_manager import (
        TierManager,
    )

    block_size = 4
    num_kv_heads = 1
    head_size = 8
    max_blocks = 16
    summary = BlockSummaryStore(
        num_layers=1, max_blocks=max_blocks,
        block_size=block_size, num_kv_heads=num_kv_heads,
        head_size=head_size, dtype=torch.float16, device="cuda",
    )
    cpu_store = CpuKvBackingStore(
        num_layers=1, blocks_per_layer=cpu_budget,
        block_size=block_size, num_kv_heads=num_kv_heads,
        head_size=head_size, dtype=torch.float16,
    )
    residency = BlockResidency(num_layers=1, max_blocks=max_blocks)
    # Simulated GPU paged cache slot grid for one layer.
    gpu_k = torch.zeros(
        (gpu_budget, block_size, num_kv_heads, head_size),
        dtype=torch.float16, device="cuda",
    )
    gpu_v = torch.zeros_like(gpu_k)
    return TierManager(
        layer_idx=0,
        gpu_budget=gpu_budget,
        gpu_k=gpu_k,
        gpu_v=gpu_v,
        summary_store=summary,
        residency=residency,
        cpu_store=cpu_store,
        **kw,
    )


def test_on_block_filled_updates_summary_and_residency():
    tm = _build()
    k_block = torch.randn(4, 1, 8, dtype=torch.float16, device="cuda")
    v_block = torch.randn_like(k_block)
    slot = tm.on_block_filled(seq_id=0, logical_block_id=0,
                              k_block=k_block, v_block=v_block)
    from vllm.v1.attention.backends.quest.cache.residency import (
        ResidencyState,
    )
    assert slot == 0
    assert tm.residency.state(0, 0) == ResidencyState.ON_GPU
    # GPU cache populated
    assert torch.equal(tm.gpu_k[0], k_block)


def test_eviction_when_gpu_budget_exceeded():
    tm = _build(gpu_budget=2)
    blocks = []
    for b in range(3):
        k = torch.full((4, 1, 8), float(b), dtype=torch.float16, device="cuda")
        v = torch.full((4, 1, 8), float(b + 100),
                       dtype=torch.float16, device="cuda")
        blocks.append((k, v))
        tm.on_block_filled(0, b, k, v)

    from vllm.v1.attention.backends.quest.cache.residency import (
        ResidencyState,
    )
    # block 0 is the LRU and must be on CPU now
    assert tm.residency.state(0, 0) == ResidencyState.ON_CPU
    assert tm.residency.state(0, 1) == ResidencyState.ON_GPU
    assert tm.residency.state(0, 2) == ResidencyState.ON_GPU


def test_ensure_resident_loads_from_cpu():
    tm = _build(gpu_budget=2)
    blocks = []
    for b in range(3):
        k = torch.full((4, 1, 8), float(b), dtype=torch.float16, device="cuda")
        v = torch.full((4, 1, 8), float(b + 100),
                       dtype=torch.float16, device="cuda")
        blocks.append((k, v))
        tm.on_block_filled(0, b, k, v)

    # block 0 was evicted in previous test; ensure_resident pulls it back
    ids = torch.tensor([0], dtype=torch.int32, device="cuda")
    tm.ensure_resident(seq_id=0, logical_block_ids=ids)

    from vllm.v1.attention.backends.quest.cache.residency import (
        ResidencyState,
    )
    assert tm.residency.state(0, 0) == ResidencyState.ON_GPU
    # the slot now holds block 0's data
    slot = tm.logical_to_slot(seq_id=0, logical_block_id=0)
    assert torch.all(tm.gpu_k[slot] == 0.0)


def test_stats_track_hit_and_miss():
    tm = _build(gpu_budget=2)
    for b in range(3):
        k = torch.full((4, 1, 8), float(b), dtype=torch.float16, device="cuda")
        v = torch.full((4, 1, 8), float(b + 100),
                       dtype=torch.float16, device="cuda")
        tm.on_block_filled(0, b, k, v)

    ids = torch.tensor([0, 2], dtype=torch.int32, device="cuda")
    tm.ensure_resident(seq_id=0, logical_block_ids=ids)

    s = tm.stats()
    assert s.block_filled == 3
    assert s.evict_d2h >= 1
    assert s.load_h2d >= 1
    # Note: ensure_resident itself is not the same as a select, so
    # selected_* counters won't have moved.


def test_logical_to_slot_after_load_changes():
    tm = _build(gpu_budget=2)
    for b in range(3):
        k = torch.full((4, 1, 8), float(b), dtype=torch.float16, device="cuda")
        v = torch.full((4, 1, 8), float(b + 100),
                       dtype=torch.float16, device="cuda")
        tm.on_block_filled(0, b, k, v)

    # block 0 was in slot 0 originally; after eviction and reload it might
    # land in a different slot.
    slot_after = tm.logical_to_slot(seq_id=0, logical_block_id=2)
    assert 0 <= slot_after < 2

    ids = torch.tensor([0], dtype=torch.int32, device="cuda")
    tm.ensure_resident(seq_id=0, logical_block_ids=ids)
    slot0_now = tm.logical_to_slot(seq_id=0, logical_block_id=0)
    assert 0 <= slot0_now < 2


def test_lru_capacity_broader_than_gpu_cache_blocks_per_seq():
    """When wired by Phase E, capacity = vLLM-allocated num_blocks, which
    can exceed gpu_cache_blocks_per_seq. _LRUSlotMap should still evict
    the LRU when full and not silently lose blocks."""
    from vllm.v1.attention.backends.quest.cache.tier_manager import (
        _LRUSlotMap,
    )
    m = _LRUSlotMap(capacity=12)
    for i in range(12):
        slot, evicted = m.add((0, i))
        assert evicted is None
        assert 0 <= slot < 12
    # Add 13th — should evict (0, 0).
    slot, evicted = m.add((0, 12))
    assert evicted == (0, 0)
    assert 0 <= slot < 12


def test_count_resident_matches_slot_map_membership():
    """count_resident counts how many of the given logical block ids are
    currently GPU-resident, without mutating LRU recency. With gpu_budget=2
    and 3 blocks filled in order, block 0 is the LRU and gets evicted to CPU;
    blocks 1 and 2 stay on GPU."""
    tm = _build(gpu_budget=2)
    for b in range(3):
        k = torch.full((4, 1, 8), float(b), dtype=torch.float16, device="cuda")
        v = torch.full((4, 1, 8), float(b + 100),
                       dtype=torch.float16, device="cuda")
        tm.on_block_filled(0, b, k, v)

    # Block 0 evicted; 1 and 2 resident.
    assert tm.is_resident(0, 0) is False
    assert tm.is_resident(0, 1) is True
    assert tm.is_resident(0, 2) is True
    # A different seq_id shares no slots.
    assert tm.is_resident(1, 1) is False

    # Selecting {0, 1, 2}: exactly 2 are resident (1 and 2).
    selected = [0, 1, 2]
    assert tm.count_resident(0, selected) == 2
    # Selecting only the evicted block: 0 resident.
    assert tm.count_resident(0, [0]) == 0
    # Selecting only resident blocks: matches len.
    assert tm.count_resident(0, [1, 2]) == 2
    # Accepts a tensor's .tolist() ints (the real call site shape).
    ids = torch.tensor([0, 1, 2], dtype=torch.int32, device="cuda")
    assert tm.count_resident(0, ids.tolist()) == 2


def test_count_resident_does_not_bump_lru_recency():
    """count_resident / is_resident are read-only: unlike logical_to_slot
    (which calls _slot_map.get and moves the key to most-recently-used),
    probing residency must not change which block is the LRU victim."""
    tm = _build(gpu_budget=2)
    for b in range(2):
        k = torch.full((4, 1, 8), float(b), dtype=torch.float16, device="cuda")
        v = torch.full((4, 1, 8), float(b + 100),
                       dtype=torch.float16, device="cuda")
        tm.on_block_filled(0, b, k, v)
    # LRU order is [0, 1] (0 oldest). Probing block 0 must NOT promote it.
    tm.count_resident(0, [0])
    tm.is_resident(0, 0)
    # Fill a 3rd block: the LRU victim must still be block 0.
    from vllm.v1.attention.backends.quest.cache.residency import (
        ResidencyState,
    )
    k = torch.full((4, 1, 8), 2.0, dtype=torch.float16, device="cuda")
    v = torch.full((4, 1, 8), 102.0, dtype=torch.float16, device="cuda")
    tm.on_block_filled(0, 2, k, v)
    assert tm.residency.state(0, 0) == ResidencyState.ON_CPU
    assert tm.residency.state(0, 1) == ResidencyState.ON_GPU
    assert tm.residency.state(0, 2) == ResidencyState.ON_GPU


def test_event_timing_accumulates_only_when_enabled():
    """Stage 0 Item 3: with enable_event_timing=True, a sync ensure_resident
    that actually loads from CPU records H2D-wait GPU time, and an eviction
    records D2H stall time. With timing off (default) the counters stay 0."""
    # Timing OFF (default): force an eviction + reload, counters must be 0.
    tm_off = _build(gpu_budget=2)
    for b in range(3):
        k = torch.full((4, 1, 8), float(b), dtype=torch.float16, device="cuda")
        v = torch.full((4, 1, 8), float(b + 100),
                       dtype=torch.float16, device="cuda")
        tm_off.on_block_filled(0, b, k, v)  # block 0 evicted here
    ids = torch.tensor([0], dtype=torch.int32, device="cuda")
    tm_off.ensure_resident(seq_id=0, logical_block_ids=ids)  # reload block 0
    s_off = tm_off.stats()
    assert s_off.evict_d2h >= 1 and s_off.load_h2d >= 1
    assert s_off.h2d_wait_ms == 0.0
    assert s_off.evict_stall_ms == 0.0
    assert s_off.h2d_wait_events == 0
    assert s_off.evict_stall_events == 0

    # Timing ON: same workload, counters must populate.
    tm_on = _build(gpu_budget=2, enable_event_timing=True)
    for b in range(3):
        k = torch.full((4, 1, 8), float(b), dtype=torch.float16, device="cuda")
        v = torch.full((4, 1, 8), float(b + 100),
                       dtype=torch.float16, device="cuda")
        tm_on.on_block_filled(0, b, k, v)  # block 0 evicted -> D2H stall timed
    tm_on.ensure_resident(seq_id=0, logical_block_ids=ids)  # reload -> H2D timed
    s_on = tm_on.stats()
    assert s_on.evict_d2h >= 1 and s_on.load_h2d >= 1
    # GPU event timing is non-negative; the timed intervals were recorded.
    assert s_on.evict_stall_events >= 1
    assert s_on.h2d_wait_events >= 1
    assert s_on.evict_stall_ms >= 0.0
    assert s_on.h2d_wait_ms >= 0.0


def test_event_timing_skips_h2d_when_all_resident():
    """ensure_resident on already-resident blocks does no CPU load, so it must
    NOT record an H2D-wait interval even with timing enabled."""
    tm = _build(gpu_budget=4, enable_event_timing=True)
    for b in range(3):
        k = torch.full((4, 1, 8), float(b), dtype=torch.float16, device="cuda")
        v = torch.full((4, 1, 8), float(b + 100),
                       dtype=torch.float16, device="cuda")
        tm.on_block_filled(0, b, k, v)  # all 3 fit in budget=4, none evicted
    ids = torch.tensor([0, 1, 2], dtype=torch.int32, device="cuda")
    tm.ensure_resident(seq_id=0, logical_block_ids=ids)
    s = tm.stats()
    assert s.load_h2d == 0
    assert s.h2d_wait_events == 0
    assert s.h2d_wait_ms == 0.0

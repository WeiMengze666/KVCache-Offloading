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

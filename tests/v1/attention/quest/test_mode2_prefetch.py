# SPDX-License-Identifier: Apache-2.0
"""Mode 2 cross-layer prefetch: registry, correct picks, wrong picks fallback."""
from __future__ import annotations

import pytest
import torch


def _build_tm(*, async_enabled, gpu_budget=8, cpu_budget=8):
    """Two TierManagers (one per layer) sharing the same QuestStreamPool."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    from vllm.v1.attention.backends.quest.async_transfer import (
        QuestStreamPool,
    )
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

    block_size, h_kv, d = 4, 1, 8
    max_blocks = 16
    summary = BlockSummaryStore(
        num_layers=2, max_blocks=max_blocks,
        block_size=block_size, num_kv_heads=h_kv, head_size=d,
        dtype=torch.float16, device="cuda",
    )
    cpu_store = CpuKvBackingStore(
        num_layers=2, blocks_per_layer=cpu_budget,
        block_size=block_size, num_kv_heads=h_kv, head_size=d,
        dtype=torch.float16,
    )
    residency = BlockResidency(num_layers=2, max_blocks=max_blocks)
    pool = QuestStreamPool() if async_enabled else None
    tms = []
    for layer_idx in range(2):
        gpu_k = torch.zeros(
            (gpu_budget, block_size, h_kv, d),
            dtype=torch.float16, device="cuda",
        )
        gpu_v = torch.zeros_like(gpu_k)
        tms.append(TierManager(
            layer_idx=layer_idx, gpu_budget=gpu_budget,
            gpu_k=gpu_k, gpu_v=gpu_v,
            summary_store=summary, residency=residency,
            cpu_store=cpu_store, stream_pool=pool,
        ))
    return tms, pool


def test_prefetch_top_ids_registers_event_in_pool():
    tms, pool = _build_tm(async_enabled=True)
    tm0, tm1 = tms

    # Pre-populate tm1's CPU pool with one block, so prefetch has work.
    k = torch.full((4, 1, 8), 7.0, dtype=torch.float16, device="cuda")
    v = torch.full((4, 1, 8), 11.0, dtype=torch.float16, device="cuda")
    cpu_slot = tm1.cpu_store.alloc(1)
    tm1.cpu_store.store_block(1, cpu_slot, k, v)
    tm1._cpu_slots[(0, 5)] = cpu_slot
    tm1.residency.begin_evict(1, 5)
    tm1.residency.complete_evict(1, 5)

    # Prefetch from tm0's perspective into tm1.
    top_ids = torch.tensor([5], dtype=torch.int32, device="cuda")
    tm1.prefetch_top_ids(seq_id=0, logical_block_ids=top_ids)

    event = pool.pop_prefetch_event(seq_id=0, target_layer_idx=1)
    assert event is not None
    torch.cuda.current_stream().wait_event(event)
    torch.cuda.synchronize()
    # Block 5 is now on tm1's GPU pool.
    slot = tm1.logical_to_slot(0, 5)
    assert torch.all(tm1.gpu_k[slot] == 7.0)


def test_prefetch_top_ids_no_op_when_async_disabled():
    """prefetch_top_ids must be a graceful no-op when stream_pool is None."""
    tms, pool = _build_tm(async_enabled=False)
    assert pool is None
    tm0, tm1 = tms
    top_ids = torch.tensor([5], dtype=torch.int32, device="cuda")
    # Must not raise.
    tm1.prefetch_top_ids(seq_id=0, logical_block_ids=top_ids)


def test_prefetch_idempotent_when_already_resident():
    """Prefetching a block that's already on GPU is a no-op (LRU touch)."""
    tms, pool = _build_tm(async_enabled=True)
    tm0, tm1 = tms

    # Block 3 is on GPU.
    k = torch.full((4, 1, 8), 1.0, dtype=torch.float16, device="cuda")
    v = torch.full((4, 1, 8), 2.0, dtype=torch.float16, device="cuda")
    tm1.on_block_filled(0, 3, k, v)

    top_ids = torch.tensor([3], dtype=torch.int32, device="cuda")
    tm1.prefetch_top_ids(seq_id=0, logical_block_ids=top_ids)
    event = pool.pop_prefetch_event(seq_id=0, target_layer_idx=1)
    if event is not None:
        torch.cuda.current_stream().wait_event(event)
    torch.cuda.synchronize()
    # Still resident with correct data.
    slot = tm1.logical_to_slot(0, 3)
    assert torch.all(tm1.gpu_k[slot] == 1.0)


def test_wrong_prefetch_falls_back_to_ensure_resident():
    """If layer N predicts block 5 but layer N+1 actually selects block 6,
    ensure_resident on [6] must still work and produce correct data."""
    tms, pool = _build_tm(async_enabled=True)
    tm0, tm1 = tms

    # Pre-populate tm1's CPU pool with two blocks (5 and 6).
    for bid, val in [(5, 7.0), (6, 9.0)]:
        k = torch.full((4, 1, 8), val, dtype=torch.float16, device="cuda")
        v = torch.full((4, 1, 8), val + 100, dtype=torch.float16, device="cuda")
        cpu_slot = tm1.cpu_store.alloc(1)
        tm1.cpu_store.store_block(1, cpu_slot, k, v)
        tm1._cpu_slots[(0, bid)] = cpu_slot
        tm1.residency.begin_evict(1, bid)
        tm1.residency.complete_evict(1, bid)

    # layer N predicts wrong: prefetch [5].
    tm1.prefetch_top_ids(
        seq_id=0,
        logical_block_ids=torch.tensor([5], dtype=torch.int32, device="cuda"),
    )

    # layer N+1 actually wants [6]. Wait on prev event, then ensure_resident.
    event = pool.pop_prefetch_event(seq_id=0, target_layer_idx=1)
    torch.cuda.current_stream().wait_event(event)
    h2d_event = tm1.ensure_resident(
        seq_id=0,
        logical_block_ids=torch.tensor([6], dtype=torch.int32, device="cuda"),
    )
    torch.cuda.current_stream().wait_event(h2d_event)
    torch.cuda.synchronize()

    slot = tm1.logical_to_slot(0, 6)
    assert torch.all(tm1.gpu_k[slot] == 9.0)

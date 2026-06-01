# SPDX-License-Identifier: Apache-2.0
"""Mode 1 async TierManager: ensure_resident + _spill_to_cpu correctness."""
from __future__ import annotations

import pytest
import torch


def _build(*, async_enabled: bool, gpu_budget=4, cpu_budget=8):
    """Build a TierManager wired with or without a QuestStreamPool."""
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
        num_layers=1, max_blocks=max_blocks,
        block_size=block_size, num_kv_heads=h_kv, head_size=d,
        dtype=torch.float16, device="cuda",
    )
    cpu_store = CpuKvBackingStore(
        num_layers=1, blocks_per_layer=cpu_budget,
        block_size=block_size, num_kv_heads=h_kv, head_size=d,
        dtype=torch.float16,
    )
    residency = BlockResidency(num_layers=1, max_blocks=max_blocks)
    gpu_k = torch.zeros(
        (gpu_budget, block_size, h_kv, d),
        dtype=torch.float16, device="cuda",
    )
    gpu_v = torch.zeros_like(gpu_k)
    pool = QuestStreamPool() if async_enabled else None
    return TierManager(
        layer_idx=0, gpu_budget=gpu_budget,
        gpu_k=gpu_k, gpu_v=gpu_v,
        summary_store=summary, residency=residency,
        cpu_store=cpu_store, stream_pool=pool,
    )


def test_ensure_resident_sync_returns_none():
    """Phase B contract: when stream_pool is None, ensure_resident returns
    None — caller doesn't wait on anything."""
    tm = _build(async_enabled=False, gpu_budget=2)
    for b in range(3):
        k = torch.full((4, 1, 8), float(b), dtype=torch.float16, device="cuda")
        v = torch.full((4, 1, 8), float(b + 100), dtype=torch.float16, device="cuda")
        tm.on_block_filled(0, b, k, v)
    ids = torch.tensor([0], dtype=torch.int32, device="cuda")
    result = tm.ensure_resident(seq_id=0, logical_block_ids=ids)
    assert result is None


def test_ensure_resident_async_returns_event():
    """Mode 1 contract: when stream_pool is set, ensure_resident returns
    an Event the caller can wait on."""
    tm = _build(async_enabled=True, gpu_budget=2)
    for b in range(3):
        k = torch.full((4, 1, 8), float(b), dtype=torch.float16, device="cuda")
        v = torch.full((4, 1, 8), float(b + 100), dtype=torch.float16, device="cuda")
        tm.on_block_filled(0, b, k, v)
    ids = torch.tensor([0], dtype=torch.int32, device="cuda")
    event = tm.ensure_resident(seq_id=0, logical_block_ids=ids)
    assert isinstance(event, torch.cuda.Event)


def test_ensure_resident_async_data_correct():
    """After waiting on the event, data must match what sync path would produce."""
    # Sync reference.
    tm_sync = _build(async_enabled=False, gpu_budget=2)
    for b in range(3):
        k = torch.full((4, 1, 8), float(b), dtype=torch.float16, device="cuda")
        v = torch.full((4, 1, 8), float(b + 100), dtype=torch.float16, device="cuda")
        tm_sync.on_block_filled(0, b, k, v)
    ids = torch.tensor([0], dtype=torch.int32, device="cuda")
    tm_sync.ensure_resident(seq_id=0, logical_block_ids=ids)
    sync_slot = tm_sync.logical_to_slot(0, 0)
    sync_k = tm_sync.gpu_k[sync_slot].clone()
    sync_v = tm_sync.gpu_v[sync_slot].clone()

    # Async path.
    tm_async = _build(async_enabled=True, gpu_budget=2)
    for b in range(3):
        k = torch.full((4, 1, 8), float(b), dtype=torch.float16, device="cuda")
        v = torch.full((4, 1, 8), float(b + 100), dtype=torch.float16, device="cuda")
        tm_async.on_block_filled(0, b, k, v)
    event = tm_async.ensure_resident(seq_id=0, logical_block_ids=ids)
    torch.cuda.current_stream().wait_event(event)
    torch.cuda.synchronize()
    async_slot = tm_async.logical_to_slot(0, 0)

    # Same logical->slot resolution + same data.
    assert torch.equal(tm_async.gpu_k[async_slot], sync_k)
    assert torch.equal(tm_async.gpu_v[async_slot], sync_v)


def test_spill_to_cpu_async_uses_record_stream():
    """When async is enabled, _spill_to_cpu must call record_stream on the
    GPU source tensor before the non_blocking D2H. We verify by spying on
    record_stream calls."""
    tm = _build(async_enabled=True, gpu_budget=2)
    record_calls = []
    orig = torch.Tensor.record_stream

    def _spy(self, stream):
        record_calls.append(stream)
        return orig(self, stream)

    # Patch globally for this test.
    torch.Tensor.record_stream = _spy
    try:
        for b in range(3):  # third block triggers eviction
            k = torch.full((4, 1, 8), float(b), dtype=torch.float16, device="cuda")
            v = torch.full((4, 1, 8), float(b + 100), dtype=torch.float16, device="cuda")
            tm.on_block_filled(0, b, k, v)
        torch.cuda.synchronize()
        # At least one record_stream call into the d2h_stream.
        assert any(s is tm.stream_pool.d2h_stream for s in record_calls)
    finally:
        torch.Tensor.record_stream = orig


def test_no_eviction_no_spill_call():
    """Sanity: when GPU pool has slack, no spill happens (no D2H, no record_stream
    calls into d2h_stream)."""
    tm = _build(async_enabled=True, gpu_budget=8)
    for b in range(3):
        k = torch.full((4, 1, 8), float(b), dtype=torch.float16, device="cuda")
        v = torch.full((4, 1, 8), float(b + 100), dtype=torch.float16, device="cuda")
        tm.on_block_filled(0, b, k, v)
    assert tm.stats().evict_d2h == 0


def test_async_ensure_resident_matches_sync_on_arena():
    """Mode 1 async H2D into the arena, waited via the returned event, must land
    the SAME bytes as the synchronous reload. Both paths trim the SAME engine KV
    into a cap-block arena, spill the overflow, then reload a spilled block; the
    reloaded arena slot must be bit-identical under sync and async."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    from vllm.v1.attention.backends.quest.async_transfer import QuestStreamPool
    from vllm.v1.attention.backends.quest.cache.block_summary import BlockSummaryStore
    from vllm.v1.attention.backends.quest.cache.cpu_backing_store import CpuKvBackingStore
    from vllm.v1.attention.backends.quest.cache.residency import BlockResidency
    from vllm.v1.attention.backends.quest.cache.tier_manager import TierManager

    cap, bs, h, d, P, nb = 4, 256, 2, 64, 10, 16
    engine = torch.randn(nb, 2, bs, h, d, dtype=torch.float16, device="cuda")
    block_table_row = list(range(P))

    def build(async_enabled):
        summary = BlockSummaryStore(num_layers=1, max_blocks=nb, block_size=bs,
                                    num_kv_heads=h, head_size=d,
                                    dtype=torch.float16, device="cuda")
        cpu = CpuKvBackingStore(num_layers=1, blocks_per_layer=nb, block_size=bs,
                                num_kv_heads=h, head_size=d, dtype=torch.float16)
        res = BlockResidency(num_layers=1, max_blocks=nb)
        pool = QuestStreamPool() if async_enabled else None
        return TierManager(
            layer_idx=0, gpu_budget=cap,
            gpu_k=torch.empty(cap, bs, h, d, dtype=torch.float16, device="cuda"),
            gpu_v=torch.empty(cap, bs, h, d, dtype=torch.float16, device="cuda"),
            summary_store=summary, residency=res, cpu_store=cpu,
            stream_pool=pool, engine_kv_cache=engine,
            gpu_pool_aliases_kv_cache=False,
        )

    def reload_block_3(async_enabled):
        tm = build(async_enabled)
        tm.trim_to_working_set(seq_id=0, num_full_blocks=P, kv_cache=engine,
                               block_table_row=block_table_row)
        # block 3 was spilled (trim keeps last cap-1=3 -> {7,8,9}).
        ev = tm.ensure_resident(seq_id=0,
                                logical_block_ids=torch.tensor([3], device="cuda"))
        if ev is not None:
            torch.cuda.current_stream().wait_event(ev)
        torch.cuda.synchronize()
        slot = tm.logical_to_slot(0, 3)
        return tm.gpu_k[slot].clone(), tm.gpu_v[slot].clone(), tm.stats()

    k_sync, v_sync, st_sync = reload_block_3(async_enabled=False)
    k_async, v_async, st_async = reload_block_3(async_enabled=True)
    # The reloaded KV must equal the original engine block under both paths.
    assert torch.equal(k_sync, engine[3, 0])
    assert torch.equal(k_async, engine[3, 0])
    assert torch.equal(v_sync, engine[3, 1])
    assert torch.equal(v_async, engine[3, 1])
    assert st_sync.load_h2d == 1 and st_async.load_h2d == 1

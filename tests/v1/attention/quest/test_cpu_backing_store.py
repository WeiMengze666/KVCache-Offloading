# SPDX-License-Identifier: Apache-2.0
"""CpuKvBackingStore: pinned pool, free-list, sync transfers."""
from __future__ import annotations

import pytest
import torch


@pytest.fixture
def store():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    from vllm.v1.attention.backends.quest.cache.cpu_backing_store import (
        CpuKvBackingStore,
    )
    return CpuKvBackingStore(
        num_layers=4,
        blocks_per_layer=8,
        block_size=16,
        num_kv_heads=2,
        head_size=64,
        dtype=torch.float16,
    )


def test_pinned_memory(store):
    assert store.k.is_pinned()
    assert store.v.is_pinned()
    assert store.k.shape == (4, 8, 16, 2, 64)
    assert store.v.shape == (4, 8, 16, 2, 64)


def test_alloc_returns_unique_slot(store):
    a = store.alloc(layer_idx=2)
    b = store.alloc(layer_idx=2)
    assert a != b
    assert 0 <= a < 8
    assert 0 <= b < 8


def test_alloc_free_round_trip(store):
    a = store.alloc(0)
    store.free(0, a)
    b = store.alloc(0)
    assert b == a   # reused immediately


def test_alloc_full_layer_raises(store):
    used = [store.alloc(0) for _ in range(8)]
    with pytest.raises(RuntimeError, match="layer 0 CPU pool is full"):
        store.alloc(0)
    for s in used:
        store.free(0, s)


def test_store_then_load_preserves_data(store):
    k_block = torch.randn(16, 2, 64, dtype=torch.float16, device="cuda")
    v_block = torch.randn(16, 2, 64, dtype=torch.float16, device="cuda")

    cpu_slot = store.alloc(layer_idx=1)
    store.store_block(layer_idx=1, cpu_slot=cpu_slot,
                       k_block=k_block, v_block=v_block)

    k_dst = torch.empty_like(k_block)
    v_dst = torch.empty_like(v_block)
    store.load_block(layer_idx=1, cpu_slot=cpu_slot,
                      k_dst=k_dst, v_dst=v_dst)

    assert torch.equal(k_dst, k_block)
    assert torch.equal(v_dst, v_block)


def test_layer_isolation(store):
    """Slots are layer-local — slot 0 in layer 0 != slot 0 in layer 3."""
    k = torch.full((16, 2, 64), 1.0, dtype=torch.float16, device="cuda")
    v = torch.full((16, 2, 64), 2.0, dtype=torch.float16, device="cuda")
    store.store_block(0, 0, k, v)

    out_k = torch.empty_like(k)
    out_v = torch.empty_like(v)
    store.load_block(layer_idx=3, cpu_slot=0, k_dst=out_k, v_dst=out_v)
    # layer 3 slot 0 must NOT be 1.0 / 2.0 — it was never written
    assert not torch.all(out_k == 1.0)


def test_stats_alloc_free_balance(store):
    for layer in range(4):
        slots = [store.alloc(layer) for _ in range(3)]
        for s in slots:
            store.free(layer, s)
    s = store.stats()
    assert s.alloc_count == s.free_count == 12


def test_store_block_accepts_non_blocking_kwarg(store):
    """non_blocking=True must work without raising; data must transfer
    correctly when synchronized."""
    k = torch.full((16, 2, 64), 7.0, dtype=torch.float16, device="cuda")
    v = torch.full((16, 2, 64), 9.0, dtype=torch.float16, device="cuda")
    cpu_slot = store.alloc(1)
    store.store_block(1, cpu_slot, k, v, non_blocking=True)
    torch.cuda.synchronize()
    assert torch.all(store.k[1, cpu_slot] == 7.0)
    assert torch.all(store.v[1, cpu_slot] == 9.0)


def test_load_block_accepts_non_blocking_kwarg(store):
    """non_blocking=True for H2D must work + roundtrip correctly."""
    k_src = torch.full((16, 2, 64), 3.0, dtype=torch.float16, device="cuda")
    v_src = torch.full((16, 2, 64), 5.0, dtype=torch.float16, device="cuda")
    cpu_slot = store.alloc(2)
    store.store_block(2, cpu_slot, k_src, v_src)

    k_dst = torch.empty_like(k_src)
    v_dst = torch.empty_like(v_src)
    store.load_block(2, cpu_slot, k_dst, v_dst, non_blocking=True)
    torch.cuda.synchronize()
    assert torch.equal(k_dst, k_src)
    assert torch.equal(v_dst, v_src)

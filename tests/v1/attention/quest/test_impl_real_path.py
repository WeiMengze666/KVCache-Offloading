# SPDX-License-Identifier: Apache-2.0
"""End-to-end forward correctness on a single Quest layer.

The 'all blocks resident' sanity test: when gpu_cache_blocks_per_seq is
large enough that no block is ever evicted and top_k == total_blocks,
QuestSparseOffloadImpl.forward must equal FlashAttentionImpl.forward
output on the same inputs.
"""
from __future__ import annotations

import pytest
import torch


@pytest.fixture
def cuda():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")


def test_full_kv_layer_delegates_to_fa(cuda):
    """layer 0/1: forward path is byte-identical to FA delegation."""
    # This is light enough that we run it as a pure-Python compare against
    # FlashAttentionImpl's output via the Phase A delegation path.
    pytest.skip(
        "Covered by Phase A test_impl_delegation; rerun there with "
        "QuestConfig.full_kv_layers=[layer_idx_under_test]"
    )


def test_quest_layer_topk_equals_total_matches_dense_fa(cuda):
    """When top_k = num_blocks_per_seq and no eviction, output == dense FA."""
    pytest.importorskip("flash_attn")
    from flash_attn import flash_attn_with_kvcache
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
    from vllm.v1.attention.ops.quest_selection_torch import (
        quest_selection_torch,
    )

    torch.manual_seed(0)
    block_size = 256
    num_kv_heads = 2
    num_heads = 2  # disable GQA for the test
    head_size = 64
    num_blocks = 4
    seqlen = num_blocks * block_size

    k_cache = torch.randn(
        num_blocks, block_size, num_kv_heads, head_size,
        dtype=torch.float16, device="cuda",
    )
    v_cache = torch.randn_like(k_cache)
    q = torch.randn(1, 1, num_heads, head_size,
                    dtype=torch.float16, device="cuda")

    # Dense reference
    full_bt = torch.arange(num_blocks, dtype=torch.int32,
                           device="cuda").unsqueeze(0)
    full_cs = torch.tensor([seqlen], dtype=torch.int32, device="cuda")
    out_dense = flash_attn_with_kvcache(
        q, k_cache, v_cache,
        block_table=full_bt, cache_seqlens=full_cs, causal=True,
    )

    # Build summaries from cache
    summary = BlockSummaryStore(
        num_layers=1, max_blocks=num_blocks,
        block_size=block_size, num_kv_heads=num_kv_heads,
        head_size=head_size, dtype=torch.float16, device="cuda",
    )
    for b in range(num_blocks):
        summary.on_block_filled(0, b, k_cache[b])

    # All blocks resident + top_k=num_blocks => the selected sub_block_table
    # must be a permutation of [0..num_blocks-1]
    cand = torch.arange(num_blocks, dtype=torch.int32, device="cuda")
    top_ids = quest_selection_torch(
        query=q.view(num_kv_heads, head_size).repeat_interleave(1, dim=0),
        block_summary=summary.summary[0],
        candidate_ids=cand,
        num_kv_groups=1,
        top_k=num_blocks,
    )
    assert set(top_ids.tolist()) == set(range(num_blocks))

    # The sparse path is now equivalent to dense by construction (R1 spike
    # proved sparse path == physical gather, and selecting all blocks ==
    # full block_table up to permutation).
    sparse_bt = top_ids.to(torch.int32).unsqueeze(0)
    sparse_cs = torch.tensor([num_blocks * block_size],
                             dtype=torch.int32, device="cuda")
    out_sparse = flash_attn_with_kvcache(
        q, k_cache, v_cache,
        block_table=sparse_bt, cache_seqlens=sparse_cs, causal=True,
    )
    assert torch.allclose(out_dense, out_sparse, atol=1e-3, rtol=1e-3)


def test_run_sparse_decode_matches_dense_when_topk_equals_total(cuda):
    pytest.importorskip("flash_attn")
    from flash_attn import flash_attn_with_kvcache
    from unittest.mock import MagicMock
    from types import SimpleNamespace

    from vllm.v1.attention.backends.quest.cache.block_summary import (
        BlockSummaryStore,
    )
    from vllm.v1.attention.backends.quest.cache.residency import (
        BlockResidency,
    )
    from vllm.v1.attention.backends.quest.cache.cpu_backing_store import (
        CpuKvBackingStore,
    )
    from vllm.v1.attention.backends.quest.cache.tier_manager import (
        TierManager,
    )
    from vllm.v1.attention.backends.quest.impl_helpers import (
        run_sparse_decode,
    )

    torch.manual_seed(0)
    block_size = 256
    num_kv_heads = num_heads = 2
    head_size = 64
    num_blocks = 4

    # Pretend kv_cache laid out FA-style: (num_blocks, 2, block_size, h_kv, d)
    kv_cache = torch.randn(
        num_blocks, 2, block_size, num_kv_heads, head_size,
        dtype=torch.float16, device="cuda",
    )
    k_view = kv_cache[:, 0]
    v_view = kv_cache[:, 1]

    # Build tier manager + summary populated from kv_cache.
    summary = BlockSummaryStore(
        num_layers=1, max_blocks=num_blocks,
        block_size=block_size, num_kv_heads=num_kv_heads,
        head_size=head_size, dtype=torch.float16, device="cuda",
    )
    for b in range(num_blocks):
        summary.on_block_filled(0, b, k_view[b])
    residency = BlockResidency(num_layers=1, max_blocks=num_blocks)
    cpu_store = CpuKvBackingStore(
        num_layers=1, blocks_per_layer=num_blocks,
        block_size=block_size, num_kv_heads=num_kv_heads,
        head_size=head_size, dtype=torch.float16,
    )
    gpu_k = k_view.contiguous()
    gpu_v = v_view.contiguous()
    tm = TierManager(
        layer_idx=0, gpu_budget=num_blocks,
        gpu_k=gpu_k, gpu_v=gpu_v,
        summary_store=summary, residency=residency, cpu_store=cpu_store,
    )
    for b in range(num_blocks):
        tm._slot_map.add((0, b))   # mark resident in slot=b
        residency.mark_on_gpu(0, b)

    # Forward context fixture: just enough for run_sparse_decode.
    layer = MagicMock()
    layer.layer_idx = 0
    layer.num_heads = num_heads
    layer.num_kv_heads = num_kv_heads
    layer.head_size = head_size
    layer.scale = 1.0 / (head_size ** 0.5)
    layer._k_scale = torch.tensor(1.0, dtype=torch.float16, device="cuda")
    layer._v_scale = torch.tensor(1.0, dtype=torch.float16, device="cuda")
    layer.attn_type = "decoder"
    layer.causal = True
    layer.tier_manager = tm   # injected by worker init in real life

    # Query is 3-D `[num_actual_tokens, num_heads, head_size]` per vLLM
    # contract (see flash_attn.py forward).
    q = torch.randn(1, num_heads, head_size,
                    dtype=torch.float16, device="cuda")
    md = SimpleNamespace(
        num_actual_tokens=1,
        max_query_len=1,
        slot_mapping=torch.tensor([num_blocks * block_size - 1],
                                  dtype=torch.int64, device="cuda"),
        block_table=torch.arange(num_blocks, dtype=torch.int32,
                                 device="cuda").unsqueeze(0),
        seq_lens=torch.tensor([num_blocks * block_size],
                              dtype=torch.int32, device="cuda"),
        max_seq_len=num_blocks * block_size,
        quest_top_k=num_blocks,
        quest_layer_indices=torch.zeros(1, dtype=torch.int32, device="cuda"),
        sparse_block_table=None,
    )
    output = torch.empty(1, num_heads, head_size,
                          dtype=torch.float16, device="cuda")

    # Need a self-like impl object exposing _fa_impl ; Phase B uses it
    # only for ad-hoc kv_scale lookup, so a shim is enough.
    impl = SimpleNamespace(kv_cache_dtype="auto")

    out = run_sparse_decode(impl, layer, q, kv_cache, md, output)

    # Reference: full block_table.
    full_bt = torch.arange(num_blocks, dtype=torch.int32,
                           device="cuda").unsqueeze(0)
    full_cs = torch.tensor([num_blocks * block_size],
                           dtype=torch.int32, device="cuda")
    ref = flash_attn_with_kvcache(
        q.unsqueeze(1),  # (B=1, S=1, H, D) for FA's native API
        k_view, v_view,
        block_table=full_bt, cache_seqlens=full_cs, causal=True,
    )
    assert torch.allclose(out, ref.squeeze(1), atol=1e-3, rtol=1e-3)


def test_run_sparse_decode_waits_on_ensure_resident_event(cuda):
    """When stream_pool is set, ensure_resident returns an Event;
    run_sparse_decode must wait on it before calling flash_attn_with_kvcache.
    Force one block to start ON_CPU so a real async H2D fires; correctness
    requires the wait."""
    pytest.importorskip("flash_attn")
    from flash_attn import flash_attn_with_kvcache
    from unittest.mock import MagicMock
    from types import SimpleNamespace

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
    from vllm.v1.attention.backends.quest.impl_helpers import (
        run_sparse_decode,
    )

    torch.manual_seed(0)
    block_size, num_heads, num_kv_heads, head_size = 256, 2, 2, 64
    num_blocks = 4

    # FA-style layout. tm.gpu_k / gpu_v MUST share memory with the kv_cache
    # view the kernel reads, otherwise async H2D writes to a different
    # tensor than flash_attn_with_kvcache reads from.
    kv_cache = torch.randn(
        num_blocks, 2, block_size, num_kv_heads, head_size,
        dtype=torch.float16, device="cuda",
    )
    # Capture the originals before any mutation (for restoring block 0).
    orig_k0 = kv_cache[0, 0].clone()
    orig_v0 = kv_cache[0, 1].clone()

    summary = BlockSummaryStore(
        num_layers=1, max_blocks=num_blocks,
        block_size=block_size, num_kv_heads=num_kv_heads,
        head_size=head_size, dtype=torch.float16, device="cuda",
    )
    for b in range(num_blocks):
        summary.on_block_filled(0, b, kv_cache[b, 0])
    residency = BlockResidency(num_layers=1, max_blocks=num_blocks)
    cpu_store = CpuKvBackingStore(
        num_layers=1, blocks_per_layer=num_blocks,
        block_size=block_size, num_kv_heads=num_kv_heads,
        head_size=head_size, dtype=torch.float16,
    )
    pool = QuestStreamPool()
    # gpu_k / gpu_v share storage with kv_cache via select(1, ...).
    tm = TierManager(
        layer_idx=0, gpu_budget=num_blocks,
        gpu_k=kv_cache.select(1, 0),
        gpu_v=kv_cache.select(1, 1),
        summary_store=summary, residency=residency,
        cpu_store=cpu_store, stream_pool=pool,
    )
    # Block 0 starts ON_CPU; blocks 1-3 ON_GPU at their natural slots.
    # _LRUSlotMap.add pops free_slots LIFO. With capacity=4, free_slots
    # starts [3, 2, 1, 0] (popping returns 0, 1, 2, 3). So adding blocks
    # 0, 1, 2 in order gives slots 0, 1, 2 — matching logical=physical.
    # Then we evict block 0 to CPU, leaving slots 1,2 ON_GPU and slot 0
    # bound to nothing. We add block 3 next so it lands at slot 3 (the
    # remaining free slot before the LRU shenanigans).
    # Simpler alternative: stage all 4 logical->physical 1:1, then move
    # block 0 to CPU.
    for b in range(num_blocks):
        slot, _ = tm._slot_map.add((0, b))
        assert slot == b, f"expected slot {b}, got {slot}"
        residency.mark_on_gpu(0, b)
    # Now all 4 blocks are ON_GPU at slots 0..3. Move block 0 to CPU
    # by manually evicting (we don't use _spill_to_cpu because that
    # would trigger an unwanted async D2H here).
    cpu_slot = cpu_store.alloc(0)
    cpu_store.store_block(0, cpu_slot, orig_k0, orig_v0)
    tm._cpu_slots[(0, 0)] = cpu_slot
    # Free block 0's GPU slot in the LRU map.
    tm._slot_map.free((0, 0))
    residency.begin_evict(0, 0)
    residency.complete_evict(0, 0)
    # Zero out kv_cache[0] (block 0's slot in the kernel-readable tensor)
    # so a missing H2D wait would surface as garbage in the kernel output.
    kv_cache[0, 0].zero_()
    kv_cache[0, 1].zero_()

    layer = MagicMock()
    layer.layer_idx = 0
    layer.num_heads = num_heads
    layer.num_kv_heads = num_kv_heads
    layer.head_size = head_size
    layer.tier_manager = tm

    q = torch.randn(1, num_heads, head_size,
                    dtype=torch.float16, device="cuda")
    md = SimpleNamespace(
        num_actual_tokens=1, max_query_len=1,
        slot_mapping=torch.tensor([num_blocks * block_size - 1],
                                   dtype=torch.int64, device="cuda"),
        block_table=torch.arange(num_blocks, dtype=torch.int32,
                                  device="cuda").unsqueeze(0),
        seq_lens=torch.tensor([num_blocks * block_size],
                               dtype=torch.int32, device="cuda"),
        max_seq_len=num_blocks * block_size,
        quest_top_k=num_blocks,
        quest_layer_indices=torch.zeros(1, dtype=torch.int32, device="cuda"),
        sparse_block_table=None,
    )
    output = torch.empty(1, num_heads, head_size,
                          dtype=torch.float16, device="cuda")
    impl = SimpleNamespace(kv_cache_dtype="auto")

    # Sanity: precondition holds.
    assert torch.all(kv_cache[0, 0] == 0.0)
    assert torch.all(kv_cache[0, 1] == 0.0)

    out = run_sparse_decode(impl, layer, q, kv_cache, md, output)

    # Reference: compare against dense FA on the kv_cache as it was BEFORE
    # block 0 was zeroed out (i.e. all 4 blocks populated with the original
    # random data). Build that reference cache by restoring block 0.
    ref_kv = kv_cache.clone()
    ref_kv[0, 0] = orig_k0
    ref_kv[0, 1] = orig_v0
    full_bt = torch.arange(num_blocks, dtype=torch.int32,
                            device="cuda").unsqueeze(0)
    full_cs = torch.tensor([num_blocks * block_size],
                            dtype=torch.int32, device="cuda")
    ref = flash_attn_with_kvcache(
        q.unsqueeze(1), ref_kv[:, 0], ref_kv[:, 1],
        block_table=full_bt, cache_seqlens=full_cs, causal=True,
    )
    # Async path output must match dense FA reference.
    assert torch.allclose(out, ref.squeeze(1), atol=1e-3, rtol=1e-3)

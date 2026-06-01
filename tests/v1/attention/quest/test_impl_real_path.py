# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
        num_blocks,
        block_size,
        num_kv_heads,
        head_size,
        dtype=torch.float16,
        device="cuda",
    )
    v_cache = torch.randn_like(k_cache)
    q = torch.randn(1, 1, num_heads, head_size, dtype=torch.float16, device="cuda")

    # Dense reference
    full_bt = torch.arange(num_blocks, dtype=torch.int32, device="cuda").unsqueeze(0)
    full_cs = torch.tensor([seqlen], dtype=torch.int32, device="cuda")
    out_dense = flash_attn_with_kvcache(
        q,
        k_cache,
        v_cache,
        block_table=full_bt,
        cache_seqlens=full_cs,
        causal=True,
    )

    # Build summaries from cache
    summary = BlockSummaryStore(
        num_layers=1,
        max_blocks=num_blocks,
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=torch.float16,
        device="cuda",
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
    sparse_cs = torch.tensor(
        [num_blocks * block_size], dtype=torch.int32, device="cuda"
    )
    out_sparse = flash_attn_with_kvcache(
        q,
        k_cache,
        v_cache,
        block_table=sparse_bt,
        cache_seqlens=sparse_cs,
        causal=True,
    )
    assert torch.allclose(out_dense, out_sparse, atol=1e-3, rtol=1e-3)


def _build_real_path_state():
    """Reusable fixture for run_sparse_decode tests. Returns the
    6-tuple (impl, layer, query, kv_cache, md, output) where:
      - kv_cache is FA-laid-out (num_blocks, 2, block_size, h_kv, head_size)
      - layer.tier_manager is fully populated and all blocks marked resident
      - md.quest_top_k == num_blocks (no eviction; sparse path == dense path)
      - layer is a MagicMock with the attributes run_sparse_decode reads.
    Caller may overwrite `layer._quest_selection_callable_ref` to swap impls.
    """
    from types import SimpleNamespace
    from unittest.mock import MagicMock

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

    torch.manual_seed(0)
    block_size = 256
    num_kv_heads = num_heads = 2
    head_size = 64
    num_blocks = 4

    kv_cache = torch.randn(
        num_blocks,
        2,
        block_size,
        num_kv_heads,
        head_size,
        dtype=torch.float16,
        device="cuda",
    )
    k_view = kv_cache[:, 0]
    v_view = kv_cache[:, 1]

    summary = BlockSummaryStore(
        num_layers=1,
        max_blocks=num_blocks,
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=torch.float16,
        device="cuda",
    )
    for b in range(num_blocks):
        summary.on_block_filled(0, b, k_view[b])
    residency = BlockResidency(num_layers=1, max_blocks=num_blocks)
    cpu_store = CpuKvBackingStore(
        num_layers=1,
        blocks_per_layer=num_blocks,
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=torch.float16,
    )
    gpu_k = k_view.contiguous()
    gpu_v = v_view.contiguous()
    tm = TierManager(
        layer_idx=0,
        gpu_budget=num_blocks,
        gpu_k=gpu_k,
        gpu_v=gpu_v,
        summary_store=summary,
        residency=residency,
        cpu_store=cpu_store,
    )
    for b in range(num_blocks):
        tm._slot_map.add((0, b))
        residency.mark_on_gpu(0, b)

    layer = MagicMock()
    layer.layer_idx = 0
    layer.num_heads = num_heads
    layer.num_kv_heads = num_kv_heads
    layer.head_size = head_size
    layer.scale = 1.0 / (head_size**0.5)
    layer._k_scale = torch.tensor(1.0, dtype=torch.float16, device="cuda")
    layer._v_scale = torch.tensor(1.0, dtype=torch.float16, device="cuda")
    layer.attn_type = "decoder"
    layer.causal = True
    layer.tier_manager = tm
    # Default to None so run_sparse_decode's getattr-fallback to torch
    # oracle works. MagicMock would otherwise auto-synthesize a Mock for
    # this attribute and break dispatch.
    layer._quest_selection_callable_ref = None

    q = torch.randn(1, num_heads, head_size, dtype=torch.float16, device="cuda")
    md = SimpleNamespace(
        num_actual_tokens=1,
        max_query_len=1,
        slot_mapping=torch.tensor(
            [num_blocks * block_size - 1],
            dtype=torch.int64,
            device="cuda",
        ),
        block_table=torch.arange(
            num_blocks,
            dtype=torch.int32,
            device="cuda",
        ).unsqueeze(0),
        seq_lens=torch.tensor(
            [num_blocks * block_size],
            dtype=torch.int32,
            device="cuda",
        ),
        max_seq_len=num_blocks * block_size,
        quest_top_k=num_blocks,
        quest_layer_indices=torch.zeros(
            1,
            dtype=torch.int32,
            device="cuda",
        ),
        sparse_block_table=None,
    )
    output = torch.empty(
        1,
        num_heads,
        head_size,
        dtype=torch.float16,
        device="cuda",
    )
    impl = SimpleNamespace(kv_cache_dtype="auto")
    return impl, layer, q, kv_cache, md, output


def test_run_sparse_decode_matches_dense_when_topk_equals_total(cuda):
    pytest.importorskip("flash_attn")
    from flash_attn import flash_attn_with_kvcache

    from vllm.v1.attention.backends.quest.impl_helpers import (
        run_sparse_decode,
    )

    impl, layer, q, kv_cache, md, output = _build_real_path_state()
    num_blocks = md.quest_top_k
    block_size = layer.tier_manager.gpu_k.shape[1]

    out = run_sparse_decode(impl, layer, q, kv_cache, md, output)

    full_bt = torch.arange(num_blocks, dtype=torch.int32, device="cuda").unsqueeze(0)
    full_cs = torch.tensor([num_blocks * block_size], dtype=torch.int32, device="cuda")
    ref = flash_attn_with_kvcache(
        q.unsqueeze(1),
        kv_cache[:, 0],
        kv_cache[:, 1],
        block_table=full_bt,
        cache_seqlens=full_cs,
        causal=True,
    )
    assert torch.allclose(out, ref.squeeze(1), atol=1e-3, rtol=1e-3)


def test_run_sparse_decode_counts_selected_on_gpu(cuda):
    """Item 2 acceptance: selected_on_gpu counts how many of the selected
    blocks were ALREADY GPU-resident at selection time (before
    ensure_resident pulls the rest back). Hand-computed expectation: with
    4 candidate blocks, top_k=4 selecting all of them, and exactly 1 block
    pre-evicted to CPU, selected_total == 4 and selected_on_gpu == 3.
    """
    pytest.importorskip("flash_attn")

    from vllm.v1.attention.backends.quest.cache.residency import (
        ResidencyState,
    )
    from vllm.v1.attention.backends.quest.impl_helpers import (
        run_sparse_decode,
    )

    impl, layer, q, kv_cache, md, output = _build_real_path_state()
    tm = layer.tier_manager
    num_blocks = md.quest_top_k  # == 4, top_k selects every candidate

    # Pre-evict block 0 to CPU so it is NOT resident when selection runs.
    # Mirror the eviction dance used elsewhere (no _spill_to_cpu, to avoid
    # an unwanted async D2H): copy into CPU pool, drop the GPU slot, flip
    # residency. ensure_resident inside run_sparse_decode will pull it back.
    cpu_slot = tm.cpu_store.alloc(0)
    tm.cpu_store.store_block(0, cpu_slot, tm.gpu_k[0], tm.gpu_v[0])
    tm._cpu_slots[(0, 0)] = cpu_slot
    tm._slot_map.free((0, 0))
    tm.residency.begin_evict(0, 0)
    tm.residency.complete_evict(0, 0)

    assert tm.is_resident(0, 0) is False
    assert tm.count_resident(0, [0, 1, 2, 3]) == 3

    run_sparse_decode(impl, layer, q, kv_cache, md, output)

    s = tm.stats()
    # All 4 candidate blocks selected (top_k == num_blocks), of which 3 were
    # resident at selection time.
    assert s.selected_total == num_blocks
    assert s.selected_on_gpu == 3
    # Invariant required by the spec.
    assert 0 <= s.selected_on_gpu <= s.selected_total
    # ensure_resident must have made block 0 resident again afterwards.
    assert tm.residency.state(0, 0) == ResidencyState.ON_GPU


def test_run_sparse_decode_waits_on_ensure_resident_event(cuda):
    """When stream_pool is set, ensure_resident returns an Event;
    run_sparse_decode must wait on it before calling flash_attn_with_kvcache.
    Force one block to start ON_CPU so a real async H2D fires; correctness
    requires the wait."""
    pytest.importorskip("flash_attn")
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from flash_attn import flash_attn_with_kvcache

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
        num_blocks,
        2,
        block_size,
        num_kv_heads,
        head_size,
        dtype=torch.float16,
        device="cuda",
    )
    # Capture the originals before any mutation (for restoring block 0).
    orig_k0 = kv_cache[0, 0].clone()
    orig_v0 = kv_cache[0, 1].clone()

    summary = BlockSummaryStore(
        num_layers=1,
        max_blocks=num_blocks,
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=torch.float16,
        device="cuda",
    )
    for b in range(num_blocks):
        summary.on_block_filled(0, b, kv_cache[b, 0])
    residency = BlockResidency(num_layers=1, max_blocks=num_blocks)
    cpu_store = CpuKvBackingStore(
        num_layers=1,
        blocks_per_layer=num_blocks,
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=torch.float16,
    )
    pool = QuestStreamPool()
    # gpu_k / gpu_v share storage with kv_cache via select(1, ...).
    tm = TierManager(
        layer_idx=0,
        gpu_budget=num_blocks,
        gpu_k=kv_cache.select(1, 0),
        gpu_v=kv_cache.select(1, 1),
        summary_store=summary,
        residency=residency,
        cpu_store=cpu_store,
        stream_pool=pool,
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
    # Default to None so run_sparse_decode falls back to the torch oracle
    # rather than auto-synthesizing a Mock for this attribute.
    layer._quest_selection_callable_ref = None

    q = torch.randn(1, num_heads, head_size, dtype=torch.float16, device="cuda")
    md = SimpleNamespace(
        num_actual_tokens=1,
        max_query_len=1,
        slot_mapping=torch.tensor(
            [num_blocks * block_size - 1], dtype=torch.int64, device="cuda"
        ),
        block_table=torch.arange(
            num_blocks, dtype=torch.int32, device="cuda"
        ).unsqueeze(0),
        seq_lens=torch.tensor(
            [num_blocks * block_size], dtype=torch.int32, device="cuda"
        ),
        max_seq_len=num_blocks * block_size,
        quest_top_k=num_blocks,
        quest_layer_indices=torch.zeros(1, dtype=torch.int32, device="cuda"),
        sparse_block_table=None,
    )
    output = torch.empty(1, num_heads, head_size, dtype=torch.float16, device="cuda")
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
    full_bt = torch.arange(num_blocks, dtype=torch.int32, device="cuda").unsqueeze(0)
    full_cs = torch.tensor([num_blocks * block_size], dtype=torch.int32, device="cuda")
    ref = flash_attn_with_kvcache(
        q.unsqueeze(1),
        ref_kv[:, 0],
        ref_kv[:, 1],
        block_table=full_bt,
        cache_seqlens=full_cs,
        causal=True,
    )
    # Async path output must match dense FA reference.
    assert torch.allclose(out, ref.squeeze(1), atol=1e-3, rtol=1e-3)


def test_run_sparse_decode_uses_layer_callable_ref(cuda):
    """When `_quest_selection_callable_ref` is stashed on the layer,
    run_sparse_decode calls it instead of importing torch oracle."""
    pytest.importorskip("flash_attn")
    from vllm.v1.attention.backends.quest.impl_helpers import (
        run_sparse_decode,
    )
    from vllm.v1.attention.ops.quest_selection_torch import (
        quest_selection_torch,
    )

    impl, layer, q, kv_cache, md, output = _build_real_path_state()
    calls = []

    def spy(*args, **kwargs):
        calls.append(kwargs.get("top_k"))
        return quest_selection_torch(*args, **kwargs)

    layer._quest_selection_callable_ref = spy
    run_sparse_decode(impl, layer, q, kv_cache, md, output)
    assert len(calls) >= 1, "run_sparse_decode never called the layer ref"


@pytest.mark.parametrize(
    "selection_impl",
    ["torch", "triton", "cuda"],
)
def test_run_sparse_decode_dispatches_per_selection_impl(
    cuda,
    selection_impl,
):
    """run_sparse_decode produces equivalent output across all three
    selection_impl values when the same query / kv state is used.

    Equivalence is set-of-block-ids (not numeric attention output),
    because top-k tie-break order can differ across kernels — the
    existing Phase B / triton tests already establish this convention.
    """
    pytest.importorskip("flash_attn")
    if selection_impl == "cuda":
        try:
            import vllm._C  # noqa: F401
        except ImportError as exc:
            pytest.skip(f"vllm._C not built: {exc}")

    from vllm.v1.attention.ops.quest_selection_dispatch import (
        _resolve_selection_callable,
    )
    from vllm.v1.attention.backends.quest.impl_helpers import (
        run_sparse_decode,
    )

    impl, layer, q, kv_cache, md, output = _build_real_path_state()
    layer._quest_selection_callable_ref = _resolve_selection_callable(
        selection_impl,
    )

    out = run_sparse_decode(impl, layer, q, kv_cache, md, output)
    assert out is not None
    assert torch.isfinite(out).all(), f"{selection_impl}: non-finite values in output"


def _build_arena_path_state(cap, num_full_blocks=6, with_partial=True,
                            partial_len=128, seed=0):
    """Stage 2A arena fixture. Like _build_real_path_state but the TierManager
    has a genuinely BOUNDED private arena (gpu_budget=cap, which may be < the
    number of logical blocks) and keeps the engine tensor as engine_kv_cache.

    The engine kv_cache is fully populated for every logical block (prefill
    residency); the arena is NOT pre-populated — the one-shot trim
    (notify_filled_blocks_after_decode -> trim_to_working_set) establishes it.
    Block summaries ARE registered for all full blocks so selection can score.

    md.block_table is an identity row: logical block b -> engine slot b. The
    sequence spans num_full_blocks full blocks plus an optional partial block
    of `partial_len` tokens (sl = num_full_blocks*block_size + partial_len when
    with_partial else num_full_blocks*block_size).

    Returns (impl, layer, q, kv_cache, md, output, full_blocks, sl).
    """
    from types import SimpleNamespace
    from unittest.mock import MagicMock

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

    torch.manual_seed(seed)
    block_size = 256
    num_kv_heads = num_heads = 2
    head_size = 64
    full_blocks = num_full_blocks
    residual = partial_len if with_partial else 0
    sl = full_blocks * block_size + residual
    # Engine cache must have a slot for every logical block (full + partial).
    num_blocks = full_blocks + (1 if with_partial else 0)
    # __ARENA_HELPER_PART2__
    kv_cache = torch.randn(
        num_blocks, 2, block_size, num_kv_heads, head_size,
        dtype=torch.float16, device="cuda",
    )
    k_view = kv_cache[:, 0]
    summary = BlockSummaryStore(
        num_layers=1, max_blocks=num_blocks, block_size=block_size,
        num_kv_heads=num_kv_heads, head_size=head_size,
        dtype=torch.float16, device="cuda",
    )
    # Register summaries for the FULL blocks only (partial cannot be scored).
    for b in range(full_blocks):
        summary.on_block_filled(0, b, k_view[b])
    residency = BlockResidency(num_layers=1, max_blocks=num_blocks)
    cpu_store = CpuKvBackingStore(
        num_layers=1, blocks_per_layer=num_blocks, block_size=block_size,
        num_kv_heads=num_kv_heads, head_size=head_size, dtype=torch.float16,
    )
    # Private bounded arena: cap blocks, NOT a view of kv_cache.
    gpu_k = torch.empty(cap, block_size, num_kv_heads, head_size,
                        dtype=torch.float16, device="cuda")
    gpu_v = torch.empty_like(gpu_k)
    tm = TierManager(
        layer_idx=0, gpu_budget=cap, gpu_k=gpu_k, gpu_v=gpu_v,
        summary_store=summary, residency=residency, cpu_store=cpu_store,
        engine_kv_cache=kv_cache, gpu_pool_aliases_kv_cache=False,
    )

    layer = MagicMock()
    layer.layer_idx = 0
    layer.num_heads = num_heads
    layer.num_kv_heads = num_kv_heads
    layer.head_size = head_size
    layer.scale = 1.0 / (head_size**0.5)
    layer._k_scale = torch.tensor(1.0, dtype=torch.float16, device="cuda")
    layer._v_scale = torch.tensor(1.0, dtype=torch.float16, device="cuda")
    layer.attn_type = "decoder"
    layer.causal = True
    layer.tier_manager = tm
    layer._quest_selection_callable_ref = None

    q = torch.randn(1, num_heads, head_size, dtype=torch.float16, device="cuda")
    md = SimpleNamespace(
        num_actual_tokens=1,
        max_query_len=1,
        slot_mapping=torch.tensor([sl - 1], dtype=torch.int64, device="cuda"),
        block_table=torch.arange(
            num_blocks, dtype=torch.int32, device="cuda",
        ).unsqueeze(0),
        seq_lens=torch.tensor([sl], dtype=torch.int32, device="cuda"),
        max_seq_len=sl,
        quest_top_k=min(full_blocks, cap - 1),
        quest_layer_indices=torch.zeros(1, dtype=torch.int32, device="cuda"),
        sparse_block_table=None,
    )
    output = torch.empty(
        1, num_heads, head_size, dtype=torch.float16, device="cuda",
    )
    impl = SimpleNamespace(kv_cache_dtype="auto")
    return impl, layer, q, kv_cache, md, output, full_blocks, sl


def test_decode_live_block_lands_in_arena(cuda):
    """After a decode step, the live partial block is readable from the arena
    slot keyed (seq, full_blocks), matching the engine slot; trim fired."""
    import torch
    from vllm.v1.attention.backends.quest.impl_helpers import (
        notify_filled_blocks_after_decode,
    )
    impl, layer, q, kv_cache, md, output, full_blocks, sl = \
        _build_arena_path_state(cap=4, num_full_blocks=6, with_partial=True)
    tm = layer.tier_manager
    notify_filled_blocks_after_decode(layer, kv_cache, md)
    assert getattr(tm, "_trimmed", set())  # trim fired
    live_slot = tm.logical_to_slot(0, full_blocks)  # the partial block
    phys = int(md.block_table[0, full_blocks].item())
    assert torch.equal(tm.gpu_k[live_slot], kv_cache[phys, 0])


def test_arena_decode_reload_is_lossless_vs_no_spill(cuda):
    """Offload-correctness via a cap A/B. Same top_k (<= cap), same query/KV.
    A small arena (cap_small) forces spill+reload; a large arena (cap_big) never
    spills. The decode outputs must MATCH because reloaded KV is bit-identical to
    what was spilled. (We canNOT use 'select-all == dense' here: with cap < blocks
    you cannot gather more than `cap` blocks in one flash_attn call, and the R1
    '== dense' invariant inherently needs all selected blocks resident. The cap
    A/B isolates the offload round-trip instead.)

    NOTE: this is a post-revert CORRECTNESS assertion, not a red-first test —
    Stage-0's gather is cap-agnostic (always reads the engine kv_cache), so it
    reads the same intact source for both caps and would pass trivially. The
    red-first guard that the read source actually moved to the arena is
    test_arena_decode_reads_arena_not_engine below (it clobbers the engine)."""
    import torch
    from vllm.v1.attention.backends.quest.impl_helpers import (
        notify_filled_blocks_after_decode, run_sparse_decode,
    )
    # 6 full blocks + partial; top_k=3 (<= both caps). cap_small=4 forces spill
    # of blocks {0,1,2} at trim (keeps last 3 + live); cap_big=8 holds all.
    def run(cap):
        impl, layer, q, kv_cache, md, output, full_blocks, sl = \
            _build_arena_path_state(cap=cap, num_full_blocks=6, with_partial=True,
                                    seed=0)
        md.quest_top_k = 3
        notify_filled_blocks_after_decode(layer, kv_cache, md)
        run_sparse_decode(impl, layer, q, kv_cache, md, output)
        return output.clone(), layer.tier_manager.stats()
    out_small, st_small = run(4)
    out_big, st_big = run(8)
    assert torch.allclose(out_small, out_big, atol=2e-3, rtol=2e-3), \
        "reload must be lossless: small-arena output != large-arena output"
    assert st_small.evict_d2h > 0 and st_small.load_h2d >= 0  # small arena spilled
    assert st_big.evict_d2h == 0  # big arena never spilled


def test_arena_decode_reads_arena_not_engine(cuda):
    """Red-first guard for the Task-5 revert: the decode gather must read the
    Quest ARENA (tm.gpu_k/gpu_v via logical_to_slot), NOT the engine kv_cache.

    We populate the arena via notify, then CLOBBER the engine kv_cache with
    garbage (the arena + CPU copies, taken before the clobber, stay correct).
    A correct arena read reproduces the dense reference computed before the
    clobber; the Stage-0 engine read would instead see garbage. cap=8 with
    top_k>=full_blocks keeps every block resident (no overflow), so ==dense is
    valid here."""
    import torch
    from flash_attn import flash_attn_with_kvcache
    from vllm.v1.attention.backends.quest.impl_helpers import (
        notify_filled_blocks_after_decode, run_sparse_decode,
    )
    impl, layer, q, kv_cache, md, output, full_blocks, sl = \
        _build_arena_path_state(cap=8, num_full_blocks=6, with_partial=True,
                                partial_len=128, seed=0)
    md.quest_top_k = full_blocks  # select all -> ==dense valid, all resident
    n_gather = full_blocks + 1  # 6 full + 1 partial
    bt = md.block_table[0, :n_gather].to(torch.int32).unsqueeze(0)
    cs = torch.tensor([sl], dtype=torch.int32, device="cuda")
    dense = flash_attn_with_kvcache(
        q[0:1].unsqueeze(1), kv_cache[:, 0], kv_cache[:, 1],
        block_table=bt, cache_seqlens=cs, causal=True).squeeze(1)
    notify_filled_blocks_after_decode(layer, kv_cache, md)
    # Clobber the ENGINE cache: only an arena read survives this.
    kv_cache.fill_(99.0)
    run_sparse_decode(impl, layer, q, kv_cache, md, output)
    assert torch.allclose(output, dense.reshape_as(output), atol=2e-3, rtol=2e-3), \
        "decode must gather from the arena, not the clobbered engine cache"


def test_partial_block_unscored_pinned_and_transitions(cuda):
    """Stage A+B: the live partial block is never a selection candidate, stays
    arena-resident (pinned) under eviction pressure, and converts to a normal
    full block on the next boundary without leaking an arena slot."""
    import torch
    from vllm.v1.attention.backends.quest.impl_helpers import (
        notify_filled_blocks_after_decode, run_sparse_decode,
    )
    # cap small so eviction pressure exists; sl just past a boundary (partial=1).
    impl, layer, q, kv_cache, md, output, fb, sl = _build_arena_path_state(
        cap=4, num_full_blocks=5, with_partial=True, partial_len=1, seed=0)
    tm = layer.tier_manager
    md.quest_top_k = 3  # top_k <= cap-1
    notify_filled_blocks_after_decode(layer, kv_cache, md)
    # (A) the partial block id == fb is NOT scored: on_block_filled (which bumps
    #     block_filled) fires only for FULL blocks, never for the live partial.
    bf_before = tm.stats().block_filled
    # (B) the live block is resident and pinned: a decode step that selects and
    #     reloads other blocks must not evict it.
    run_sparse_decode(impl, layer, q, kv_cache, md, output)
    assert tm.is_resident(0, fb), "live partial block must stay arena-resident"
    # (C) transition: advance sl to the next boundary; the (now-full) block fb
    #     gets a summary via on_block_filled and keeps its single arena slot.
    md.seq_lens = torch.tensor([(fb + 1) * tm.gpu_k.shape[1]],
                               device=md.seq_lens.device, dtype=md.seq_lens.dtype)
    notify_filled_blocks_after_decode(layer, kv_cache, md)
    assert tm.stats().block_filled == bf_before + 1, "fb should fill exactly once"
    # the key (0, fb) maps to exactly one arena slot (no double-occupancy)
    keys_for_fb = [k for k in tm._slot_map._key_to_slot if k == (0, fb)]
    assert len(keys_for_fb) == 1


@pytest.mark.parametrize("extra_tokens", [0, 1, 128])
def test_partial_block_attention_use_matches_dense(cuda, extra_tokens):
    """Stage C: sl = N*block_size + extra_tokens. extra=0 (no partial),
    1 (minimal), 128 (mid). Arena large (no overflow), top_k >= full_blocks so
    select-all == dense. Verifies gather order + cache_seqlens math."""
    import torch
    from flash_attn import flash_attn_with_kvcache
    from vllm.v1.attention.backends.quest.impl_helpers import (
        notify_filled_blocks_after_decode, run_sparse_decode,
    )
    full_blocks = 3
    impl, layer, q, kv_cache, md, output, fb, sl = _build_arena_path_state(
        cap=8, num_full_blocks=full_blocks, with_partial=(extra_tokens > 0),
        partial_len=extra_tokens, seed=0)
    md.quest_top_k = full_blocks  # select all -> ==dense valid
    n_gather = full_blocks + (1 if extra_tokens > 0 else 0)
    bt = md.block_table[0, :n_gather].to(torch.int32).unsqueeze(0)
    cs = torch.tensor([sl], dtype=torch.int32, device="cuda")
    dense = flash_attn_with_kvcache(q[0:1].unsqueeze(1), kv_cache[:, 0],
                                    kv_cache[:, 1], block_table=bt,
                                    cache_seqlens=cs, causal=True).squeeze(1)
    notify_filled_blocks_after_decode(layer, kv_cache, md)
    run_sparse_decode(impl, layer, q, kv_cache, md, output)
    assert torch.allclose(output, dense.reshape_as(output), atol=2e-3, rtol=2e-3), \
        f"attention-use mismatch at extra_tokens={extra_tokens}"

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage 2C-v2 Stage B: Quest-owned KV write + per-layer offload, sourced from
the layer's key/value forward args (NOT a kv_cache tensor).

Under footprint_kvshare the non-full-KV Quest layers are aliased to ONE physical
scratch tensor and the engine's auto-write is skipped, so the 2A path that reads
prompt/decode KV back from the engine cache reads garbage. These tests prove the
key/value-sourced ingest reproduces the SAME arena + CPU residency as the 2A
engine-sourced trim_to_working_set, at the TierManager level (no engine).
"""
from __future__ import annotations

import pytest
import torch


def _build(gpu_budget=4, cpu_budget=16, **kw):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    from vllm.v1.attention.backends.quest.cache.block_summary import (
        BlockSummaryStore,
    )
    from vllm.v1.attention.backends.quest.cache.cpu_backing_store import (
        CpuKvBackingStore,
    )
    from vllm.v1.attention.backends.quest.cache.residency import BlockResidency
    from vllm.v1.attention.backends.quest.cache.tier_manager import TierManager

    bs, h, d, max_blocks = 4, 1, 8, 64
    summary = BlockSummaryStore(
        num_layers=1, max_blocks=max_blocks, block_size=bs,
        num_kv_heads=h, head_size=d, dtype=torch.float16, device="cuda",
    )
    cpu_store = CpuKvBackingStore(
        num_layers=1, blocks_per_layer=cpu_budget, block_size=bs,
        num_kv_heads=h, head_size=d, dtype=torch.float16,
    )
    residency = BlockResidency(num_layers=1, max_blocks=max_blocks)
    gpu_k = torch.zeros((gpu_budget, bs, h, d), dtype=torch.float16, device="cuda")
    gpu_v = torch.zeros_like(gpu_k)
    return TierManager(
        layer_idx=0, gpu_budget=gpu_budget, gpu_k=gpu_k, gpu_v=gpu_v,
        summary_store=summary, residency=residency, cpu_store=cpu_store, **kw,
    )


def _prompt_kv(num_full_blocks, bs=4, h=1, d=8):
    """Build a (num_full_blocks*bs, h, d) key/value where block b is filled with
    distinct values so we can check which block landed where."""
    n = num_full_blocks * bs
    key = torch.empty((n, h, d), dtype=torch.float16, device="cuda")
    val = torch.empty((n, h, d), dtype=torch.float16, device="cuda")
    for b in range(num_full_blocks):
        key[b * bs:(b + 1) * bs] = float(b + 1)
        val[b * bs:(b + 1) * bs] = float(b + 1 + 100)
    return key, val


def test_prefill_ingest_kvshare_keeps_last_cap_minus_1_in_arena():
    """prefill_ingest_kvshare keeps the last (cap-1) full blocks in the arena
    (1 slot reserved for the live decode block) and spills the rest to CPU —
    the same keep/spill split as trim_to_working_set, but sourced from
    key/value."""
    from vllm.v1.attention.backends.quest.cache.residency import ResidencyState

    cap = 4  # arena holds 4; keep cap-1 = 3, spill the rest
    tm = _build(gpu_budget=cap)
    num_full = 6
    key, val = _prompt_kv(num_full)

    tm.prefill_ingest_kvshare(
        seq_id=0, num_full_blocks=num_full, key=key, value=val, block_size=4,
    )

    # Last cap-1 = blocks {3,4,5} resident in arena; first {0,1,2} spilled.
    for b in (3, 4, 5):
        assert tm.is_resident(0, b), f"block {b} should be in arena"
        slot = tm.logical_to_slot(0, b)
        assert torch.equal(tm.gpu_k[slot], torch.full((4, 1, 8), float(b + 1),
                                                       dtype=torch.float16,
                                                       device="cuda"))
    for b in (0, 1, 2):
        assert not tm.is_resident(0, b), f"block {b} should be spilled"
        assert tm.residency.state(0, b) == ResidencyState.ON_CPU
    # Every block's summary was registered (selection needs it).
    # (block 0 summary survives the spill — that's the whole point.)
    assert tm.stats().evict_d2h == 3  # 3 spilled (write-back default)


def test_prefill_ingest_kvshare_spilled_block_reloads_bit_equal():
    """A spilled prompt block reloads (H2D) bit-equal — proving the CPU backup
    written from key/value is faithful."""
    cap = 4
    tm = _build(gpu_budget=cap)
    num_full = 6
    key, val = _prompt_kv(num_full)
    tm.prefill_ingest_kvshare(seq_id=0, num_full_blocks=num_full,
                              key=key, value=val, block_size=4)

    # Reload spilled block 0.
    tm.ensure_resident(seq_id=0, logical_block_ids=torch.tensor([0], device="cuda"))
    slot = tm.logical_to_slot(0, 0)
    assert torch.equal(tm.gpu_k[slot], torch.full((4, 1, 8), 1.0,
                                                  dtype=torch.float16, device="cuda"))
    assert torch.equal(tm.gpu_v[slot], torch.full((4, 1, 8), 101.0,
                                                  dtype=torch.float16, device="cuda"))


def test_decode_append_kvshare_accumulates_live_block_then_promotes():
    """append_decode_token_kvshare stages the live block token-by-token in the
    arena; on the boundary the completed block is registered as a normal full
    block (summary + ON_GPU) in place — no scratch round-trip."""
    from vllm.v1.attention.backends.quest.cache.residency import ResidencyState

    cap = 8
    tm = _build(gpu_budget=cap)
    bs = 4
    # Pretend prefill left 2 full blocks (8 tokens). Decode appends tokens for
    # logical positions 8,9,10,11 -> completes block 2 at position 11 (sl=12).
    key, val = _prompt_kv(2)
    tm.prefill_ingest_kvshare(seq_id=0, num_full_blocks=2, key=key, value=val,
                              block_size=bs)

    # 4 decode steps fill block index 2.
    for i in range(bs):
        sl = 8 + i  # seq length BEFORE this token is 8+i; token at position 8+i
        k_tok = torch.full((1, 1, 8), 3.0, dtype=torch.float16, device="cuda")
        v_tok = torch.full((1, 1, 8), 103.0, dtype=torch.float16, device="cuda")
        tm.append_decode_token_kvshare(
            seq_id=0, seq_len_before=sl, k_tok=k_tok, v_tok=v_tok, block_size=bs,
        )

    # Block 2 is now a complete full block, resident, summarized.
    assert tm.is_resident(0, 2)
    slot = tm.logical_to_slot(0, 2)
    assert torch.equal(tm.gpu_k[slot], torch.full((4, 1, 8), 3.0,
                                                  dtype=torch.float16, device="cuda"))
    assert tm.residency.state(0, 2) == ResidencyState.ON_GPU

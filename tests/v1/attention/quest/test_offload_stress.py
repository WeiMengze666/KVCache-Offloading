# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Content-based randomized stress test for the Quest GPU/CPU offload + LRU.

This test does NOT run attention. It validates the TWO-TIER STORAGE + LRU round
trip in isolation: every logical block is filled with a content fingerprint
derived from its block id, so after any sequence of trims / decode writes /
ensure_resident reloads we can read the arena slot back and assert it still holds
the RIGHT block's bytes. Verification uses a cheap structural reduction (odd
lane-indices summed, even lane-indices subtracted) instead of attention, so a
mismatch localizes to the offload machinery, not the kernel.

It mirrors the REAL engine's per-step call order in notify_filled_blocks_after_decode
(trim-on-first-decode, on_block_filled at a boundary, write_live_block for the
partial) and deliberately crosses block boundaries, since that is the suspected
trigger. Each run overwrites a JSON trace (the random top_k picks per step) so a
failure is externally reproducible; copy the file aside to keep it.
"""
from __future__ import annotations

import json
import os
import random

import pytest
import torch

_TRACE_PATH = os.environ.get(
    "QUEST_STRESS_TRACE", "/tmp/quest_offload_stress_trace.json"
)


def _fingerprint_block(block_id: int, bs: int, h: int, d: int) -> torch.Tensor:
    """A (bs, h, d) block whose content encodes block_id deterministically, so
    a misrouted slot is detectable. Distinct per (id, lane)."""
    base = torch.full((bs, h, d), float(block_id), dtype=torch.float16,
                      device="cuda")
    lane = torch.arange(bs, device="cuda", dtype=torch.float16).view(bs, 1, 1)
    return base + lane * 0.01  # block_id in the integer part, lane in the frac


def _reduce(block: torch.Tensor) -> torch.Tensor:
    """Structural reduction standing in for attention: sum odd lanes, subtract
    even lanes (over the bs axis). Deterministic, cheap, and sensitive to which
    block's bytes are present."""
    bs = block.shape[0]
    sign = torch.ones(bs, device=block.device, dtype=block.dtype)
    sign[::2] = -1.0  # even lanes subtracted, odd lanes added
    return (block.float() * sign.view(bs, 1, 1).float()).sum(dim=0)


def _build_tm(cap, bs, h, d, max_blocks, write_through=False):
    from vllm.v1.attention.backends.quest.cache.block_summary import BlockSummaryStore
    from vllm.v1.attention.backends.quest.cache.cpu_backing_store import CpuKvBackingStore
    from vllm.v1.attention.backends.quest.cache.residency import BlockResidency
    from vllm.v1.attention.backends.quest.cache.tier_manager import TierManager

    summary = BlockSummaryStore(num_layers=1, max_blocks=max_blocks, block_size=bs,
                                num_kv_heads=h, head_size=d,
                                dtype=torch.float16, device="cuda")
    cpu = CpuKvBackingStore(num_layers=1, blocks_per_layer=max_blocks, block_size=bs,
                            num_kv_heads=h, head_size=d, dtype=torch.float16)
    res = BlockResidency(num_layers=1, max_blocks=max_blocks)
    return TierManager(
        layer_idx=0, gpu_budget=cap,
        gpu_k=torch.empty(cap, bs, h, d, dtype=torch.float16, device="cuda"),
        gpu_v=torch.empty(cap, bs, h, d, dtype=torch.float16, device="cuda"),
        summary_store=summary, residency=res, cpu_store=cpu,
        gpu_pool_aliases_kv_cache=False,
        enable_write_through=write_through,
    )


# __STRESS_PART2__


def _engine_kv_with_fingerprints(num_blocks, bs, h, d):
    """Engine FA-layout tensor (nb,2,bs,h,d); block b's K and V both carry the
    fingerprint for id b, so a misrouted physical/arena slot is detectable."""
    kv = torch.empty(num_blocks, 2, bs, h, d, dtype=torch.float16, device="cuda")
    for b in range(num_blocks):
        fp = _fingerprint_block(b, bs, h, d)
        kv[b, 0].copy_(fp)
        kv[b, 1].copy_(fp)
    return kv


@pytest.mark.parametrize("write_through", [False, True])
@pytest.mark.parametrize("seed", list(range(8)))
def test_offload_lru_content_roundtrip(seed, write_through):
    """Drive the exact engine per-step sequence over many decode steps that
    cross block boundaries, with random top_k <= cap-1 each step, and after
    every step assert each selected block's arena slot still holds THAT block's
    fingerprint (so the offload + two-pass ensure_resident round trip is
    content-lossless and never KeyErrors). Parametrized over write-back (2A,
    write_through=False) and write-through (2B): both must be lossless."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    from vllm.v1.attention.backends.quest.cache.tier_manager import TierManager  # noqa: F401

    rng = random.Random(seed)
    cap, bs, h, d = 8, 16, 2, 8
    P0 = 11                      # prompt full-block count at first decode
    n_steps = 40                 # enough decode steps to cross several boundaries
    max_blocks = P0 + n_steps + 4
    engine = _engine_kv_with_fingerprints(max_blocks, bs, h, d)
    tm = _build_tm(cap, bs, h, d, max_blocks, write_through=write_through)
    # identity block_table: logical b -> engine slot b
    block_table_row = list(range(max_blocks))
    seq_id = 0
    trace = []

    # Pre-register summaries for the prompt's full blocks (prefill path does
    # this via register_prefill_summary; selection scores over them).
    for b in range(P0):
        tm.register_prefill_summary(seq_id, b, engine[b, 0])

    # Start at sl just past P0 full blocks (first decode token => 1 partial tok).
    sl = P0 * bs + 1
    for step in range(n_steps):
        full_blocks = sl // bs
        has_partial = (sl % bs) != 0
        # --- mirror notify_filled_blocks_after_decode ordering ---
        tm.trim_to_working_set(seq_id=seq_id, num_full_blocks=full_blocks,
                               kv_cache=engine, block_table_row=block_table_row)
        if sl != 0 and sl % bs == 0:
            blk = sl // bs - 1
            tm.on_block_filled(seq_id, blk, engine[blk, 0], engine[blk, 1])
            # a newly-completed full block needs a summary to be selectable
            tm.register_prefill_summary(seq_id, blk, engine[blk, 0])
        if has_partial:
            tm.write_live_block(seq_id, full_blocks,
                                engine[full_blocks, 0], engine[full_blocks, 1])

        # --- random selection over the full blocks, top_k <= cap-1 ---
        k = rng.randint(1, min(cap - 1, full_blocks))
        top_ids = rng.sample(range(full_blocks), k)
        trace.append({"step": step, "sl": sl, "full_blocks": full_blocks,
                      "has_partial": has_partial, "top_ids": top_ids})

        # --- mirror run_sparse_decode residency + gather ---
        keep = [full_blocks] if has_partial else None
        tm.ensure_resident(
            seq_id=seq_id,
            logical_block_ids=torch.tensor(top_ids, device="cuda"),
            keep_resident_ids=keep,
        )
        # Verify every selected block (and the live block) is arena-resident AND
        # carries the correct fingerprint (content round-trip).
        check_ids = list(top_ids) + ([full_blocks] if has_partial else [])
        for b in check_ids:
            slot = tm.logical_to_slot(seq_id, b)  # KeyError here == the bug
            got_k = _reduce(tm.gpu_k[slot])
            want_k = _reduce(engine[b, 0])
            assert torch.equal(got_k, want_k), (
                f"seed={seed} step={step} block {b}: arena slot {slot} holds "
                f"wrong content (got reduce {got_k.flatten()[:3].tolist()}, "
                f"want {want_k.flatten()[:3].tolist()}); trace at {_TRACE_PATH}"
            )

        sl += 1  # one decode token per step (crosses a boundary every bs steps)

    # Overwrite the trace each run; copy aside to preserve a failing seed.
    with open(_TRACE_PATH, "w") as f:
        json.dump({"seed": seed, "cap": cap, "bs": bs, "steps": trace}, f, indent=0)

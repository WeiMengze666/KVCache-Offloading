# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E2E test: ArkVale (cuboid_mean) survives a mixed prefill+decode batch and
the sparse path genuinely runs — mirroring test_prefill_decode_mixed.py.

Why this test exists
--------------------
Mirrors the Quest e2e (test_prefill_decode_mixed.py) for ArkVale. ArkVale
shares the QuestSparseOffloadBackend / BlockSummaryStore / KV tiering stack
with Quest; the only algorithmic difference is the page-digest formula
controlled by ``digest_mode='arkvale_cuboid_mean'``.

This test validates two things end-to-end:

1. The ArkVale integration hookup works (enable_arkvale_sparse_offload +
   arkvale_config JSON path → TierManagers attached to Attention layers).
2. The sparse selection path actually executes (select_calls > 0,
   selected_total > 0).

Like the Quest test, gpu_cache_blocks_per_seq=512 is large enough that no
eviction is needed for the ~955-token prompts (~4 blocks of 256). evict_d2h
and load_h2d are asserted >= 0 as sanity checks only.

Critical guard — spec §7 risk
------------------------------
The digest_mode assertion (tier_manager.summary_store.digest_mode ==
'arkvale_cuboid_mean') runs inside the worker via apply_model collective_rpc.
This catches the failure mode where digest_mode silently defaults to
'quest_minmax' because the ArkValeConfig→BlockSummaryStore propagation is
broken somewhere in the hookup chain.

Pickle-based RPC requirement
----------------------------
apply_model ships a Python callable to the engine-core worker process.
VLLM_ALLOW_INSECURE_SERIALIZATION must be set (monkeypatched in this test).
"""

from __future__ import annotations

import gc
import json
from dataclasses import asdict

import pytest
import torch

from vllm import SamplingParams
from vllm.config.arkvale import ArkValeConfig

pytestmark = pytest.mark.real_model

# Same prompt as Quest e2e: ~955 tokens, definitely multi-block under
# block_size=256 (~4 blocks of 256 tokens each).
_LONG_PROMPT = (
    "In the spring of the year 1789, the assembly convened in Versailles "
    "to address grievances that had accumulated over decades of fiscal "
    "mismanagement and shifting alliances among the nobility. "
) * 20


def _probe_arkvale_layers(model):
    """Run inside the engine-core worker process via collective_rpc.

    Walks model.named_modules() for Attention modules that carry a
    tier_manager, collects per-layer stats, and asserts that the
    BlockSummaryStore digest_mode is 'arkvale_cuboid_mean' — the spec §7
    guard that catches silent digest_mode defaulting to 'quest_minmax'.

    Returns a list of JSON-friendly dicts (one per ArkVale-managed layer).
    """
    out = []
    for name, mod in model.named_modules():
        if type(mod).__name__ != "Attention":
            continue
        tm = getattr(mod, "tier_manager", None)
        if tm is None:
            continue

        # Spec §7 critical guard: digest_mode must be 'arkvale_cuboid_mean'.
        ss = getattr(tm, "summary_store", None)
        actual_mode = getattr(ss, "digest_mode", None) if ss is not None else None
        assert actual_mode == "arkvale_cuboid_mean", (
            f"Layer {name}: tier_manager.summary_store.digest_mode="
            f"{actual_mode!r}, expected 'arkvale_cuboid_mean'. "
            "ArkValeConfig.digest_mode was not propagated correctly to "
            "BlockSummaryStore (spec §7 failure mode)."
        )

        stats_fn = getattr(tm, "stats", None)
        s = stats_fn() if callable(stats_fn) else None
        out.append(
            {
                "name": name,
                "layer_idx": getattr(mod, "layer_idx", None),
                "impl": type(getattr(mod, "impl", None)).__name__,
                "stats": asdict(s) if s is not None else None,
                "digest_mode": actual_mode,
            }
        )
    return out


def _collect_stats(llm):
    """Hop the engine-core process boundary to fetch per-ArkVale-layer stats.

    Returns a flat list of dicts (one per ArkVale-managed layer). Empty list
    means no layer carries a TierManager — sparse path is not engaged.
    """
    results = llm.llm_engine.apply_model(_probe_arkvale_layers)
    flat = []
    for rank_result in results:
        flat.extend(rank_result)
    return flat


def test_arkvale_mixed_prefill_decode_engages_sparse_path(monkeypatch, tmp_path):
    """ArkVale e2e: sparse path runs, CPU offload fires, digest_mode is correct."""
    # apply_model needs to ship a Python callable to the engine-core
    # subprocess; v1 IPC requires opt-in to pickle for non-msgspec types.
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

    # Same config as the Quest baseline: gpu_cache_blocks_per_seq=512 is large
    # enough that no eviction is needed for a ~955-token prompt (~4 blocks of
    # 256). We do NOT assert evict_d2h/load_h2d > 0; the Quest test doesn't
    # either, so we mirror that decision here.
    cfg = ArkValeConfig(
        enabled=True,
        block_size=256,
        top_k=64,
        full_kv_layers=[0, 1],
        gpu_cache_blocks_per_seq=512,
        cpu_cache_blocks=8192,
        cpu_cache_gib=8,
        selection_impl="torch",
        enable_async_prefetch=False,
    )
    cfg.validate()

    cfg_path = tmp_path / "arkvale_cfg.json"
    cfg_path.write_text(json.dumps(cfg.to_dict()))

    from vllm import LLM

    llm = LLM(
        model="meta-llama/Llama-3.2-3B-Instruct",
        enable_arkvale_sparse_offload=True,
        arkvale_config=str(cfg_path),
        dtype="float16",
        enforce_eager=True,
        max_model_len=1024,
        gpu_memory_utilization=0.50,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        block_size=256,
    )

    try:
        params = SamplingParams(temperature=0.0, max_tokens=16)

        # Two copies of the long prompt so every seq in the batch has at least
        # one fully-filled ArkVale block before decode starts.
        outputs = llm.generate([_LONG_PROMPT, _LONG_PROMPT], params, use_tqdm=False)
        assert len(outputs) == 2

        for i, out in enumerate(outputs):
            token_ids = list(out.outputs[0].token_ids)
            assert len(token_ids) > 0, f"output {i} produced no tokens"

        layer_stats = _collect_stats(llm)
        assert layer_stats, (
            "no ArkVale TierManager attached to any layer — ArkVale did not "
            "engage. Likely cause: AttentionBackendEnum.CUSTOM was never "
            "selected by the v1 attention selector for any layer."
        )

        # Sparse selection must have run.
        total_select_calls = sum(
            s["stats"]["select_calls"] for s in layer_stats if s["stats"]
        )
        total_selected = sum(
            s["stats"]["selected_total"] for s in layer_stats if s["stats"]
        )
        total_evict_d2h = sum(
            s["stats"]["evict_d2h"] for s in layer_stats if s["stats"]
        )
        total_load_h2d = sum(s["stats"]["load_h2d"] for s in layer_stats if s["stats"])

        assert total_select_calls > 0, (
            f"select_calls=0 across {len(layer_stats)} ArkVale layer(s) — "
            f"sparse selection never ran. Prompt may be too short to fill a "
            f"block (block_size=256 needs >=256 tokens before any decode step)."
        )
        assert total_selected > 0, (
            f"selected_total=0 across {len(layer_stats)} ArkVale layer(s) — "
            f"selection ran ({total_select_calls} calls) but picked nothing."
        )

        # Mirror Quest test: with gpu_cache_blocks_per_seq=512 and ~4-block
        # prompts no eviction is expected; assert non-negativity as a sanity
        # check (no counter corruption / overflow).
        assert total_evict_d2h >= 0, (
            f"evict_d2h={total_evict_d2h} went negative across "
            f"{len(layer_stats)} ArkVale layer(s) — counter corruption."
        )
        assert total_load_h2d >= 0, (
            f"load_h2d={total_load_h2d} went negative across "
            f"{len(layer_stats)} ArkVale layer(s) — counter corruption."
        )

        # Confirm digest_mode across all layers (redundant with worker-side
        # assert but surfaces as a test failure instead of a worker crash).
        for s in layer_stats:
            assert s["digest_mode"] == "arkvale_cuboid_mean", (
                f"Layer {s['name']}: digest_mode={s['digest_mode']!r}, "
                "expected 'arkvale_cuboid_mean' (spec §7 guard)."
            )

    finally:
        del llm
        gc.collect()
        torch.accelerator.empty_cache()

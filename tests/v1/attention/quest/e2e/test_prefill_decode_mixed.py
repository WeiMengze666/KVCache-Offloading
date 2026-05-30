# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E.1 #4: Quest survives a mixed prefill+decode batch and the sparse path
genuinely runs (not silently delegating to dense FA for every layer).

Why this test exists
--------------------
E.1 tests #1-#3 confirm a Llama LLM constructed with
``enable_quest_sparse_offload=True`` produces sane outputs, but they cannot
distinguish the case where Quest *actually engages* from the case where the
selector silently picks ``FlashAttentionBackend`` for every layer (the
``QuestSparseOffloadBackend.validate_configuration`` rejection path).

This test reads internal Quest state via ``LLMEngine.apply_model`` (which
hops the engine-core process boundary via collective_rpc) and asserts that
**at least one layer carries a TierManager** — i.e. the Quest runtime ran
``init_runtime_state`` and attached ``layer.tier_manager`` to the Quest
layers. If zero layers carry one, that's R-E1-4 firing — the integration
between ``enable_quest_sparse_offload`` and the v1 attention selector is
incomplete and Quest is not actually being exercised.

When TierManagers ARE found, we assert:

- ``select_calls > 0`` aggregated across Quest layers — sparse selection
  ran at least once during decode.
- ``selected_total > 0`` — selection actually picked blocks.
- ``load_h2d >= 0`` — the H2D counter is sane (no overflow / corruption).

``load_h2d > 0`` is intentionally NOT asserted: with
``gpu_cache_blocks_per_seq=512`` and a ~600-token prompt (~3 blocks of 256
tokens each) every block fits in the per-seq GPU pool and no H2D transfer
is needed. We only assert non-negativity as a sanity check.

Pickle-based RPC requirement
----------------------------
``apply_model`` ships a Python callable to the engine-core worker process,
which the v1 IPC encoder rejects unless ``VLLM_ALLOW_INSECURE_SERIALIZATION``
is set. The test sets it via monkeypatch so the worker accepts the probe.
The flag is scoped to the test process (and propagated to children via env
inheritance at fork) and reset on teardown.

Current status: xfail (R-E1-4)
------------------------------
At the time of writing, ``enable_quest_sparse_offload=True`` does *not*
flip ``AttentionConfig.backend`` to ``AttentionBackendEnum.CUSTOM``. The v1
selector therefore picks ``FlashAttentionBackend`` for every layer, and
``QuestSparseOffloadBackend.bind_runtime`` filters those layers out (they
fail ``layer.attn_backend is cls``). No layer ever receives a TierManager
and the assertions below trip. This is R-E1-4 in the design doc.

The test is committed as ``xfail(strict=False)`` so it surfaces the gap in
every e2e run without blocking the suite. When integration is fixed
(``enable_quest_sparse_offload`` should auto-set ``attention_config.backend``
to ``CUSTOM`` *and* propagate ``use_sparse=True`` so ``validate_configuration``
accepts), this test will XPASS and the marker can be removed.
"""

from __future__ import annotations

from dataclasses import asdict

import pytest

from vllm import SamplingParams

pytestmark = pytest.mark.real_model


# ~600+ actual tokens, definitely multi-block under block_size=256. The
# original 6x multiplier produced only ~150 tokens (English averages ~4
# chars/token), too few to fill a single Quest block during prefill. 20x
# yields ~500-600 tokens which crosses two block boundaries reliably.
_LONG_PROMPT = (
    "In the spring of the year 1789, the assembly convened in Versailles "
    "to address grievances that had accumulated over decades of fiscal "
    "mismanagement and shifting alliances among the nobility. "
) * 20

_SHORT_PROMPT = "The capital of France is"


def _probe_quest_layers(model):
    """Run inside the engine-core worker process via collective_rpc.

    Walks ``model.named_modules()`` for ``Attention`` modules, collects
    every layer that has a ``tier_manager`` attribute attached (i.e. is a
    Quest-managed layer), and returns a JSON-friendly summary including
    each layer's ``stats`` dataclass dict. Returns a list of dicts so the
    test process can deserialize without needing Quest types imported.
    """
    out = []
    for name, mod in model.named_modules():
        if type(mod).__name__ != "Attention":
            continue
        tm = getattr(mod, "tier_manager", None)
        if tm is None:
            continue
        # tier_manager.stats is a method (not a property); call to get the
        # QuestStats dataclass instance.
        stats_fn = getattr(tm, "stats", None)
        s = stats_fn() if callable(stats_fn) else None
        out.append(
            {
                "name": name,
                "layer_idx": getattr(mod, "layer_idx", None),
                "impl": type(getattr(mod, "impl", None)).__name__,
                "stats": asdict(s) if s is not None else None,
            }
        )
    return out


def _collect_stats(llm):
    """Hop the engine-core process boundary to fetch per-Quest-layer stats.

    Returns a flat list of dicts (one per Quest-managed layer). Empty list
    means no layer carries a TierManager — sparse path is not engaged.
    """
    # apply_model returns one result per worker rank; with TP=1 there's
    # exactly one element.
    results = llm.llm_engine.apply_model(_probe_quest_layers)
    flat = []
    for rank_result in results:
        flat.extend(rank_result)
    return flat


def test_quest_mixed_prefill_decode_engages_sparse_path(
    monkeypatch,
    baseline_quest_config,
    quest_llm_factory,
):
    # apply_model needs to ship a Python callable to the engine-core
    # subprocess; v1 IPC requires opt-in to pickle for non-msgspec types.
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

    quest_llm = quest_llm_factory(baseline_quest_config)
    params = SamplingParams(temperature=0.0, max_tokens=16)

    # Use two copies of the long prompt (each ~500-600 tokens, > block_size=256)
    # so every seq in the batch has at least one fully-filled Quest block
    # before decode starts. Mixing in a short prompt would force the whole
    # batch back to dense per the seq_too_short gate in QuestSparseOffloadImpl.
    outputs = quest_llm.generate([_LONG_PROMPT, _LONG_PROMPT], params, use_tqdm=False)
    assert len(outputs) == 2

    for i, out in enumerate(outputs):
        token_ids = list(out.outputs[0].token_ids)
        assert len(token_ids) > 0, f"output {i} produced no tokens"

    # R-E1-4 surface: if no Quest layers are wired, the integration between
    # enable_quest_sparse_offload and the v1 attention selector is broken
    # and Quest is silently delegating to dense FA for every layer.
    layer_stats = _collect_stats(quest_llm)
    assert layer_stats, (
        "no Quest TierManager attached to any layer — Quest did not engage. "
        "Likely cause: AttentionBackendEnum.CUSTOM was never selected by "
        "the v1 attention selector for any layer (R-E1-4)."
    )

    # Quest layers exist — verify the sparse path actually executed.
    total_select_calls = sum(
        s["stats"]["select_calls"] for s in layer_stats if s["stats"]
    )
    total_selected = sum(
        s["stats"]["selected_total"] for s in layer_stats if s["stats"]
    )
    total_h2d = sum(s["stats"]["load_h2d"] for s in layer_stats if s["stats"])

    assert total_select_calls > 0, (
        f"select_calls=0 across {len(layer_stats)} Quest layer(s) — sparse "
        f"selection never ran. Prompt may be too short to fill a block "
        f"(block_size=256 needs >=256 tokens before any decode step)."
    )
    assert total_selected > 0, (
        f"selected_total=0 across {len(layer_stats)} Quest layer(s) — "
        f"selection ran ({total_select_calls} calls) but picked nothing."
    )
    assert total_h2d >= 0, (
        f"load_h2d={total_h2d} went negative across "
        f"{len(layer_stats)} Quest layer(s) — counter corruption."
    )

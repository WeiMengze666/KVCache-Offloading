# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""R1 alignment: Quest (top_k = ALL) vs dense FA on a real model.

Spec appendix A.7: with top_k larger than the prompt's block count, Quest
selection degenerates to "pick every block", so the sparse decode path should
be numerically equivalent to dense FlashAttention (the R1 invariant). We assert
the greedy text matches.

Three things were wrong here historically and are now fixed:
  1. Model was TinyLlama (not cached on this machine, network blocked). Unified
     on ``meta-llama/Llama-3.2-3B-Instruct`` (the same model the rest of the
     e2e suite uses via ``QUEST_E2E_MODEL_ID``).
  2. Both engines were built in one process → the upstream EngineCore
     CUDA-context leak OOM'd the second engine. Each engine now runs in its own
     spawned subprocess.
  3. The ``QuestConfig`` was constructed but never actually plumbed into the
     LLM (a dead variable) — Quest was not really enabled. Now enabled via the
     official ``enable_quest_sparse_offload=True`` + ``quest_config=<json>``
     path.

CURRENTLY PASSING: this exercises the real sparse decode path. Bug "B1" (the
trailing partial block holding the live decode token was dropped from the
gather, so the decode token could not attend to itself) AND the coupled
pool-slot aliasing bug (selected full blocks were read from / written to Quest
LRU pool slots that alias the wrong engine kv_cache rows) are now fixed: the
sparse decode gathers every selected full block plus the trailing partial block
straight from the engine block_table and uses the true cache_seqlens, so with
top_k >= num_blocks the greedy text matches dense FA exactly (the R1 invariant).
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
from pathlib import Path

import pytest
import torch

# Same model the rest of the quest e2e suite uses (cached locally). This file
# lives in quest/ (not quest/e2e/), so we don't import the e2e conftest; keep a
# local constant to avoid a cross-package import at collection time.
QUEST_E2E_MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"

# Long enough (~600 tokens > block_size=256) to clear the seq_too_short gate so
# the REAL sparse decode path engages. A short prompt would fall back to dense
# delegation for every layer and never test the R1 invariant at all.
_PROMPT = (
    "The Eiffel Tower was completed in 1889 for the World's Fair held in "
    "Paris to celebrate the centennial of the French Revolution. It stood as "
    "the tallest man-made structure in the world for forty-one years until "
    "the Chrysler Building was finished in New York. "
) * 12 + "\nQ: In what year was the Eiffel Tower completed?\nA:"

_SHARED_KWARGS = dict(
    dtype="float16",
    enforce_eager=True,
    max_model_len=1024,
    gpu_memory_utilization=0.50,
    enable_prefix_caching=False,
    enable_chunked_prefill=False,
    block_size=256,
)


def _engine_worker(out_path: str, quest_json_path: str | None) -> None:
    """Build one engine in a spawned subprocess, generate greedily, dump text."""
    from vllm import LLM, SamplingParams

    if quest_json_path is None:
        llm = LLM(model=QUEST_E2E_MODEL_ID, **_SHARED_KWARGS)
    else:
        llm = LLM(
            model=QUEST_E2E_MODEL_ID,
            enable_quest_sparse_offload=True,
            quest_config=quest_json_path,
            **_SHARED_KWARGS,
        )
    sp = SamplingParams(max_tokens=8, temperature=0.0)
    text = llm.generate([_PROMPT], sp, use_tqdm=False)[0].outputs[0].text
    Path(out_path).write_text(json.dumps({"text": text}))


# Long prompt (~3000+ tokens) for the forced-overflow cap A/B test: it must
# exceed SMALL_CAP blocks (8 * 256 = 2048 tokens) so the small-arena engine
# genuinely spills, while the large-arena engine holds everything. The default
# _PROMPT (~600 tokens) is far too short to overflow an 8-block arena.
_LONG_PROMPT = (
    "The Eiffel Tower was completed in 1889 for the World's Fair held in "
    "Paris to celebrate the centennial of the French Revolution. It stood as "
    "the tallest man-made structure in the world for forty-one years until "
    "the Chrysler Building was finished in New York. "
) * 60 + "\nQ: In what year was the Eiffel Tower completed?\nA:"

# Same as _SHARED_KWARGS but with a longer context window so the long prompt
# fits and the sequence spans well over SMALL_CAP blocks. max_num_seqs=1 pins
# the engine to the concurrency=1 scope the Quest offload design targets — and
# is what makes Stage 2B write-through's host-pool sizing satisfiable (need =
# cdiv(max_model_len, block_size) * max_num_seqs = 16 blocks/layer, not 4096).
_LONG_KWARGS = dict(_SHARED_KWARGS, max_model_len=4096, max_num_seqs=1)


def _engine_worker_long(out_path: str, quest_json_path: str) -> None:
    """Forced-overflow worker: long prompt + 4096 context, Quest always on."""
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=QUEST_E2E_MODEL_ID,
        enable_quest_sparse_offload=True,
        quest_config=quest_json_path,
        **_LONG_KWARGS,
    )
    sp = SamplingParams(max_tokens=8, temperature=0.0)
    text = llm.generate([_LONG_PROMPT], sp, use_tqdm=False)[0].outputs[0].text
    Path(out_path).write_text(json.dumps({"text": text}))


def _run_long_engine_in_subprocess(out_path: str, quest_json_path: str) -> None:
    ctx = mp.get_context("spawn")
    p = ctx.Process(target=_engine_worker_long, args=(out_path, quest_json_path))
    p.start()
    p.join()
    if p.exitcode != 0:
        raise RuntimeError(f"long engine subprocess exited {p.exitcode}")


def _run_engine_in_subprocess(out_path: str, quest_json_path: str | None) -> None:
    ctx = mp.get_context("spawn")
    p = ctx.Process(target=_engine_worker, args=(out_path, quest_json_path))
    p.start()
    p.join()
    if p.exitcode != 0:
        raise RuntimeError(
            f"engine subprocess exited {p.exitcode} "
            f"(quest={quest_json_path is not None})"
        )


@pytest.mark.slow_test
def test_quest_topk_full_matches_dense_fa(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    if os.environ.get("VLLM_QUEST_RUN_ALIGNMENT") != "1":
        pytest.skip("set VLLM_QUEST_RUN_ALIGNMENT=1 to run this slow test")

    from vllm.config.quest import QuestConfig

    # Dense reference engine.
    dense_out = tmp_path / "dense.json"
    _run_engine_in_subprocess(str(dense_out), None)
    out_dense = json.loads(dense_out.read_text())["text"]

    # Quest with top_k larger than any plausible block count for this prompt,
    # so selection degenerates to "all blocks" (the R1 invariant).
    cfg = QuestConfig(
        enabled=True,
        top_k=64,
        gpu_cache_blocks_per_seq=512,
        full_kv_layers=[0, 1],
        block_size=256,
        cpu_cache_blocks=8192,
        cpu_cache_gib=8,
        selection_impl="torch",
        enable_async_prefetch=False,
    )
    cfg.validate()
    quest_cfg_path = tmp_path / "quest_cfg.json"
    quest_cfg_path.write_text(json.dumps(cfg.to_dict()))
    quest_out = tmp_path / "quest.json"
    _run_engine_in_subprocess(str(quest_out), str(quest_cfg_path))
    out_quest = json.loads(quest_out.read_text())["text"]

    assert out_dense.strip() == out_quest.strip(), (
        f"dense={out_dense!r} quest={out_quest!r}"
    )


@pytest.mark.slow_test
def test_quest_offload_reload_lossless_small_vs_large_arena(tmp_path):
    """Engine-level offload round-trip proof. Two Quest engines, SAME top_k,
    DIFFERENT arena caps: a small cap that forces spill+reload vs a large cap
    that never spills. Equal greedy text proves the spill->CPU->reload round
    trip is lossless. This does NOT compare against dense (impossible under
    overflow: you cannot gather > cap blocks in one flash_attn call); the R1
    '== dense' invariant is covered by test_quest_topk_full_matches_dense_fa.
    Requires top_k <= small_cap - 1 (arena holds top_k selected + 1 live block)
    and a prompt long enough that the block count exceeds small_cap."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    if os.environ.get("VLLM_QUEST_RUN_ALIGNMENT") != "1":
        pytest.skip("set VLLM_QUEST_RUN_ALIGNMENT=1 to run this slow test")
    from vllm.config.quest import QuestConfig

    SMALL_CAP, TOP_K = 8, 6  # top_k <= cap-1 (6 <= 7); arena = 6 selected + 1 live

    def quest_text(cap):
        cfg = QuestConfig(enabled=True, top_k=TOP_K, gpu_cache_blocks_per_seq=cap,
                          full_kv_layers=[0, 1], block_size=256,
                          cpu_cache_blocks=8192, cpu_cache_gib=8,
                          selection_impl="torch", enable_async_prefetch=False)
        cfg.validate()
        p = tmp_path / f"cfg_{cap}.json"
        p.write_text(json.dumps(cfg.to_dict()))
        out = tmp_path / f"q_{cap}.json"
        _run_long_engine_in_subprocess(str(out), str(p))
        return json.loads(out.read_text())["text"]

    # Same top_k => same selection => equal text iff reload is lossless.
    small = quest_text(SMALL_CAP)
    big = quest_text(512)
    assert small.strip() == big.strip(), f"small={small!r} big={big!r}"


@pytest.mark.slow_test
def test_quest_write_through_offload_lossless_small_vs_large_arena(tmp_path):
    """Stage 2B: same as the offload round-trip proof above, but BOTH engines
    run with enable_write_through=True. A small cap forces eviction (now a
    GPU-slot drop, the host backup mirrored at fill) + H2D reload; equal greedy
    text vs a large cap that never spills proves the write-through
    fill-mirror -> drop -> reload round trip is lossless."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    if os.environ.get("VLLM_QUEST_RUN_ALIGNMENT") != "1":
        pytest.skip("set VLLM_QUEST_RUN_ALIGNMENT=1 to run this slow test")
    from vllm.config.quest import QuestConfig

    SMALL_CAP, TOP_K = 8, 6  # top_k <= cap-1 (6 <= 7); arena = 6 selected + 1 live

    def quest_text(cap):
        cfg = QuestConfig(enabled=True, top_k=TOP_K, gpu_cache_blocks_per_seq=cap,
                          full_kv_layers=[0, 1], block_size=256,
                          cpu_cache_blocks=8192, cpu_cache_gib=8,
                          selection_impl="torch", enable_async_prefetch=False,
                          enable_write_through=True)
        cfg.validate()
        p = tmp_path / f"wt_cfg_{cap}.json"
        p.write_text(json.dumps(cfg.to_dict()))
        out = tmp_path / f"wt_q_{cap}.json"
        _run_long_engine_in_subprocess(str(out), str(p))
        return json.loads(out.read_text())["text"]

    small = quest_text(SMALL_CAP)
    big = quest_text(512)
    assert small.strip() == big.strip(), (
        f"write-through small={small!r} big={big!r}"
    )


@pytest.mark.slow_test
def test_quest_write_through_lossless_vs_writeback(tmp_path):
    """Stage 2B primary correctness guard: two real engines, SAME top_k / cap /
    prompt, one enable_write_through=True and one False. Equal greedy text
    proves write-through is an equal-correctness alternative to the 2A
    write-back default (write-through == write-back == correct). The small cap
    + _LONG_PROMPT guarantees spill/eviction actually fires on both paths."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    if os.environ.get("VLLM_QUEST_RUN_ALIGNMENT") != "1":
        pytest.skip("set VLLM_QUEST_RUN_ALIGNMENT=1 to run this slow test")
    from vllm.config.quest import QuestConfig

    CAP, TOP_K = 8, 6  # top_k <= cap-1; small enough to force eviction

    def quest_text(write_through):
        cfg = QuestConfig(enabled=True, top_k=TOP_K, gpu_cache_blocks_per_seq=CAP,
                          full_kv_layers=[0, 1], block_size=256,
                          cpu_cache_blocks=8192, cpu_cache_gib=8,
                          selection_impl="torch", enable_async_prefetch=False,
                          enable_write_through=write_through)
        cfg.validate()
        tag = "wt" if write_through else "wb"
        p = tmp_path / f"vs_cfg_{tag}.json"
        p.write_text(json.dumps(cfg.to_dict()))
        out = tmp_path / f"vs_q_{tag}.json"
        _run_long_engine_in_subprocess(str(out), str(p))
        return json.loads(out.read_text())["text"]

    write_through = quest_text(True)
    write_back = quest_text(False)
    assert write_through.strip() == write_back.strip(), (
        f"write_through={write_through!r} write_back={write_back!r}"
    )

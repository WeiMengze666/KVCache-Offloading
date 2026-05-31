# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E.1 #2: Quest with top_k=ALL ≈ dense FA on first-step logprobs.

Short prompt (~6 tokens, < block_size=256) → the ``seq_too_short`` gate in
``QuestSparseOffloadImpl`` forces every layer to delegate to dense FA, so
"top_k=ALL" degenerates to "dense vs dense". First-step (step-0) logprobs come
from the **prefill** forward (full attention) and should match dense within
fp16 accumulation noise.

This used to be xfail because dense and Quest engines were built in the SAME
process; the upstream EngineCore CUDA-context leak OOM'd the second engine, so
the comparison ran against a half-built engine. We now build each engine in its
OWN spawned subprocess (same pattern as ``test_alignment_real_sparse.py``), and
— after the KV-write-contract fix (``forward_includes_kv_cache_update=False`` +
``do_kv_cache_update``) — the dense-delegation path is numerically correct, so
this is a real pass at cosine ≥ 0.999 (no longer xfail).
"""

from __future__ import annotations

import json
import math
import multiprocessing as mp
from pathlib import Path

import pytest

from .conftest import QUEST_E2E_MODEL_ID

pytestmark = pytest.mark.real_model

_PROMPT = "The capital of France is"

# Mirror conftest._LLM_SHARED_KWARGS (can't share a fixture across processes).
_SHARED_KWARGS = dict(
    dtype="float16",
    enforce_eager=True,
    max_model_len=1024,
    gpu_memory_utilization=0.50,
    enable_prefix_caching=False,
    enable_chunked_prefill=False,
    block_size=256,
)


def _logprob_dict_to_aligned_vectors(d_dense: dict, d_quest: dict):
    """Given two `{token_id: logprob}` maps, return matched fp64 vectors over
    the *intersection* of their keys, sorted by token id for determinism.
    Logprobs are in log-space; we convert to probabilities for cosine.
    """
    common = sorted(set(d_dense) & set(d_quest))
    if not common:
        raise AssertionError("no overlap between dense and quest top-N tokens")
    dense_v = [math.exp(d_dense[t]) for t in common]
    quest_v = [math.exp(d_quest[t]) for t in common]
    return dense_v, quest_v


def _cosine(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _engine_worker(out_path: str, quest_json_path: str | None) -> None:
    """Subprocess body: build one engine, generate one step, dump step-0
    logprobs as {str(token_id): logprob}. CUDA context dies with the process,
    sidestepping the same-process multi-engine OOM leak.
    """
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
    params = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20)
    out = llm.generate([_PROMPT], params, use_tqdm=False)[0]
    step0 = {str(tid): lp.logprob for tid, lp in out.outputs[0].logprobs[0].items()}
    Path(out_path).write_text(json.dumps(step0))


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


def test_quest_top_k_all_matches_dense_first_logprobs(tmp_path):
    from vllm.config.quest import QuestConfig

    dense_out = tmp_path / "dense_lp.json"
    _run_engine_in_subprocess(str(dense_out), None)
    dense_lp = json.loads(dense_out.read_text())

    # Baseline Quest config (top_k=64 ≥ any block count for a ~6-token prompt,
    # so selection is degenerate; the seq_too_short gate forces dense FA anyway).
    cfg = QuestConfig(
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
    quest_cfg_path = tmp_path / "quest_cfg.json"
    quest_cfg_path.write_text(json.dumps(cfg.to_dict()))
    quest_out = tmp_path / "quest_lp.json"
    _run_engine_in_subprocess(str(quest_out), str(quest_cfg_path))
    quest_lp = json.loads(quest_out.read_text())

    dense_v, quest_v = _logprob_dict_to_aligned_vectors(dense_lp, quest_lp)
    cos = _cosine(dense_v, quest_v)
    print(
        f"[cross-engine alignment] dense vs quest(seq_too_short→dense-FA) "
        f"first-step logprob cosine={cos:.6f} over {len(dense_v)} shared tokens"
    )
    assert cos >= 0.999, (
        f"dense vs quest first-step (prefill/dense-delegation) logprob "
        f"cosine={cos:.6f} < 0.999 over {len(dense_v)} shared tokens. The "
        f"short-prompt path delegates wholesale to dense FA, so this must "
        f"match; a regression here points at the KV-write contract."
    )

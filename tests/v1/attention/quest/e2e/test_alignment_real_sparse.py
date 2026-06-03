# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage 0 Item 1 — real-sparse alignment gate (Quest vs dense FA).

Unlike ``test_minimal_inference.py`` (top_k=ALL, short prompt → the
``seq_too_short`` gate forces dense delegation for every layer), this test
drives the **real sparse decode path**: prompts are long enough (~600 tokens
≥ block_size=256) to clear the ``seq_too_short`` gate, so ``QuestSparseOffloadImpl``
runs its genuine selection + sparse-``block_table`` gather + ``ensure_resident``
path rather than delegating to dense FA. We compare Quest-vs-dense **step-wise**
greedy logprob cosine.

Threshold (per roadmap §3.0 "真稀疏 gate"):
  - **Hard floor = 0.90.** Below this the build fails.
  - Recommended/target is HIGHER: 0.95, ideally 0.99. 0.90 is only the minimum
    "basically similar" bar, NOT the goal. A regression that drops cosine from
    ~0.99 to ~0.91 still passes here but should be investigated.

Note on sparsity under this config: with ``block_size=256``,
``max_model_len=1024`` and a ~600-token prompt, a sequence has at most ~3
fully-filled candidate blocks while ``top_k=64`` ≥ that — so every candidate
block is selected. This test therefore validates that the **sparse-gather code
path is numerically equivalent to dense FA** (the end-to-end R1 invariant) on
real model weights with the ``seq_too_short`` gate cleared.

Engine lifetime isolation: dense and Quest engines run in **separate spawned
subprocesses** (not co-resident, and not sequentially in this process). The
upstream EngineCore CUDA-context leak means ``del llm`` does NOT release the
~24 GiB a 0.50-utilization engine reserves, so a second engine in the same
process OOMs at startup (quest-e2e-howto §6.1). A subprocess that fully exits
releases the context cleanly. The roadmap sanctions "separate processes" for
exactly this.
"""

from __future__ import annotations

import json
import math
import multiprocessing as mp
from pathlib import Path

import pytest

# Reuse the model id + cosine helper (single source of truth).
from .conftest import QUEST_E2E_MODEL_ID
from .test_minimal_inference import _cosine

pytestmark = pytest.mark.real_model

# ~600+ real tokens (> block_size=256), multi-block so the seq_too_short gate
# is cleared and the sparse decode path engages.
_LONG_PROMPT = (
    "In the spring of the year 1789, the assembly convened in Versailles "
    "to address grievances that had accumulated over decades of fiscal "
    "mismanagement and shifting alliances among the nobility. "
) * 20

# Hard floor; recommended target is higher (see module docstring).
_COSINE_FLOOR = 0.90
_COSINE_RECOMMENDED = 0.99

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
    """Subprocess body: build one engine, generate, dump per-step logprobs.

    Runs in a spawned child so its CUDA context dies with the process. Writes
    a JSON list of {str(token_id): logprob} dicts (one per decode step).
    `quest_json_path` None → dense engine; else Quest via the EngineArgs path.
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
    params = SamplingParams(
        temperature=0.0, max_tokens=16, logprobs=20, seed=1234
    )
    out = llm.generate([_LONG_PROMPT], params, use_tqdm=False)[0]
    steps = [
        {str(tid): lp.logprob for tid, lp in step.items()}
        for step in out.outputs[0].logprobs
    ]
    Path(out_path).write_text(json.dumps(steps))


def _run_engine_in_subprocess(out_path: str, quest_json_path: str | None):
    ctx = mp.get_context("spawn")
    p = ctx.Process(target=_engine_worker, args=(out_path, quest_json_path))
    p.start()
    p.join()
    if p.exitcode != 0:
        raise RuntimeError(
            f"engine subprocess exited with code {p.exitcode} "
            f"(quest={quest_json_path is not None})"
        )


def _stepwise_cosines(dense_lp, quest_lp) -> list[float]:
    """Per-decode-step cosine over the intersection of top-N token ids.

    A step with ZERO shared token ids means the two engines disagree on the
    entire top-N — a total divergence. We report that as cosine 0.0 (rather
    than raising) so the hard-floor assertion below fires with a clear
    "below floor" message instead of a cryptic KeyError/empty-vector error.
    """
    steps = min(len(dense_lp), len(quest_lp))
    assert steps > 0, "no decode steps produced"
    cosines = []
    for i in range(steps):
        common = sorted(set(dense_lp[i]) & set(quest_lp[i]))
        if not common:
            cosines.append(0.0)
            continue
        dv = [math.exp(dense_lp[i][t]) for t in common]
        qv = [math.exp(quest_lp[i][t]) for t in common]
        cosines.append(_cosine(dv, qv))
    return cosines


def test_real_sparse_decode_matches_dense_stepwise_cosine(tmp_path):
    from vllm.config.quest import QuestConfig

    # Dense engine in its own process.
    dense_out = tmp_path / "dense_lp.json"
    _run_engine_in_subprocess(str(dense_out), None)
    dense_lp = json.loads(dense_out.read_text())

    # Quest engine in its own process, via the official EngineArgs JSON path.
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

    assert dense_lp and quest_lp, "engine returned no per-step logprobs"

    cosines = _stepwise_cosines(dense_lp, quest_lp)
    min_cos = min(cosines)
    mean_cos = sum(cosines) / len(cosines)

    # Measurement-first logging: this is the headline quality metric.
    print(
        f"[real-sparse alignment] steps={len(cosines)} "
        f"min_cosine={min_cos:.6f} mean_cosine={mean_cos:.6f} "
        f"(floor={_COSINE_FLOOR}, recommended>={_COSINE_RECOMMENDED}) "
        f"per_step={[round(c, 5) for c in cosines]}"
    )
    if min_cos < _COSINE_RECOMMENDED:
        print(
            f"[real-sparse alignment] NOTE: min cosine {min_cos:.6f} is below "
            f"the recommended {_COSINE_RECOMMENDED}; passes the {_COSINE_FLOOR} "
            f"floor but worth investigating if this is a regression."
        )

    # Hard gate: every step must clear the floor.
    assert min_cos >= _COSINE_FLOOR, (
        f"real-sparse Quest-vs-dense step-wise logprob cosine min={min_cos:.6f} "
        f"< hard floor {_COSINE_FLOOR} over {len(cosines)} steps "
        f"({[round(c, 5) for c in cosines]}). This is a real sparse-vs-dense "
        f"divergence, not noise — do NOT loosen the floor; investigate."
    )

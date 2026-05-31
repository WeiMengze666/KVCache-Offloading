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

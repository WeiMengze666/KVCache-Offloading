# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""R1 alignment: ArkVale (top_k = ALL) vs dense FA on a real model.

Spec appendix A.7 (ArkVale variant): with top_k larger than the prompt's block
count, ArkVale selection degenerates to "pick every block", so the sparse decode
path should be numerically equivalent to dense FlashAttention (the R1 invariant).
We assert the greedy text matches.

The digest_mode does not affect correctness here — at top_k=ALL every block is
selected regardless of which formula produced the digest scores, so the sparse
compute over all blocks is bitwise-equal to dense FA output.

Gate: set VLLM_ARKVALE_RUN_ALIGNMENT=1 to run this slow test (mirrors the
Quest gate VLLM_QUEST_RUN_ALIGNMENT=1 in test_alignment_vllm_dense_fa.py).
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
from pathlib import Path

import pytest
import torch

# Same model the rest of the quest/arkvale e2e suite uses (cached locally).
ARKVALE_E2E_MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"

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


def _engine_worker(out_path: str, arkvale_json_path: str | None) -> None:
    """Build one engine in a spawned subprocess, generate greedily, dump text."""
    from vllm import LLM, SamplingParams

    if arkvale_json_path is None:
        llm = LLM(model=ARKVALE_E2E_MODEL_ID, **_SHARED_KWARGS)
    else:
        llm = LLM(
            model=ARKVALE_E2E_MODEL_ID,
            enable_arkvale_sparse_offload=True,
            arkvale_config=arkvale_json_path,
            **_SHARED_KWARGS,
        )
    sp = SamplingParams(max_tokens=8, temperature=0.0)
    text = llm.generate([_PROMPT], sp, use_tqdm=False)[0].outputs[0].text
    Path(out_path).write_text(json.dumps({"text": text}))


def _run_engine_in_subprocess(out_path: str, arkvale_json_path: str | None) -> None:
    ctx = mp.get_context("spawn")
    p = ctx.Process(target=_engine_worker, args=(out_path, arkvale_json_path))
    p.start()
    p.join()
    if p.exitcode != 0:
        raise RuntimeError(
            f"engine subprocess exited {p.exitcode} "
            f"(arkvale={arkvale_json_path is not None})"
        )


@pytest.mark.slow_test
def test_arkvale_topk_full_matches_dense_fa(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    if os.environ.get("VLLM_ARKVALE_RUN_ALIGNMENT") != "1":
        pytest.skip("set VLLM_ARKVALE_RUN_ALIGNMENT=1 to run this slow test")

    from vllm.config.arkvale import ArkValeConfig

    # Dense reference engine.
    dense_out = tmp_path / "dense.json"
    _run_engine_in_subprocess(str(dense_out), None)
    out_dense = json.loads(dense_out.read_text())["text"]

    # ArkVale with top_k larger than any plausible block count for this prompt,
    # so selection degenerates to "all blocks" (the R1 invariant).
    # The prompt is ~600 tokens; with block_size=256 that is ~3 blocks.
    # top_k=64 >> 3, so every block is always selected.
    cfg = ArkValeConfig(
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
    arkvale_cfg_path = tmp_path / "arkvale_cfg.json"
    arkvale_cfg_path.write_text(json.dumps(cfg.to_dict()))
    arkvale_out = tmp_path / "arkvale.json"
    _run_engine_in_subprocess(str(arkvale_out), str(arkvale_cfg_path))
    out_arkvale = json.loads(arkvale_out.read_text())["text"]

    assert out_dense.strip() == out_arkvale.strip(), (
        f"dense={out_dense!r} arkvale={out_arkvale!r}"
    )

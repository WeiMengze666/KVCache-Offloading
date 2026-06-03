# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage 2C-v2 Task A4: kv-share footprint lever — engine-level validation.

Gated (VLLM_QUEST_RUN_ALIGNMENT=1, real GPU, downloads/loads Llama-3.2-3B).
Each engine runs in a spawned subprocess (EngineCore CUDA-context isolation).

Stage A proves only the CHANNEL: with footprint_kvshare=True the non-full-KV
Quest layers leave HMA (allocate zero blocks → "GPU KV cache size" jumps) and
forward dispatches to our impl without KeyError. OUTPUT CORRECTNESS IS NOT
ASSERTED HERE — the KV write path is rewired in Stage B. We assert:
  - the engine boots and generates without raising,
  - the reported "GPU KV cache size" (tokens) is strictly larger with
    footprint_kvshare on than off, at the SAME util (the footprint lever).
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
from pathlib import Path

import pytest
import torch

QUEST_E2E_MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"

_PROMPT = (
    "The Eiffel Tower was completed in 1889 for the World's Fair held in "
    "Paris to celebrate the centennial of the French Revolution. "
) * 8 + "\nQ: When was it completed?\nA:"

_KWARGS = dict(
    dtype="float16",
    enforce_eager=True,
    max_model_len=1024,
    max_num_seqs=1,
    gpu_memory_utilization=0.50,
    enable_prefix_caching=False,
    enable_chunked_prefill=False,
    block_size=256,
)


def _base_cfg(footprint_kvshare: bool) -> dict:
    from vllm.config.quest import QuestConfig

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
        footprint_kvshare=footprint_kvshare,
    )
    cfg.validate()
    return cfg.to_dict()


def _engine_worker(out_path: str, quest_json_path: str) -> None:
    """Boot one Quest engine, generate greedily, dump
    {text, num_gpu_blocks}. num_gpu_blocks is read programmatically from the
    resolved cache_config (robust vs scraping the KV-size log line, which is
    emitted by the separate EngineCore subprocess)."""
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=QUEST_E2E_MODEL_ID,
        enable_quest_sparse_offload=True,
        quest_config=quest_json_path,
        **_KWARGS,
    )
    num_gpu_blocks = int(
        llm.llm_engine.vllm_config.cache_config.num_gpu_blocks or 0
    )
    sp = SamplingParams(max_tokens=8, temperature=0.0)
    out = llm.generate([_PROMPT], sp, use_tqdm=False)
    text = out[0].outputs[0].text
    Path(out_path).write_text(
        json.dumps({"text": text, "num_gpu_blocks": num_gpu_blocks})
    )


def _run(out_path: str, cfg_path: str) -> dict:
    ctx = mp.get_context("spawn")
    p = ctx.Process(target=_engine_worker, args=(out_path, cfg_path))
    p.start()
    p.join()
    if p.exitcode != 0:
        raise RuntimeError(f"engine subprocess exited {p.exitcode}")
    return json.loads(Path(out_path).read_text())


@pytest.mark.slow_test
def test_footprint_kvshare_rebounds_kv_cache_size(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    if os.environ.get("VLLM_QUEST_RUN_ALIGNMENT") != "1":
        pytest.skip("set VLLM_QUEST_RUN_ALIGNMENT=1 to run this slow test")

    def blocks(footprint_kvshare: bool) -> int:
        tag = "kvshare" if footprint_kvshare else "hma"
        cfg = tmp_path / f"cfg_{tag}.json"
        cfg.write_text(json.dumps(_base_cfg(footprint_kvshare)))
        res = _run(str(tmp_path / f"out_{tag}.json"), str(cfg))
        assert res["text"] is not None  # booted + generated, no KeyError
        assert res["num_gpu_blocks"] > 0, f"no num_gpu_blocks ({tag})"
        return res["num_gpu_blocks"]

    on = blocks(True)
    off = blocks(False)
    # Footprint lever: routing Quest layers out of HMA rebounds num_blocks, so
    # the same util now backs many more tokens (spike: 1,339 -> 34,302).
    assert on > off, (
        f"footprint_kvshare did not rebound num_gpu_blocks: "
        f"on={on} vs off={off} (expected on >> off)"
    )

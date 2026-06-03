#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""End-to-end sparse-offload smoke harness for vLLM on Llama-3.2-3B-Instruct.

Runs five variants in subprocess-isolated spawned children (the v1 EngineCore
leaks CUDA context across LLM(...) constructions in the same process — file
isolation is the documented workaround):

    dense      : FlashAttention reference (no Quest, no ArkVale)
    quest_wb   : Quest 2A write-back            (default offload path)
    quest_wt   : Quest 2B write-through         (enable_write_through=True)
    quest_kv   : Quest 2C-v2 footprint kv-share (footprint_kvshare=True)
    arkvale    : ArkVale cuboid_mean digest     (digest_mode=arkvale_cuboid_mean)

For each non-dense variant we (1) generate greedy text on three long prompts,
(2) probe per-layer TierManager stats via LLMEngine.apply_model, and (3) write
a JSON record under /tmp/quest_e2e_smoke/<variant>.json. The driver compares
each variant's outputs against dense and reports whether the sparse path and
the offload (evict/load) machinery actually engaged.

USAGE (must run from the repo root with the project venv on PATH):

    cd /home/yijun/offload_attn/KVCache-Offloading
    HF_HUB_OFFLINE=1 .venv/bin/python run_sparse_offload_e2e.py

The script self-prepends ``./.venv/bin`` to PATH so flashinfer's JIT compiler
(``ninja``) is found inside the spawned engine-core subprocess.

The Llama-3.2-3B snapshot must already be in your local HuggingFace cache
(this script does not download). Typical full-run time on a single GPU:
~4-6 min (the 3B weights are loaded once per variant).
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
from dataclasses import asdict
from pathlib import Path

# Make the venv's binaries (notably `ninja`, used by flashinfer's JIT path
# in topk_topp_sampler) discoverable from the spawned engine-core children
# regardless of how the user invoked the script. Without this, a `python`
# invocation from a shell that doesn't have .venv/bin on PATH will load
# the engine, then crash inside flashinfer with FileNotFoundError: 'ninja'.
_VENV_BIN = Path(__file__).resolve().parent / ".venv" / "bin"
if _VENV_BIN.is_dir():
    cur = os.environ.get("PATH", "")
    if str(_VENV_BIN) not in cur.split(os.pathsep):
        os.environ["PATH"] = f"{_VENV_BIN}{os.pathsep}{cur}"

MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
OUT_DIR = Path("/tmp/quest_e2e_smoke")

# Three long prompts, each big enough to overflow the 8-block GPU pool
# (block_size=256 → 2048 prompt tokens cap). Forces evict_d2h / load_h2d
# traffic so the offload paths are exercised, not bypassed.
PROMPTS = [
    "In the spring of 1789, the assembly convened in Versailles to address "
    "grievances that had accumulated over decades of fiscal mismanagement "
    "and shifting alliances among the nobility. " * 90,
    "The Roman Empire reached its greatest territorial extent under Trajan "
    "in 117 AD, spanning from Britain in the northwest to Mesopotamia in "
    "the southeast. Roman engineering produced aqueducts, paved roads, and "
    "concrete structures whose remains still stand. " * 50,
    "Modern transformer language models compute attention scores by taking "
    "the dot product of query and key vectors, dividing by the square root "
    "of the head dimension, applying softmax, and using the resulting "
    "weights to combine value vectors. Sparse attention exploits the "
    "observation that for many real prompts a small subset of past tokens "
    "dominates the attention output. " * 50,
]

# Engine kwargs shared across every variant. block_size=256 +
# gpu_memory_utilization=0.50 + max_model_len=4096 are picked so prompts
# above ~2048 tokens overflow the 8-block GPU pool and force eviction.
# gpu_memory_utilization is passed in via CLI (see main()), not hard-coded
# here, so a single run can target a tight or generous KV reservation.
SHARED_LLM_KWARGS = dict(
    dtype="float16",
    enforce_eager=True,
    max_model_len=4096,
    enable_prefix_caching=False,
    enable_chunked_prefill=False,
    block_size=256,
    max_num_seqs=1,
)

# Common Quest/ArkVale knobs. top_k=4 ≤ gpu_cache_blocks_per_seq=8 leaves
# headroom for the working set; cpu_cache_blocks=8192 + cpu_cache_gib=8
# matches the conftest baseline.
SPARSE_CFG_BASE = dict(
    enabled=True,
    block_size=256,
    top_k=4,
    full_kv_layers=[0, 1],
    gpu_cache_blocks_per_seq=8,
    cpu_cache_blocks=8192,
    cpu_cache_gib=8,
    selection_impl="torch",
    enable_async_prefetch=False,
)


def _build_cfg(variant: str) -> dict:
    cfg = dict(SPARSE_CFG_BASE)
    if variant == "quest_wb":
        cfg.update(enable_write_through=False, footprint_kvshare=False)
    elif variant == "quest_wt":
        cfg.update(enable_write_through=True, footprint_kvshare=False)
    elif variant == "quest_kv":
        cfg.update(enable_write_through=False, footprint_kvshare=True)
    elif variant == "arkvale":
        # ArkValeConfig has no write_through/kvshare fields; instead the
        # digest formula switches to cuboid_mean. Same backend, same offload
        # stack — only the page-digest math differs.
        cfg.update(digest_mode="arkvale_cuboid_mean")
    else:
        raise ValueError(variant)
    return cfg


def _probe_sparse_layers(model):
    """Run inside the engine-core worker via collective_rpc.

    Returns one record per Attention module that has a tier_manager attached.
    Empty list ⇒ the sparse path didn't engage (everything fell through to
    the dense kernel, e.g. because full_kv_layers covered every layer).
    """
    out = []
    for name, mod in model.named_modules():
        if type(mod).__name__ != "Attention":
            continue
        tm = getattr(mod, "tier_manager", None)
        if tm is None:
            continue
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


def _engine_worker(
    out_path: str, variant: str, cfg_json_path: str, gpu_mem_util: float,
) -> None:
    """Subprocess body — one engine, generate over PROMPTS, probe stats."""
    # apply_model ships a pickled callable across the engine-core IPC
    # boundary. v1 requires opt-in.
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

    from vllm import LLM, SamplingParams

    common = dict(SHARED_LLM_KWARGS, gpu_memory_utilization=gpu_mem_util)
    if variant == "dense":
        llm = LLM(model=MODEL_ID, **common)
    elif variant == "arkvale":
        llm = LLM(
            model=MODEL_ID,
            enable_arkvale_sparse_offload=True,
            arkvale_config=cfg_json_path,
            **common,
        )
    else:  # quest_wb / quest_wt / quest_kv
        llm = LLM(
            model=MODEL_ID,
            enable_quest_sparse_offload=True,
            quest_config=cfg_json_path,
            **common,
        )

    params = SamplingParams(temperature=0.0, max_tokens=32, seed=1234)

    outputs = []
    for prompt in PROMPTS:
        out = llm.generate([prompt], params, use_tqdm=False)[0]
        outputs.append(
            {
                "prompt_len_chars": len(prompt),
                "prompt_token_count": len(out.prompt_token_ids),
                "generated_token_ids": list(out.outputs[0].token_ids),
                "generated_text": out.outputs[0].text,
            }
        )

    layer_stats: list = []
    if variant != "dense":
        for rank_result in llm.llm_engine.apply_model(_probe_sparse_layers):
            layer_stats.extend(rank_result)

    Path(out_path).write_text(
        json.dumps(
            {"variant": variant, "outputs": outputs, "layer_stats": layer_stats},
            indent=2,
        )
    )


def _run_in_subprocess(
    out_path: str, variant: str, cfg_json_path: str, gpu_mem_util: float,
) -> int:
    ctx = mp.get_context("spawn")
    p = ctx.Process(
        target=_engine_worker,
        args=(out_path, variant, cfg_json_path, gpu_mem_util),
    )
    p.start()
    p.join()
    return p.exitcode if p.exitcode is not None else -1


def _aggregate(layer_stats: list) -> dict:
    keys = (
        "select_calls",
        "selected_total",
        "selected_on_gpu",
        "load_h2d",
        "evict_d2h",
        "evict_drop",
        "block_filled",
    )
    agg = {k: 0 for k in keys}
    agg["num_quest_layers"] = 0
    for rec in layer_stats:
        s = rec.get("stats")
        if s is None:
            continue
        agg["num_quest_layers"] += 1
        for k in keys:
            agg[k] += int(s.get(k, 0))
    return agg


def _write_cfg_json(variant: str) -> str:
    cfg = _build_cfg(variant)
    p = OUT_DIR / f"{variant}_cfg.json"
    p.write_text(json.dumps(cfg))
    return str(p)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="End-to-end sparse-offload smoke harness for vLLM on "
        "Llama-3.2-3B-Instruct.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.50,
        help="Fraction of total GPU memory the engine may reserve for "
        "weights + KV cache. Lower this if other processes share the GPU "
        "or to verify Stage 2C-v2's footprint reduction (kvshare lets you "
        "drop it without losing capacity). Default: 0.50.",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    variants = ("dense", "quest_wb", "quest_wt", "quest_kv", "arkvale")
    records: dict[str, dict] = {}
    failures: list[str] = []

    for v in variants:
        out_path = str(OUT_DIR / f"{v}.json")
        cfg_path = "" if v == "dense" else _write_cfg_json(v)
        print(f"[run] variant={v} gpu_memory_utilization={args.gpu_memory_utilization}")
        rc = _run_in_subprocess(
            out_path, v, cfg_path, args.gpu_memory_utilization,
        )
        if rc != 0:
            print(f"  subprocess failed (rc={rc}); see logs")
            failures.append(v)
            continue
        records[v] = json.loads(Path(out_path).read_text())
        print(f"  wrote {out_path}")

    print("\n=== output comparison vs dense ===")
    if "dense" in records:
        dense_outs = [o["generated_text"] for o in records["dense"]["outputs"]]
        for v in variants:
            if v == "dense" or v not in records:
                continue
            qouts = [o["generated_text"] for o in records[v]["outputs"]]
            matches = [d == q for d, q in zip(dense_outs, qouts)]
            print(
                f"  {v:<10} text-vs-dense: "
                f"{sum(matches)}/{len(matches)} prompts identical"
            )

    print("\n=== sparse / offload engagement ===")
    for v in variants:
        if v == "dense" or v not in records:
            continue
        agg = _aggregate(records[v]["layer_stats"])
        sparse_engaged = (
            agg["num_quest_layers"] > 0
            and agg["select_calls"] > 0
            and agg["selected_total"] > 0
        )
        offload_engaged = (
            agg["evict_d2h"] + agg["evict_drop"] + agg["load_h2d"]
        ) > 0
        print(
            f"  {v:<10} sparse_engaged={sparse_engaged} "
            f"offload_traffic={offload_engaged} "
            f"(evict_d2h={agg['evict_d2h']}, "
            f"evict_drop={agg['evict_drop']}, "
            f"load_h2d={agg['load_h2d']}, "
            f"block_filled={agg['block_filled']})"
        )

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

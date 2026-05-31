# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage 0 Item 3 — Quest sparse-offload benchmark harness.

Standalone script (vLLM-official benchmark style). Measures Quest vs dense FA
and offload-on vs offload-off, reporting quality / memory / speed / behavior
per config point to JSON + CSV.

Two comparison groups (both produced by a single run):
  1. Quest vs dense FA           — same model/prompts, Quest disabled vs enabled.
  2. offload on vs off           — same Quest config except gpu_cache_blocks_per_seq
                                   large (no overflow) vs small (forced overflow).

CRITICAL design note (see roadmap §3.2): with the default
gpu_cache_blocks_per_seq=512 and block_size=256 a sequence needs >131072 tokens
to overflow the GPU working set, so offload NEVER triggers. This harness shrinks
gpu_cache_blocks_per_seq (default 8) so a ~1-2k-token prompt produces several
times more blocks than the GPU budget, forcing real D2H eviction + H2D reload.
The "offload" config point ASSERTS evict_d2h>0 AND load_h2d>0; if not, the run
errors out (it would otherwise silently measure the no-offload path).

Engine isolation: each engine (dense and each Quest variant) is built and run in
its OWN spawned subprocess, so the upstream EngineCore CUDA-context leak (which
makes `del llm` fail to release ~24 GiB) cannot accumulate and OOM a single 4090.
GPU-internal H2D-wait / eviction-stall timing uses torch.cuda.Event inside
TierManager, gated by QuestConfig.enable_debug_counters (zero-cost otherwise).

Usage:
  export PATH=/usr/local/cuda-12.8/bin:$PATH CUDA_HOME=/usr/local/cuda-12.8
  export HF_HUB_OFFLINE=1   # if model already cached
  .venv/bin/python benchmarks/benchmark_quest.py --output-dir /tmp/quest_bench

Run with --help for all knobs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Model / prompt defaults (roadmap: Llama-3.2-3B-Instruct, temperature=0, seeded).
DEFAULT_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
DEFAULT_SEED = 1234
DEFAULT_BLOCK_SIZE = 256

# A ~1-2k token prompt: repeated paragraph, long enough that with block_size=256
# it spans several blocks (>> a small gpu_cache_blocks_per_seq budget).
_PARAGRAPH = (
    "In the spring of the year 1789, the assembly convened in Versailles "
    "to address grievances that had accumulated over decades of fiscal "
    "mismanagement and shifting alliances among the nobility. The delegates "
    "argued late into the night about representation, taxation, and the "
    "ancient privileges of the clergy and the aristocracy. "
)
# PLACEHOLDER_BODY


def build_prompt(num_paragraphs: int) -> str:
    return (_PARAGRAPH * num_paragraphs).strip()


# ----------------------------------------------------------------------------
# Config describing one engine run (serializable across the process boundary).
# ----------------------------------------------------------------------------
@dataclass
class RunConfig:
    name: str                       # human label, e.g. "dense" / "quest_offload"
    quest_enabled: bool
    model: str = DEFAULT_MODEL
    block_size: int = DEFAULT_BLOCK_SIZE
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.55
    num_paragraphs: int = 6         # prompt length knob (~ a few hundred tokens each)
    max_tokens: int = 64
    seed: int = DEFAULT_SEED
    # Quest-only knobs (ignored when quest_enabled is False).
    top_k: int = 64
    gpu_cache_blocks_per_seq: int = 512
    cpu_cache_blocks: int = 8192
    cpu_cache_gib: int = 8
    selection_impl: str = "torch"
    full_kv_layers: list[int] = field(default_factory=lambda: [0, 1])
    # Expect offload to trigger (asserted in-worker for the small-budget point).
    expect_offload: bool = False


# ----------------------------------------------------------------------------
# In-worker probe: aggregate per-Quest-layer stats across the engine-core
# process boundary, mirroring tests/.../e2e/test_prefill_decode_mixed.py.
# ----------------------------------------------------------------------------
def _probe_quest_layers(model):
    """Runs inside the engine-core worker process via collective_rpc."""
    from dataclasses import asdict as _asdict

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
                "stats": _asdict(s) if s is not None else None,
            }
        )
    return out


def _aggregate_stats(per_layer: list[dict]) -> dict[str, float]:
    """Sum the per-layer QuestStats dicts into a flat aggregate."""
    keys = [
        "block_filled", "evict_d2h", "load_h2d", "select_calls",
        "selected_total", "selected_on_gpu",
        "h2d_wait_ms", "evict_stall_ms",
        "h2d_wait_events", "evict_stall_events",
    ]
    agg = {k: 0 for k in keys}
    agg["num_quest_layers"] = 0
    for entry in per_layer:
        s = entry.get("stats")
        if not s:
            continue
        agg["num_quest_layers"] += 1
        for k in keys:
            agg[k] += s.get(k, 0) or 0
    # Derived rates.
    sel_total = agg["selected_total"]
    agg["gpu_hit_rate"] = (
        agg["selected_on_gpu"] / sel_total if sel_total else 0.0
    )
    agg["mean_h2d_wait_ms"] = (
        agg["h2d_wait_ms"] / agg["h2d_wait_events"]
        if agg["h2d_wait_events"] else 0.0
    )
    agg["mean_evict_stall_ms"] = (
        agg["evict_stall_ms"] / agg["evict_stall_events"]
        if agg["evict_stall_events"] else 0.0
    )
    return agg
# PLACEHOLDER_WORKER


def _engine_worker(cfg_dict: dict, tmp_dir: str, out_path: str) -> None:
    """Subprocess body: build ONE engine, run generation, dump metrics+logprobs.

    Runs in a spawned child so its CUDA context dies with the process (avoids
    the upstream EngineCore leak accumulating across engines). Writes a JSON
    result dict to out_path.
    """
    import torch

    from vllm import LLM, SamplingParams

    # apply_model ships a Python callable to the engine-core worker; v1 IPC
    # rejects non-msgspec types unless this opt-in is set. Set it here too
    # (belt-and-suspenders alongside main()) so the engine-core grandchild
    # process inherits it regardless of how this worker was launched.
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

    cfg = RunConfig(**cfg_dict)
    prompt = build_prompt(cfg.num_paragraphs)

    shared = dict(
        model=cfg.model,
        dtype="float16",
        enforce_eager=True,
        max_model_len=cfg.max_model_len,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        block_size=cfg.block_size,
        seed=cfg.seed,
    )

    if cfg.quest_enabled:
        from vllm.config.quest import QuestConfig

        qc = QuestConfig(
            enabled=True,
            block_size=cfg.block_size,
            top_k=cfg.top_k,
            full_kv_layers=list(cfg.full_kv_layers),
            gpu_cache_blocks_per_seq=cfg.gpu_cache_blocks_per_seq,
            cpu_cache_blocks=cfg.cpu_cache_blocks,
            cpu_cache_gib=cfg.cpu_cache_gib,
            selection_impl=cfg.selection_impl,
            enable_async_prefetch=False,
            enable_debug_counters=True,   # turns on cuda-Event timing hooks
        )
        qc.validate()
        json_path = Path(tmp_dir) / f"quest_cfg_{cfg.name}.json"
        json_path.write_text(json.dumps(qc.to_dict()))
        llm = LLM(
            model=cfg.model,
            enable_quest_sparse_offload=True,
            quest_config=str(json_path),
            **{k: v for k, v in shared.items() if k != "model"},
        )
    else:
        llm = LLM(**shared)

    params = SamplingParams(
        temperature=0.0, max_tokens=cfg.max_tokens, logprobs=20, seed=cfg.seed,
    )

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    outputs = llm.generate([prompt], params, use_tqdm=False)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    out = outputs[0]
    gen_tokens = len(out.outputs[0].token_ids)
    # Per-step logprobs as {str(token_id): logprob} (survives serialization).
    step_logprobs = [
        {str(tid): lp.logprob for tid, lp in step.items()}
        for step in (out.outputs[0].logprobs or [])
    ]
    peak_gpu_bytes = int(torch.cuda.max_memory_allocated())

    result: dict[str, Any] = {
        "name": cfg.name,
        "config": cfg_dict,
        "prompt_chars": len(prompt),
        "prompt_tokens": len(out.prompt_token_ids),
        "gen_tokens": gen_tokens,
        "latency_s": elapsed,
        "decode_tokens_per_s": gen_tokens / elapsed if elapsed else 0.0,
        "peak_gpu_alloc_bytes": peak_gpu_bytes,
        "peak_gpu_alloc_gib": peak_gpu_bytes / (1024 ** 3),
        "output_text": out.outputs[0].text,
        "step_logprobs": step_logprobs,
    }

    if cfg.quest_enabled:
        per_layer = llm.llm_engine.apply_model(_probe_quest_layers)
        flat = []
        for rank in per_layer:
            flat.extend(rank)
        agg = _aggregate_stats(flat)
        result["quest_stats"] = agg
        # Acceptance gate: a "must offload" point that didn't offload is a
        # silent no-op measuring the wrong path. Fail loudly in-worker.
        if cfg.expect_offload:
            if not (agg["evict_d2h"] > 0 and agg["load_h2d"] > 0):
                raise RuntimeError(
                    f"[{cfg.name}] expected offload but evict_d2h="
                    f"{agg['evict_d2h']} load_h2d={agg['load_h2d']}. "
                    f"gpu_cache_blocks_per_seq="
                    f"{cfg.gpu_cache_blocks_per_seq} too large for prompt "
                    f"({result['prompt_tokens']} tokens, block_size="
                    f"{cfg.block_size}) — shrink the budget."
                )

    Path(out_path).write_text(json.dumps(result))
# PLACEHOLDER_DRIVER


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def stepwise_cosine_vs_dense(dense_lp, other_lp) -> dict[str, float]:
    """Mean/min per-step logprob cosine between two engines' outputs.

    Zero top-N overlap on a step counts as cosine 0.0 (total divergence)
    rather than being skipped, so a collapse is visible in the metric.
    """
    steps = min(len(dense_lp), len(other_lp))
    if steps == 0:
        return {"mean": 0.0, "min": 0.0, "steps": 0}
    cosines = []
    for i in range(steps):
        common = sorted(set(dense_lp[i]) & set(other_lp[i]))
        if not common:
            cosines.append(0.0)
            continue
        dv = [math.exp(dense_lp[i][t]) for t in common]
        ov = [math.exp(other_lp[i][t]) for t in common]
        cosines.append(_cosine(dv, ov))
    return {
        "mean": sum(cosines) / len(cosines),
        "min": min(cosines),
        "steps": steps,
        "per_step": [round(c, 5) for c in cosines],
    }


def run_one(cfg: RunConfig, tmp_dir: str) -> dict[str, Any]:
    """Run a single config in its own spawned subprocess; return its result."""
    out_path = Path(tmp_dir) / f"result_{cfg.name}.json"
    ctx = mp.get_context("spawn")
    p = ctx.Process(
        target=_engine_worker, args=(asdict(cfg), tmp_dir, str(out_path)),
    )
    p.start()
    p.join()
    if p.exitcode != 0:
        raise RuntimeError(
            f"engine subprocess for '{cfg.name}' exited with code {p.exitcode}"
        )
    return json.loads(out_path.read_text())


def write_outputs(records: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    # Full JSON (includes per-step logprobs for later re-analysis).
    (output_dir / "quest_benchmark.json").write_text(json.dumps(records, indent=2))

    # Flat CSV (one row per config point; drop the bulky logprob arrays).
    csv_cols = [
        "name", "quest_enabled", "prompt_tokens", "gen_tokens", "latency_s",
        "decode_tokens_per_s", "peak_gpu_alloc_gib",
        "cosine_vs_dense_mean", "cosine_vs_dense_min",
        "gpu_cache_blocks_per_seq", "top_k",
        "block_filled", "evict_d2h", "load_h2d",
        "selected_total", "selected_on_gpu", "gpu_hit_rate",
        "mean_h2d_wait_ms", "mean_evict_stall_ms",
    ]
    with (output_dir / "quest_benchmark.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=csv_cols)
        w.writeheader()
        for r in records:
            qs = r.get("quest_stats", {}) or {}
            cfg = r["config"]
            w.writerow({
                "name": r["name"],
                "quest_enabled": cfg["quest_enabled"],
                "prompt_tokens": r["prompt_tokens"],
                "gen_tokens": r["gen_tokens"],
                "latency_s": round(r["latency_s"], 4),
                "decode_tokens_per_s": round(r["decode_tokens_per_s"], 2),
                "peak_gpu_alloc_gib": round(r["peak_gpu_alloc_gib"], 3),
                "cosine_vs_dense_mean": round(r.get("cosine_vs_dense_mean", 1.0), 5),
                "cosine_vs_dense_min": round(r.get("cosine_vs_dense_min", 1.0), 5),
                "gpu_cache_blocks_per_seq": cfg["gpu_cache_blocks_per_seq"],
                "top_k": cfg["top_k"],
                "block_filled": qs.get("block_filled", 0),
                "evict_d2h": qs.get("evict_d2h", 0),
                "load_h2d": qs.get("load_h2d", 0),
                "selected_total": qs.get("selected_total", 0),
                "selected_on_gpu": qs.get("selected_on_gpu", 0),
                "gpu_hit_rate": round(qs.get("gpu_hit_rate", 0.0), 4),
                "mean_h2d_wait_ms": round(qs.get("mean_h2d_wait_ms", 0.0), 4),
                "mean_evict_stall_ms": round(qs.get("mean_evict_stall_ms", 0.0), 4),
            })


def print_summary(records: list[dict]) -> None:
    print("\n" + "=" * 78)
    print("QUEST BENCHMARK SUMMARY")
    print("=" * 78)
    for r in records:
        cfg = r["config"]
        qs = r.get("quest_stats", {}) or {}
        print(f"\n[{r['name']}] quest={cfg['quest_enabled']} "
              f"gpu_blocks/seq={cfg['gpu_cache_blocks_per_seq']} top_k={cfg['top_k']}")
        print(f"  prompt={r['prompt_tokens']}tok gen={r['gen_tokens']}tok "
              f"latency={r['latency_s']:.3f}s "
              f"decode={r['decode_tokens_per_s']:.1f}tok/s "
              f"peak_gpu={r['peak_gpu_alloc_gib']:.2f}GiB")
        if "cosine_vs_dense_mean" in r:
            print(f"  cosine_vs_dense: mean={r['cosine_vs_dense_mean']:.5f} "
                  f"min={r['cosine_vs_dense_min']:.5f}")
        if qs:
            print(f"  behavior: block_filled={qs['block_filled']} "
                  f"evict_d2h={qs['evict_d2h']} load_h2d={qs['load_h2d']} "
                  f"hit_rate={qs['gpu_hit_rate']:.3f} "
                  f"({qs['selected_on_gpu']}/{qs['selected_total']})")
            print(f"  gpu-timing: mean_h2d_wait={qs['mean_h2d_wait_ms']:.3f}ms "
                  f"mean_evict_stall={qs['mean_evict_stall_ms']:.3f}ms")
    print("=" * 78 + "\n")
# PLACEHOLDER_MAIN


def build_run_plan(args) -> list[RunConfig]:
    """Construct the config points for both comparison groups.

    - dense:          baseline (Quest disabled).
    - quest_no_off:   Quest enabled, LARGE gpu budget -> no overflow (offload off).
    - quest_offload:  Quest enabled, SMALL gpu budget -> forced overflow (offload on).

    Group 1 (Quest vs dense) = dense vs {quest_no_off, quest_offload}.
    Group 2 (offload on/off)  = quest_no_off vs quest_offload.
    """
    common = dict(
        model=args.model,
        block_size=args.block_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        num_paragraphs=args.num_paragraphs,
        max_tokens=args.max_tokens,
        seed=args.seed,
        top_k=args.top_k,
    )
    dense = RunConfig(name="dense", quest_enabled=False, **common)
    quest_no_off = RunConfig(
        name="quest_no_offload", quest_enabled=True,
        gpu_cache_blocks_per_seq=args.large_gpu_blocks,
        expect_offload=False, **common,
    )
    quest_offload = RunConfig(
        name="quest_offload", quest_enabled=True,
        gpu_cache_blocks_per_seq=args.small_gpu_blocks,
        expect_offload=True, **common,
    )
    return [dense, quest_no_off, quest_offload]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Quest sparse-offload benchmark harness")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--output-dir", default="./quest_bench_out")
    p.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.55)
    p.add_argument("--num-paragraphs", type=int, default=40,
                   help="prompt = paragraph repeated this many times "
                        "(~62 tokens each; default ~2400 tokens = ~9 blocks "
                        "of 256, so a small gpu budget overflows)")
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--top-k", type=int, default=4,
                   help="blocks selected per decode step. Must be "
                        "<= gpu_cache_blocks_per_seq at BOTH points, so it is "
                        "bounded by --small-gpu-blocks (validate() enforces).")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--large-gpu-blocks", type=int, default=512,
                   help="gpu_cache_blocks_per_seq for the no-offload point")
    p.add_argument("--small-gpu-blocks", type=int, default=4,
                   help="gpu_cache_blocks_per_seq for the forced-offload point; "
                        "must be < blocks the prompt produces AND >= --top-k")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    # apply_model ships a Python callable to the worker; v1 IPC needs opt-in.
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

    plan = build_run_plan(args)

    # small_gpu_blocks must still satisfy top_k <= gpu_cache_blocks_per_seq.
    if args.small_gpu_blocks < args.top_k:
        raise SystemExit(
            f"--small-gpu-blocks ({args.small_gpu_blocks}) must be >= --top-k "
            f"({args.top_k}); QuestConfig.validate() enforces "
            f"top_k <= gpu_cache_blocks_per_seq. Lower --top-k or raise "
            f"--small-gpu-blocks (keep it below the prompt's block count)."
        )

    output_dir = Path(args.output_dir)
    tmp_dir = output_dir / "_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    dense_lp = None
    for cfg in plan:
        print(f"\n>>> running config: {cfg.name} ...", flush=True)
        rec = run_one(cfg, str(tmp_dir))
        records.append(rec)
        if cfg.name == "dense":
            dense_lp = rec["step_logprobs"]

    # Quality: cosine of each Quest point vs the dense baseline.
    if dense_lp is not None:
        for rec in records:
            if rec["config"]["quest_enabled"]:
                cos = stepwise_cosine_vs_dense(dense_lp, rec["step_logprobs"])
                rec["cosine_vs_dense_mean"] = cos["mean"]
                rec["cosine_vs_dense_min"] = cos["min"]
                rec["cosine_vs_dense_per_step"] = cos.get("per_step", [])

    # Trim per-step logprobs out of the records we keep in memory before
    # writing? No — JSON keeps them for re-analysis; CSV omits them.
    write_outputs(records, output_dir)
    print_summary(records)

    print(f"Wrote {output_dir / 'quest_benchmark.json'} and .csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

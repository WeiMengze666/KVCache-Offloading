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
import sys
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

# Model / prompt defaults (roadmap: Llama-3.2-3B-Instruct, temperature=0, seeded).
DEFAULT_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
DEFAULT_SEED = 1234
DEFAULT_BLOCK_SIZE = 256

# A single paragraph (kept for reference / the legacy homogeneous prompt).
_PARAGRAPH = (
    "In the spring of the year 1789, the assembly convened in Versailles "
    "to address grievances that had accumulated over decades of fiscal "
    "mismanagement and shifting alliances among the nobility. The delegates "
    "argued late into the night about representation, taxation, and the "
    "ancient privileges of the clergy and the aristocracy. "
)

# Heterogeneous prompt vocabulary (Task 6b). build_prompt repeating ONE
# paragraph made every KV block near-identical, so Quest selection was
# degenerate: all blocks interchangeable -> every layer picked the same indices
# (cross-layer Jaccard 1.0) and any few blocks reproduced dense output (cosine
# 1.0) even at top_k=2/49. A real top_k sweep needs blocks with DIFFERENT
# content. We deterministically compose distinct numbered sentences drawn from
# several unrelated topics so adjacent blocks differ and selection is meaningful.
_TOPICS = [
    ("history", "the assembly at Versailles debated taxation and the privileges of the nobility"),
    ("biology", "the ribosome translated messenger RNA into a chain of amino acids inside the cell"),
    ("astronomy", "the spectrometer measured redshift in distant galaxies receding across the void"),
    ("cooking", "the chef reduced the stock slowly, folding shallots and thyme into the simmering pan"),
    ("law", "the appellate court weighed precedent on the statute of limitations and remanded the case"),
    ("geology", "tectonic plates ground past one another, uplifting the ridge over countless millennia"),
    ("music", "the quartet modulated from a minor key into a bright resolving cadence at the coda"),
    ("finance", "the analyst hedged the portfolio against currency risk before the quarterly earnings call"),
    ("sailing", "the crew trimmed the mainsail and tacked upwind against a stiff northeasterly gust"),
    ("medicine", "the clinician auscultated the patient and ordered an assay to confirm the diagnosis"),
]
# PLACEHOLDER_BODY


def build_prompt(num_paragraphs: int, *, heterogeneous: bool = True) -> str:
    """Build the benchmark prompt.

    NOTE (handoff): this built-in prompt is intentionally simple and exists only
    for PIPELINE SELF-CHECK / SMOKE runs — to drive the harness end-to-end and
    produce non-degenerate signals. It is NOT the final quality dataset. The
    detailed quality sweep is done by the follow-up team using LongBench (its
    data is not downloaded on this machine). Keep this generator crude on
    purpose; point real evaluation at LongBench instead.

    heterogeneous=True (default, Task 6b): each "paragraph" is a distinct,
    numbered sentence rotating through unrelated topics, so different KV blocks
    carry genuinely different content and the top_k / overlap sweeps are
    meaningful. Deterministic (no RNG) so runs are reproducible.

    heterogeneous=False: the legacy single-paragraph repetition (kept for
    back-compat / A-B checks). It makes every KV block near-identical, which
    makes Quest selection DEGENERATE (cross-layer Jaccard and Quest-vs-dense
    cosine both stick at 1.0 even at very small top_k) — do not use it to draw
    sparsity-quality conclusions.
    """
    if not heterogeneous:
        return (_PARAGRAPH * num_paragraphs).strip()
    parts = []
    for i in range(num_paragraphs):
        topic, sentence = _TOPICS[i % len(_TOPICS)]
        # Vary wording per-iteration so even same-topic blocks differ; the
        # index + ordinal keep token content changing down the sequence.
        parts.append(
            f"Section {i + 1} ({topic}): on day {i * 7 + 3}, {sentence}; "
            f"observers numbered {1000 - i * 3} and recorded outcome {i % 97}."
        )
    return " ".join(parts)


def _run_meta(argv: list[str]) -> dict:
    """Whole-run metadata written once per run (run_meta.json). Keyed by start
    time (utc) — the user manages/compares data across versions by start time.
    No git stamping (user decision). GPU/driver info is best-effort (None on
    failure, not fatal)."""
    from datetime import datetime, timezone
    gpu_name = None
    driver = None
    try:
        import torch
        gpu_name = torch.cuda.get_device_name(0)
        driver = torch.version.cuda  # coarse "CUDA build" tag; fine to be rough
    except Exception:
        pass
    return {
        "utc": datetime.now(timezone.utc).isoformat(),  # = run start time
        "argv": list(argv),
        "gpu_name": gpu_name,
        "driver": driver,
        # Stage-1 records are the offload-bypassed baseline; Stage 2 emits "real".
        "offload_mode_note": "bypassed (experiment-s0): TierManager dormant, "
                             "gpu pool aliases full engine cache",
    }


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
    # Cross-version ablation label. Stage 1 is the offload-bypassed baseline;
    # Stage 2 (real bounded GPU pool) emits offload_mode="real" for comparison.
    offload_mode: str = "bypassed"
    # Two-pass run model: "clean" = no debug instrumentation (accurate latency/
    # throughput); "instrumented" = debug counters + driver-side gpu timeline.
    pass_kind: str = "clean"


def _debug_counters_for(cfg) -> bool:
    """Clean pass: all debug instrumentation OFF (accurate timing). Instrumented
    pass: ON (overlap buffer, cuda-event timing, NVTX)."""
    return cfg.pass_kind == "instrumented"


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
        # Cross-layer overlap (Task 4): drain this TierManager's per-step
        # selected-block log, keyed by quest slot (tm.layer_idx, contiguous
        # 0..num_quest_layers-1). Empty unless enable_overlap_capture is on
        # (instrumented pass). drain_selected may be absent on older builds.
        drain_fn = getattr(tm, "drain_selected", None)
        selected_log = drain_fn() if callable(drain_fn) else []
        out.append(
            {
                "name": name,
                "layer_idx": getattr(mod, "layer_idx", None),
                "quest_slot": getattr(tm, "layer_idx", None),
                "stats": _asdict(s) if s is not None else None,
                "selected_log": selected_log,
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


def _per_adjacent_pair_means(pairs: list[dict]) -> dict[str, float]:
    """Mean Jaccard per adjacent-slot pair, keyed "a->b"."""
    import statistics
    by_pair: dict[str, list[float]] = {}
    for p in pairs:
        by_pair.setdefault(f"{p['slot_a']}->{p['slot_b']}", []).append(
            p["jaccard"])
    return {k: statistics.fmean(v) for k, v in by_pair.items()}


def adjacent_layer_jaccard(per_slot_log: dict[int, list[dict]]) -> dict:
    """per_slot_log: {quest_slot: [{step, seq_id, block_ids}, ...]}.
    Returns mean/median Jaccard between consecutive slots over all (step,seq)."""
    import statistics
    slots = sorted(per_slot_log)
    # index by (slot) -> {(step,seq): set(block_ids)}
    idx = {s: {(d["step"], d["seq_id"]): set(d["block_ids"])
               for d in per_slot_log[s]} for s in slots}
    pairs = []
    for a, b in zip(slots, slots[1:]):
        common_keys = set(idx[a]) & set(idx[b])
        for key in common_keys:
            sa, sb = idx[a][key], idx[b][key]
            union = sa | sb
            j = len(sa & sb) / len(union) if union else 1.0
            pairs.append({"slot_a": a, "slot_b": b, "step": key[0],
                          "seq_id": key[1], "jaccard": j})
    vals = [p["jaccard"] for p in pairs]
    return {
        "mean_jaccard": statistics.fmean(vals) if vals else 0.0,
        "median_jaccard": statistics.median(vals) if vals else 0.0,
        "n_pairs": len(pairs),
        "per_pair_mean": _per_adjacent_pair_means(pairs),  # {f"{a}->{b}": mean}
        "go_no_go": ("GO: prev-layer prefetch likely worth it"
                     if (statistics.fmean(vals) if vals else 0) >= 0.5
                     else "NO-GO: low overlap; Stage 3 b/c + Stage 6 prefetch dubious"),
    }


def _probe_memory(worker) -> dict:
    """Runs INSIDE the EngineCore process via collective_rpc(worker). Real
    measured GPU memory, split by component. Uses vLLM's own MemorySnapshot
    (total = mem_get_info; torch_reserved = cudaMalloc'd by torch; non_torch =
    total-free-torch) plus the worker's weights + KV-pool-reserved bytes.

    Note (offload-bypass): kv_reserved_bytes reflects vLLM's pre-reservation to
    gpu_memory_utilization, NOT Quest's working set — constant on this baseline.
    That is exactly what we record so Stage 2 (real bounded pool) can be diffed
    against it. cuda_used_bytes is the live device footprint at probe time."""
    import torch
    free, total = torch.cuda.mem_get_info()
    torch_reserved = torch.cuda.memory_reserved()
    torch_peak = torch.cuda.memory_stats().get("allocated_bytes.all.peak", 0)
    cuda_used = total - free
    weights = int(getattr(getattr(worker, "model_runner", None),
                          "model_memory_usage", 0) or 0)
    kv_reserved = int(getattr(worker, "available_kv_cache_memory_bytes", 0) or 0)
    return {
        "total_bytes": int(total),
        "cuda_used_bytes": int(cuda_used),       # real device footprint
        "torch_reserved_bytes": int(torch_reserved),
        "torch_peak_alloc_bytes": int(torch_peak),  # ≈ activation peak + weights
        "non_torch_bytes": int(cuda_used - torch_reserved),
        "weights_bytes": weights,
        "kv_reserved_bytes": kv_reserved,
        # derived activation estimate (peak alloc minus weights), clamped >=0
        "activation_peak_bytes": max(0, int(torch_peak) - weights),
    }


def _teardown_engine(llm) -> None:
    """Explicitly release the engine (Bug A). vLLM v1 runs the engine in a
    separate VLLM::EngineCore process the worker spawned; relying on GC to fire
    the weakref.finalize finalizer races interpreter teardown and orphans that
    process holding ~23 GiB. Drive vLLM's own shutdown path deterministically
    (engine_core.shutdown -> CoreEngineProcManager.shutdown -> terminate/join/
    kill_process_tree), then drop refs + empty the cache. Never raises."""
    import gc
    try:
        core = getattr(getattr(llm, "llm_engine", None), "engine_core", None)
        if core is not None and hasattr(core, "shutdown"):
            core.shutdown()
    except Exception as e:
        print(f"[teardown] engine_core.shutdown raised (ignored): {e!r}", flush=True)
    try:
        del llm
    except Exception:
        pass
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


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

    llm = None
    try:
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
                # Two-pass: debug counters (cuda-Event timing, overlap buffer,
                # NVTX) ON only for the instrumented pass; OFF on the clean pass
                # so latency/throughput are unperturbed.
                enable_debug_counters=_debug_counters_for(cfg),
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

        result: dict[str, Any] = {
            "name": cfg.name,
            "config": cfg_dict,
            "prompt_chars": len(prompt),
            "prompt_tokens": len(out.prompt_token_ids),
            "gen_tokens": gen_tokens,
            "latency_s": elapsed,
            "decode_tokens_per_s": gen_tokens / elapsed if elapsed else 0.0,
            "output_text": out.outputs[0].text,
            "step_logprobs": step_logprobs,
        }

        # Real per-component memory, measured in-engine (Bug B fix). collective_rpc
        # passes the worker; returns one dict per rank (TP=1 -> take [0]).
        try:
            mem = llm.llm_engine.collective_rpc(_probe_memory)[0]
        except Exception as e:
            mem = {"error": repr(e)}
        result["memory_breakdown"] = mem

        if cfg.quest_enabled:
            per_layer = llm.llm_engine.apply_model(_probe_quest_layers)
            flat = []
            for rank in per_layer:
                flat.extend(rank)
            agg = _aggregate_stats(flat)
            result["quest_stats"] = agg
            # Cross-layer top-k Jaccard overlap (Task 4, headline Stage 1
            # metric). Build {quest_slot: selected_log} from the probe, then
            # compute adjacent-slot Jaccard offline. Empty on the clean pass
            # (capture off -> all selected_log empty -> n_pairs=0, mean 0.0).
            per_slot_log: dict[int, list[dict]] = {}
            for entry in flat:
                slot = entry.get("quest_slot")
                log = entry.get("selected_log") or []
                if slot is None or not log:
                    continue
                per_slot_log.setdefault(int(slot), []).extend(log)
            result["cross_layer_overlap"] = adjacent_layer_jaccard(per_slot_log)
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
    finally:
        _teardown_engine(llm)
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


def _sample_card_used_mib() -> int:
    """Card-total used GPU memory (MiB), summed over all visible GPUs. Best
    effort: returns 0 on any failure (nvidia-smi missing, parse error)."""
    import subprocess
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True).stdout
        return sum(int(x.strip()) for x in out.splitlines() if x.strip())
    except Exception:
        return 0


def _sample_card_util_pct() -> int:
    """Card GPU utilization percent (first visible GPU). 0 on failure."""
    import subprocess
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True).stdout.splitlines()
        return int(out[0].strip()) if out and out[0].strip() else 0
    except Exception:
        return 0


def _sample_gpu_timeline(stop_evt, baseline_mib, out_list, period_s=0.1):
    """Background thread (instrumented pass only): poll card-total GPU mem + util
    every period_s until stop_evt is set, appending one sample per tick.

    Simplification (plan Task 3 step 4 sanctioned): the GPU is run single-tenant
    for benchmarks, so card-total-used MINUS a baseline snapshot taken just
    before p.start() == this run's footprint, with zero in-engine perturbation
    and no fiddly EngineCore-child-pid resolution. Records
    {t, used_mib_delta_from_baseline, util_pct} per sample."""
    import time
    t0 = time.perf_counter()
    while not stop_evt.is_set():
        used = _sample_card_used_mib()
        out_list.append({
            "t": round(time.perf_counter() - t0, 4),
            "used_mib_delta_from_baseline": used - baseline_mib,
            "util_pct": _sample_card_util_pct(),
        })
        time.sleep(period_s)


def run_one(cfg: RunConfig, tmp_dir: str, join_timeout: float = 900.0) -> dict[str, Any]:
    """Run one config in its own spawned subprocess; return its result.

    Bug A: bound the join and hard-kill the process tree on timeout so a hung
    EngineCore cannot block the whole sweep. A killed/non-zero worker is
    surfaced, not swallowed.

    Task 3 (two-pass): on the instrumented pass, a driver-side background thread
    samples card-total GPU mem + util while the engine subprocess runs; the
    resulting timeline is attached as result["gpu_timeline"]. The clean pass runs
    with no sampler so its latency/throughput are unperturbed."""
    import threading

    from vllm.utils.system_utils import kill_process_tree
    out_path = Path(tmp_dir) / f"result_{cfg.name}.json"
    ctx = mp.get_context("spawn")
    p = ctx.Process(target=_engine_worker, args=(asdict(cfg), tmp_dir, str(out_path)))

    instrumented = cfg.pass_kind == "instrumented"
    stop_evt = threading.Event()
    samples: list[dict] = []
    sampler = None
    if instrumented:
        baseline_mib = _sample_card_used_mib()  # snapshot just before p.start()
        sampler = threading.Thread(
            target=_sample_gpu_timeline,
            args=(stop_evt, baseline_mib, samples),
            daemon=True,
        )
        sampler.start()

    try:
        p.start()
        p.join(join_timeout)
        if p.is_alive():
            if p.pid is not None:
                kill_process_tree(p.pid)   # reaps the EngineCore grandchild too
            p.join(30)
            raise RuntimeError(
                f"engine subprocess for '{cfg.name}' exceeded {join_timeout}s and "
                f"was killed (likely hung EngineCore — see Bug A / offload-bypass)."
            )
        if p.exitcode != 0:
            raise RuntimeError(
                f"engine subprocess for '{cfg.name}' exited with code {p.exitcode}"
            )
    finally:
        # Always stop + join the sampler so the thread never outlives run_one.
        stop_evt.set()
        if sampler is not None:
            sampler.join(5)

    rec = json.loads(out_path.read_text())
    if instrumented:
        rec["gpu_timeline"] = samples
    return rec


def run_one_profiled(cfg, tmp_dir, nsys_out_dir):
    """Profile a single config under nsys. Runs the worker as a fresh python
    -c that calls _engine_worker, wrapped by nsys so the spawned EngineCore
    child is traced too. Produces <nsys_out_dir>/<cfg.name>.nsys-rep."""
    import subprocess, sys, json as _json
    Path(nsys_out_dir).mkdir(parents=True, exist_ok=True)
    rep = Path(nsys_out_dir) / cfg.name
    out_path = Path(tmp_dir) / f"result_{cfg.name}.json"
    worker_py = (
        "import json,sys;"
        "from benchmarks.benchmark_quest import _engine_worker, RunConfig;"
        f"_engine_worker({asdict(cfg)!r}, {str(tmp_dir)!r}, {str(out_path)!r})"
    )
    cmd = [
        "nsys", "profile", "-f", "true", "-o", str(rep),
        "--trace=cuda,nvtx,osrt", "--trace-fork-before-exec=true",
        "--cuda-memory-usage=true",   # also capture alloc events (user switch)
        sys.executable, "-c", worker_py,
    ]
    env = {**os.environ, "VLLM_ALLOW_INSECURE_SERIALIZATION": "1"}
    subprocess.run(cmd, check=True, env=env)
    return _json.loads(out_path.read_text())


def write_outputs(records: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    # Full JSON (includes per-step logprobs for later re-analysis).
    (output_dir / "quest_benchmark.json").write_text(json.dumps(records, indent=2))
    # Flat CSV (one row per config point; drop the bulky logprob arrays).
    write_csv(records, output_dir / "quest_benchmark.csv")


# Flat CSV columns (one row per config point; drop the bulky logprob arrays).
# Existing columns + offload_mode (the cross-version ablation label).
# Bug B: peak_gpu_alloc_gib (always ~0.0, wrong process) replaced by the real
# in-engine memory_breakdown components (as GiB).
_CSV_COLS = [
    "name", "quest_enabled", "pass_kind", "prompt_tokens", "gen_tokens",
    "latency_s",
    "decode_tokens_per_s",
    "cuda_used_gib", "weights_gib", "kv_reserved_gib",
    "cosine_vs_dense_mean", "cosine_vs_dense_min",
    "gpu_cache_blocks_per_seq", "top_k",
    "block_filled", "evict_d2h", "load_h2d",
    "selected_total", "selected_on_gpu", "gpu_hit_rate",
    "mean_h2d_wait_ms", "mean_evict_stall_ms",
    "offload_mode",
    "error",
]


def _csv_row(r: dict) -> dict:
    """Build one flat CSV row from a record (drops the bulky logprob arrays)."""
    qs = r.get("quest_stats", {}) or {}
    cfg = r.get("config", {}) or {}
    mem = r.get("memory_breakdown", {}) or {}
    _gib = 1024 ** 3
    # Error stub (a config that raised): emit a row with name + offload_mode +
    # the error, zeros elsewhere, so the CSV still has one line per planned point.
    if r.get("error") and "prompt_tokens" not in r:
        row = {c: 0 for c in _CSV_COLS}
        row["name"] = r.get("name", "?")
        row["quest_enabled"] = cfg.get("quest_enabled", "")
        row["pass_kind"] = cfg.get("pass_kind", "")
        row["offload_mode"] = cfg.get("offload_mode", "bypassed")
        row["error"] = r["error"]
        return row
    return {
        "name": r["name"],
        "quest_enabled": cfg["quest_enabled"],
        "pass_kind": cfg.get("pass_kind", "clean"),
        "prompt_tokens": r["prompt_tokens"],
        "gen_tokens": r["gen_tokens"],
        "latency_s": round(r["latency_s"], 4),
        "decode_tokens_per_s": round(r["decode_tokens_per_s"], 2),
        "cuda_used_gib": round(mem.get("cuda_used_bytes", 0) / _gib, 3),
        "weights_gib": round(mem.get("weights_bytes", 0) / _gib, 3),
        "kv_reserved_gib": round(mem.get("kv_reserved_bytes", 0) / _gib, 3),
        "cosine_vs_dense_mean": round(r.get("cosine_vs_dense_mean", 1.0), 5),
        "cosine_vs_dense_min": round(r.get("cosine_vs_dense_min", 1.0), 5),
        "gpu_cache_blocks_per_seq": cfg.get("gpu_cache_blocks_per_seq", 0),
        "top_k": cfg.get("top_k", 0),
        "block_filled": qs.get("block_filled", 0),
        "evict_d2h": qs.get("evict_d2h", 0),
        "load_h2d": qs.get("load_h2d", 0),
        "selected_total": qs.get("selected_total", 0),
        "selected_on_gpu": qs.get("selected_on_gpu", 0),
        "gpu_hit_rate": round(qs.get("gpu_hit_rate", 0.0), 4),
        "mean_h2d_wait_ms": round(qs.get("mean_h2d_wait_ms", 0.0), 4),
        "mean_evict_stall_ms": round(qs.get("mean_evict_stall_ms", 0.0), 4),
        "offload_mode": cfg.get("offload_mode", "bypassed"),
    }


def write_csv(records: list[dict], path: Path) -> None:
    """Write the flat per-config CSV (existing cols + offload_mode)."""
    with Path(path).open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=_CSV_COLS)
        w.writeheader()
        for r in records:
            w.writerow(_csv_row(r))


def write_versioned(records: list[dict], run_meta: dict, data_root: Path) -> Path:
    # Directory key = run start time only (compact ISO basic format, UTC).
    # e.g. 2026-05-31T18:42:05.123456+00:00 -> 20260531T184205Z
    iso = run_meta["utc"]
    stamp = iso.replace("-", "").replace(":", "").split(".")[0]
    stamp = stamp.replace("+0000", "Z").replace("+00:00", "Z")
    if not stamp.endswith("Z"):
        stamp += "Z"
    out = data_root / "stage1" / stamp
    out.mkdir(parents=True, exist_ok=True)
    (out / "run_meta.json").write_text(json.dumps(run_meta, indent=2))
    with (out / "records.jsonl").open("w") as fh:
        for r in records:
            fh.write(json.dumps({**r, "meta": run_meta}) + "\n")
    # records.csv: reuse the existing flat-CSV column writer (incl. offload_mode).
    write_csv(records, out / "records.csv")
    return out


def print_summary(records: list[dict]) -> None:
    print("\n" + "=" * 78)
    print("QUEST BENCHMARK SUMMARY")
    print("=" * 78)
    for r in records:
        # Error stubs (a config that raised) carry only name/config/error.
        if r.get("error") and "prompt_tokens" not in r:
            print(f"\n[{r['name']}] FAILED: {r['error']}")
            continue
        cfg = r["config"]
        qs = r.get("quest_stats", {}) or {}
        mem = r.get("memory_breakdown", {}) or {}
        _gib = 1024 ** 3
        print(f"\n[{r['name']}] quest={cfg['quest_enabled']} "
              f"gpu_blocks/seq={cfg['gpu_cache_blocks_per_seq']} top_k={cfg['top_k']}")
        print(f"  prompt={r['prompt_tokens']}tok gen={r['gen_tokens']}tok "
              f"latency={r['latency_s']:.3f}s "
              f"decode={r['decode_tokens_per_s']:.1f}tok/s")
        if "error" in mem:
            print(f"  memory: <probe error: {mem['error']}>")
        else:
            print(f"  memory: cuda_used={mem.get('cuda_used_bytes', 0) / _gib:.2f}GiB "
                  f"weights={mem.get('weights_bytes', 0) / _gib:.2f}GiB "
                  f"kv_reserved={mem.get('kv_reserved_bytes', 0) / _gib:.2f}GiB "
                  f"non_torch={mem.get('non_torch_bytes', 0) / _gib:.2f}GiB")
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

    Task 3 (two-pass): each base config is emitted once per requested pass (see
    --pass). The clean pass carries accurate latency/throughput (no debug
    instrumentation); the instrumented pass carries the resource timeline. Each
    emitted config is tagged with its pass_kind and its name suffixed _clean /
    _inst so the records (JSONL/CSV) distinguish the two.
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

    # Task 6 — top_k quality sweep. When --top-k-sweep is set, the plan is one
    # dense baseline + one quest point per requested top_k, ALL on the LARGE gpu
    # budget (expect_offload=False everywhere) so the sweep isolates SELECTION
    # QUALITY (cosine vs top_k) rather than offload. 'ALL' maps to a top_k >= the
    # prompt's candidate-block count (--large-gpu-blocks); the impl clamps via
    # min(top_k, full_blocks) at impl_helpers.py so a large value == "all blocks".
    sweep = _parse_topk_sweep(args.top_k_sweep, args.large_gpu_blocks)
    if sweep:
        sweep_common = dict(common)
        sweep_common.pop("top_k")  # set per sweep point below
        base = [RunConfig(name="dense", quest_enabled=False, **sweep_common)]
        for label, k in sweep:
            base.append(RunConfig(
                name=f"quest_topk{label}", quest_enabled=True,
                top_k=k,
                gpu_cache_blocks_per_seq=args.large_gpu_blocks,
                expect_offload=False, **sweep_common,
            ))
    else:
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
        base = [dense, quest_no_off, quest_offload]

    # Expand each base config once per requested pass.
    passes = _passes_for(args.pass_choice)
    suffix = {"clean": "_clean", "instrumented": "_inst"}
    plan: list[RunConfig] = []
    for cfg in base:
        for pk in passes:
            plan.append(replace(cfg, name=cfg.name + suffix[pk], pass_kind=pk))
    return plan


def _parse_topk_sweep(spec: str, large_gpu_blocks: int) -> list[tuple[str, int]]:
    """Parse the --top-k-sweep comma list into (label, top_k) pairs, preserving
    order and de-duplicating. Empty/blank spec -> [] (use the fixed plan). 'ALL'
    (case-insensitive) maps to large_gpu_blocks so that, with the impl's
    min(top_k, full_blocks) clamp, every candidate block is selected (== dense).
    The label keeps the original token ('ALL' or the integer) for the point name
    quest_topk{label}."""
    out: list[tuple[str, int]] = []
    seen: set[str] = set()
    for tok in (spec or "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        label = "ALL" if tok.upper() == "ALL" else tok
        if label in seen:
            continue
        seen.add(label)
        k = large_gpu_blocks if label == "ALL" else int(tok)
        out.append((label, k))
    return out


def _passes_for(pass_choice: str) -> list[str]:
    """Resolve the --pass selector to the ordered list of pass_kinds to run.
    Clean first so the canonical (unperturbed) timing/quality lands before the
    instrumented pass."""
    if pass_choice == "both":
        return ["clean", "instrumented"]
    return [pass_choice]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Quest sparse-offload benchmark harness")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--output-dir", default="./quest_bench_out")
    p.add_argument("--pass", dest="pass_choice",
                   choices=["clean", "instrumented", "both"], default="both",
                   help="two-pass run model: 'clean' = accurate latency/"
                        "throughput (no debug instrumentation); 'instrumented' = "
                        "debug counters ON + driver-side gpu_timeline; 'both' "
                        "(default) emits each config once per pass, suffixing "
                        "names _clean/_inst.")
    p.add_argument("--data-root", default="/home/xinyan/vLLM_quest/data",
                   help="git-external root for the versioned record schema; "
                        "results land in <data-root>/stage1/<utc-timestamp>/ "
                        "(timestamp only, no git sha; kept OUTSIDE the fork repo "
                        "so it persists across branches for cross-version ablation)")
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
    p.add_argument("--top-k-sweep", default="",
                   help="comma list of top_k values for the quality sweep "
                        "(e.g. 'ALL,64,32,16,8'; default empty = the existing "
                        "fixed [dense, quest_no_off, quest_offload] plan). When "
                        "set, build_run_plan emits one dense point + one quest "
                        "point per top_k, ALL with a LARGE gpu budget "
                        "(gpu_cache_blocks_per_seq=--large-gpu-blocks, so "
                        "expect_offload=False everywhere) — the sweep measures "
                        "selection quality, not offload. 'ALL' maps to a top_k "
                        ">= the prompt's candidate-block count (--large-gpu-"
                        "blocks; impl clamps via min(top_k, full_blocks)).")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--large-gpu-blocks", type=int, default=512,
                   help="gpu_cache_blocks_per_seq for the no-offload point")
    p.add_argument("--small-gpu-blocks", type=int, default=4,
                   help="gpu_cache_blocks_per_seq for the forced-offload point; "
                        "must be < blocks the prompt produces AND >= --top-k")
    p.add_argument("--profile", action="store_true",
                   help="wrap the INSTRUMENTED-pass worker under `nsys profile` "
                        "(never the clean pass — nsys perturbs timing) so the "
                        "spawned EngineCore child is traced too "
                        "(--trace-fork-before-exec). Produces a .nsys-rep per "
                        "config carrying the gated Quest NVTX ranges "
                        "(quest.select/ensure_resident/sparse_attn).")
    p.add_argument("--nsys-out", default=None,
                   help="dir for nsys .nsys-rep traces (default <data-root>/nsys). "
                        "--cuda-memory-usage=true is the 'also trace memory "
                        "allocations' switch.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    # apply_model ships a Python callable to the worker; v1 IPC needs opt-in.
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

    # Cross-version ablation stamp: build the run metadata once for this run.
    run_meta = _run_meta(sys.argv[1:] if argv is None else list(argv))

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

    # nsys trace dir: default <data-root>/nsys (Task 5 --profile). Only used for
    # the instrumented pass; the clean pass is never wrapped by nsys.
    nsys_out = args.nsys_out or str(Path(args.data_root) / "nsys")

    records: list[dict] = []
    # Dense baseline logprobs keyed by pass_kind, so each Quest point is scored
    # against the dense point from its OWN pass (clean vs clean, inst vs inst);
    # names are now suffixed _clean/_inst (Task 3 two-pass expansion).
    dense_lp_by_pass: dict[str, list] = {}
    for cfg in plan:
        print(f"\n>>> running config: {cfg.name} ...", flush=True)
        # Fault-tolerant sweep: a single config failing (e.g. the offload-bypass
        # assertion that always fires on experiment-s0, or an OOM on one point)
        # must NOT abort the whole sweep and lose every other config's data.
        # Record an error stub and keep going; the run is always written below.
        try:
            # --profile wraps the INSTRUMENTED pass under nsys (never the clean
            # pass — nsys perturbs timing). The clean pass keeps the in-process
            # mp.Process + bounded join (Bug A teardown) path.
            if args.profile and cfg.pass_kind == "instrumented":
                rec = run_one_profiled(cfg, str(tmp_dir), nsys_out)
            else:
                rec = run_one(cfg, str(tmp_dir))
        except Exception as e:
            print(f"!!! config {cfg.name} FAILED: {e!r} — recording stub, "
                  f"continuing sweep", flush=True)
            rec = {
                "name": cfg.name,
                "config": asdict(cfg),
                "error": repr(e),
            }
            records.append(rec)
            continue
        records.append(rec)
        if not cfg.quest_enabled:
            dense_lp_by_pass[cfg.pass_kind] = rec["step_logprobs"]

    # Quality: cosine of each Quest point vs the dense baseline of the same pass.
    for rec in records:
        if rec.get("error") or "config" not in rec:
            continue
        if rec["config"]["quest_enabled"]:
            dense_lp = dense_lp_by_pass.get(rec["config"].get("pass_kind", "clean"))
            if dense_lp is None:
                continue
            cos = stepwise_cosine_vs_dense(dense_lp, rec["step_logprobs"])
            rec["cosine_vs_dense_mean"] = cos["mean"]
            rec["cosine_vs_dense_min"] = cos["min"]
            rec["cosine_vs_dense_per_step"] = cos.get("per_step", [])

    # Trim per-step logprobs out of the records we keep in memory before
    # writing? No — JSON keeps them for re-analysis; CSV omits them.
    write_outputs(records, output_dir)
    # Versioned, git-external record schema for cross-version ablation
    # (keyed by run start time only — no git sha).
    versioned_dir = write_versioned(records, run_meta, Path(args.data_root))
    print_summary(records)

    print(f"Wrote {output_dir / 'quest_benchmark.json'} and .csv")
    print(f"Wrote versioned records to {versioned_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

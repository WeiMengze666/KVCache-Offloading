# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-RunConfig subprocess body.

Spawned by __main__.py (multiprocessing.spawn). Builds an LLM, probes memory
at sample boundaries (no background sampling thread — the worker process's
collective_rpc input socket is shared with the engine's request stream and
concurrent sends from a separate thread cause IPC frame corruption), then
flushes per-config CSV + summary JSON.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

from benchmarks.quest_memory_probe import probes
from benchmarks.quest_memory_probe.configs import RunConfig
from benchmarks.quest_memory_probe.csv_writer import write_rows
from benchmarks.quest_memory_probe.summary import aggregate_samples
from benchmarks.quest_memory_probe.workload import load_samples


def _now_ms() -> int:
    return time.monotonic_ns() // 1_000_000


def _is_oom_error(e: BaseException) -> bool:
    # Best-effort OOM classification. We avoid importing torch here so the
    # helper stays test-friendly; matching by the runtime error message is
    # how vLLM itself surfaces OOM in offline runs.
    msg = str(e).lower()
    return "out of memory" in msg or "cuda oom" in msg


def _make_engine_kwargs(
    cfg: RunConfig,
    *,
    quest_json_path: str | None,
) -> dict[str, Any]:
    """Common shared kwargs for vllm.LLM(...). Returns a dict that can be
    splatted into the LLM constructor. The Quest config json is written
    elsewhere (kept outside this helper so it can be unit-tested without
    disk access).

    If cfg.max_model_len exceeds the model's native max_position_embeddings,
    hf_overrides is injected so vLLM doesn't refuse on context overflow.
    Llama-3.2 ships with 131072; LongBench-v2 long bucket starts at ~167k
    tokens, so users running long-bucket samples must raise max_model_len
    past that boundary.
    """
    base: dict[str, Any] = dict(
        model=cfg.model,
        dtype=cfg.dtype,
        enforce_eager=cfg.enforce_eager,
        max_model_len=cfg.max_model_len,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        block_size=cfg.block_size,
        seed=cfg.seed,
    )
    # 131072 is the native max_position for Llama-3.2 (the model this probe
    # targets per docs/run/probe-memory.md). Hard-coding the threshold avoids
    # an AutoConfig disk load in tests; users running a different base model
    # with a smaller native context can extend this check.
    if cfg.max_model_len > 131072:
        base["hf_overrides"] = {"max_position_embeddings": cfg.max_model_len}
    if cfg.quest_enabled and quest_json_path is not None:
        base["enable_quest_sparse_offload"] = True
        base["quest_config"] = quest_json_path
    return base


def _write_quest_config_json(cfg: RunConfig, dir_: Path) -> str:
    from vllm.config.quest import QuestConfig

    qc = QuestConfig(
        enabled=True,
        block_size=cfg.block_size,
        top_k=cfg.top_k,
        full_kv_layers=list(cfg.full_kv_layers),
        gpu_cache_blocks_per_seq=cfg.gpu_cache_blocks_per_seq,
        cpu_cache_blocks=cfg.cpu_cache_blocks,
        cpu_cache_gib=int(cfg.cpu_cache_gib) if cfg.cpu_cache_gib > 0 else None,
        selection_impl=cfg.selection_impl,
        enable_async_prefetch=False,
    )
    qc.validate()
    p = dir_ / f"quest_cfg_{cfg.name}.json"
    p.write_text(json.dumps(qc.to_dict()))
    return str(p)


def execute(cfg: RunConfig, out_dir: Path) -> None:
    """Subprocess entrypoint.

    Writes <out_dir>/<cfg.name>/{samples.csv,summary.json,stdout.log}.
    """
    cfg_dir = out_dir / cfg.name
    cfg_dir.mkdir(parents=True, exist_ok=True)
    log_f = (cfg_dir / "stdout.log").open("w", encoding="utf-8")
    sys.stdout = log_f  # type: ignore[assignment]
    sys.stderr = log_f  # type: ignore[assignment]
    # dup2 the underlying fds 1/2 onto log_f so the vLLM EngineCore subprocess
    # (which inherits these fds, not Python's sys.stdout/sys.stderr objects)
    # also writes its OOM tracebacks and ZMQ logs into the same file. Without
    # this, EngineCore deaths surface as "EngineDeadError: see stack trace
    # above" with no actual stack trace anywhere in our captured output.
    try:
        os.dup2(log_f.fileno(), 1)
        os.dup2(log_f.fileno(), 2)
    except OSError as e:
        print(f"[runner] dup2 failed (will rely on Python-level redirect): {e!r}")

    # vLLM v1 IPC: collective_rpc requires this opt-in to ship a Python
    # callable to engine-core.
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

    try:
        cfg.validate()
        samples = load_samples(
            cfg.workload_spec, model=cfg.model, longbench_full=cfg.longbench_full
        )
        # OOM-sweep configs walk samples in ascending prompt length.
        if cfg.name.endswith("_oom"):
            samples = sorted(samples, key=lambda s: s.prompt_tokens)

        from vllm import LLM, SamplingParams  # deferred import

        rows: list[dict[str, Any]] = []

        with tempfile.TemporaryDirectory() as td:
            quest_json = (
                _write_quest_config_json(cfg, Path(td)) if cfg.quest_enabled else None
            )
            kwargs = _make_engine_kwargs(cfg, quest_json_path=quest_json)
            llm = LLM(**kwargs)

            try:
                n_tm = llm.llm_engine.collective_rpc(
                    lambda w: len(probes._collect_tier_managers(w))
                )[0]
                print(f"[runner] discovered {n_tm} TierManager(s) on cfg={cfg.name}")
            except Exception as e:
                print(f"[runner] introspection failed: {e!r}")

            try:
                bpb = llm.llm_engine.collective_rpc(probes.probe_bytes_per_block)[0]
            except Exception as e:
                print(f"[runner] bytes_per_block probe failed: {e!r}")
                bpb = None

            def take_snapshot(phase: str, **extra: Any) -> None:
                """Probe once and append a row tagged `phase`. Catches probe
                failures so generation continues even if a single RPC fails.
                """
                try:
                    snap = llm.llm_engine.collective_rpc(
                        probes.probe_snapshot,
                        args=(bpb,),
                    )[0]
                except Exception as e:
                    snap = {"error": repr(e)}
                snap["ts_ms"] = _now_ms()
                snap["phase"] = phase
                snap.update(extra)
                rows.append(snap)

            rows.append({"ts_ms": _now_ms(), "phase": "engine_init_done"})
            # Initial baseline snapshot inside a synthetic "warmup" window so
            # the aggregator records a steady-state pre-generation point.
            rows.append(
                {
                    "ts_ms": _now_ms(),
                    "phase": "sample_start",
                    "sample_id": "_warmup",
                    "prompt_tokens": 0,
                }
            )
            take_snapshot("sampling")
            rows.append(
                {
                    "ts_ms": _now_ms(),
                    "phase": "sample_end",
                    "sample_id": "_warmup",
                    "gen_tokens": 0,
                    "latency_s": 0.0,
                }
            )

            params = SamplingParams(
                temperature=0.0,
                max_tokens=cfg.max_tokens,
                seed=cfg.seed,
            )

            consecutive_oom = 0
            for s in samples:
                rows.append(
                    {
                        "ts_ms": _now_ms(),
                        "phase": "sample_start",
                        "sample_id": s.sample_id,
                        "prompt_tokens": s.prompt_tokens,
                    }
                )
                # Probe once at sample start (before peak counter reset). This
                # falls into the active window because we just emitted
                # sample_start above.
                take_snapshot("sampling")
                try:
                    llm.llm_engine.collective_rpc(probes.reset_peak_stats)
                    t0 = time.perf_counter()
                    out = llm.generate([s.prompt], params, use_tqdm=False)[0]
                    elapsed = time.perf_counter() - t0
                    # Probe immediately after generate so the post-decode
                    # state (peak resident, hit ratio, slack) lands in this
                    # sample's window.
                    take_snapshot("sampling")
                    rows.append(
                        {
                            "ts_ms": _now_ms(),
                            "phase": "sample_end",
                            "sample_id": s.sample_id,
                            "gen_tokens": len(out.outputs[0].token_ids),
                            "latency_s": elapsed,
                        }
                    )
                    consecutive_oom = 0
                except BaseException as e:
                    if not _is_oom_error(e):
                        raise
                    consecutive_oom += 1
                    rows.append(
                        {
                            "ts_ms": _now_ms(),
                            "phase": "oom_at_sample",
                            "sample_id": s.sample_id,
                            "error": repr(e),
                        }
                    )
                    if cfg.name.endswith("_oom") and consecutive_oom >= 2:
                        break

            rows.append({"ts_ms": _now_ms(), "phase": "teardown"})
            _teardown(llm)

        write_rows(cfg_dir / "samples.csv", rows)
        per_sample = aggregate_samples(rows)
        (cfg_dir / "summary.json").write_text(
            json.dumps(
                {
                    "config": cfg.to_dict(),
                    "samples": per_sample,
                },
                indent=2,
            )
        )
        print(f"[runner] cfg={cfg.name} done; {len(per_sample)} samples")
    except Exception:
        traceback.print_exc()
        raise
    finally:
        log_f.flush()
        log_f.close()


def _teardown(llm) -> None:
    """Mirror benchmark_quest._teardown_engine to ensure engine_core dies."""
    import gc

    try:
        core = getattr(getattr(llm, "llm_engine", None), "engine_core", None)
        if core is not None and hasattr(core, "shutdown"):
            core.shutdown()
    except Exception as e:
        print(f"[teardown] engine_core.shutdown raised (ignored): {e!r}")
    with contextlib.suppress(Exception):
        del llm
    gc.collect()
    try:
        import torch

        torch.accelerator.empty_cache()
    except Exception:
        pass


def run_in_child(cfg_dict: dict, out_dir_str: str) -> None:
    """Spawn target. Defined in `runner` (not `__main__`) so multiprocessing.spawn
    can re-import it across the pickling boundary when the parent was launched
    via `python -m benchmarks.quest_memory_probe`.
    """
    # vLLM v1 spawns an EngineCore subprocess; if it inherits the default 'fork'
    # start method, init_device hits "Cannot re-initialize CUDA in forked
    # subprocess" because this child has already touched CUDA (NVML probe etc.).
    # Force spawn via vLLM's own env knob — must be set BEFORE the deferred
    # `from vllm import LLM` inside execute().
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    cfg = RunConfig.from_dict(cfg_dict)
    execute(cfg, Path(out_dir_str))

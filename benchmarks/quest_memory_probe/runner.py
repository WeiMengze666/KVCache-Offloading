# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-RunConfig subprocess body.

Spawned by __main__.py (multiprocessing.spawn). Builds an LLM, runs the
sampler in the background, generates each Sample from the workload, then
flushes per-config CSV + summary JSON.
"""

from __future__ import annotations

import contextlib
import json
import os
import queue
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

from benchmarks.quest_memory_probe import probes
from benchmarks.quest_memory_probe.configs import RunConfig
from benchmarks.quest_memory_probe.csv_writer import write_rows
from benchmarks.quest_memory_probe.sampler import Sampler
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
    disk access)."""
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
        cpu_cache_gib=cfg.cpu_cache_gib,
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

    # vLLM v1 IPC: collective_rpc requires this opt-in to ship a Python
    # callable to engine-core.
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

    try:
        cfg.validate()
        samples = load_samples(cfg.workload_spec, model=cfg.model)
        # OOM-sweep configs walk samples in ascending prompt length.
        if cfg.name.endswith("_oom"):
            samples = sorted(samples, key=lambda s: s.prompt_tokens)

        from vllm import LLM, SamplingParams  # deferred import

        rows: list[dict[str, Any]] = []
        q: queue.Queue = queue.Queue()

        with tempfile.TemporaryDirectory() as td:
            quest_json = (
                _write_quest_config_json(cfg, Path(td)) if cfg.quest_enabled else None
            )
            kwargs = _make_engine_kwargs(cfg, quest_json_path=quest_json)
            llm = LLM(**kwargs)

            # One-shot introspection for the log; helps debug if the
            # collective_rpc probe later finds 0 tier_managers.
            try:
                n_tm = llm.llm_engine.collective_rpc(
                    lambda w: len(probes._collect_tier_managers(w))
                )[0]
                print(f"[runner] discovered {n_tm} TierManager(s) on cfg={cfg.name}")
            except Exception as e:
                print(f"[runner] introspection failed: {e!r}")

            # Cache bytes_per_block once; it's a function of layer geometry.
            try:
                bpb = llm.llm_engine.collective_rpc(probes.probe_bytes_per_block)[0]
            except Exception as e:
                print(f"[runner] bytes_per_block probe failed: {e!r}")
                bpb = None

            def snap():
                return llm.llm_engine.collective_rpc(
                    probes.probe_snapshot,
                    args=(bpb,),
                )[0]

            rows.append({"ts_ms": _now_ms(), "phase": "engine_init_done"})
            sampler = Sampler(
                snapshot_fn=snap,
                interval_s=cfg.probe_interval_ms / 1000.0,
                queue_=q,
            )
            sampler.start()

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
                try:
                    llm.llm_engine.collective_rpc(probes.reset_peak_stats)
                    t0 = time.perf_counter()
                    out = llm.generate([s.prompt], params, use_tqdm=False)[0]
                    elapsed = time.perf_counter() - t0
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

            sampler.stop()
            sampler.join(timeout=5.0)

            while not q.empty():
                rows.append(q.get_nowait())
            rows.sort(key=lambda r: r.get("ts_ms", 0))

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

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CLI entrypoint.

Parses subcommand args, expands to a list of RunConfigs, then spawns one
child process per config to run benchmarks/quest_memory_probe/runner.py:execute.
After all configs finish, calls report.build_report(out_dir).

Parent process MUST NOT import torch or vllm — that creates a CUDA context
the spawned children would inherit and OOM on. We import only at the top of
runner.execute (which always runs in a child).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from benchmarks.quest_memory_probe.configs import (
    RunConfig,
    expand_dense_vs_quest,
    expand_oom_sweep,
    expand_pool_size,
)


def _commit_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def _csv_ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="benchmarks.quest_memory_probe",
        description="Quest memory observation experiment harness.",
    )
    sub = p.add_subparsers(dest="subcommand", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--samples",
        required=True,
        help="workload spec, e.g. 'longbench:narrativeqa:lengths=short,medium:n=2'",
    )
    common.add_argument(
        "--top-k",
        type=int,
        required=True,
        help="quest top_k (must divide pool sizes)",
    )
    common.add_argument("--probe-interval-ms", type=int, default=250)
    common.add_argument(
        "--out-dir",
        required=True,
        help="output directory; created if missing",
    )
    common.add_argument(
        "--max-tokens",
        type=int,
        default=64,
        help="max generated tokens per sample",
    )
    common.add_argument("--gpu-mem-util", type=float, default=0.55)

    a = sub.add_parser("compare-dense-vs-quest", parents=[common])
    a.add_argument("--quest-pool", type=int, required=True)

    b = sub.add_parser("compare-pool-size", parents=[common])
    b.add_argument(
        "--pool-sizes",
        type=_csv_ints,
        required=True,
        help="comma-separated pool sizes",
    )

    c = sub.add_parser("oom-sweep", parents=[common])
    c.add_argument("--quest-pool", type=int, required=True)

    return p


def _apply_common(cfgs: list[RunConfig], args) -> list[RunConfig]:
    """Inject CLI overrides (probe interval, gpu_memory_utilization, max_tokens)."""
    out = []
    for c in cfgs:
        out.append(
            RunConfig(
                **{
                    **c.to_dict(),
                    "probe_interval_ms": args.probe_interval_ms,
                    "gpu_memory_utilization": args.gpu_mem_util,
                    "max_tokens": args.max_tokens,
                }
            )
        )
    for c in out:
        c.validate()
    return out


def args_to_configs(args) -> list[RunConfig]:
    if args.subcommand == "compare-dense-vs-quest":
        cfgs = expand_dense_vs_quest(
            workload_spec=args.samples,
            top_k=args.top_k,
            quest_pool=args.quest_pool,
        )
    elif args.subcommand == "compare-pool-size":
        cfgs = expand_pool_size(
            workload_spec=args.samples,
            top_k=args.top_k,
            pool_sizes=args.pool_sizes,
        )
    elif args.subcommand == "oom-sweep":
        cfgs = expand_oom_sweep(
            workload_spec=args.samples,
            top_k=args.top_k,
            quest_pool=args.quest_pool,
        )
    else:
        raise ValueError(f"unknown subcommand {args.subcommand!r}")
    return _apply_common(cfgs, args)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfgs = args_to_configs(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "subcommand": args.subcommand,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "commit": _commit_hash(),
        "configs": [c.to_dict() for c in cfgs],
        "argv": sys.argv,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    ctx = mp.get_context("spawn")
    from benchmarks.quest_memory_probe.runner import run_in_child

    for c in cfgs:
        print(f"[main] launching cfg={c.name}", flush=True)
        t0 = time.perf_counter()
        proc = ctx.Process(
            target=run_in_child,
            args=(c.to_dict(), str(out_dir)),
            name=f"runner-{c.name}",
        )
        proc.start()
        proc.join()
        print(
            f"[main] cfg={c.name} exited code={proc.exitcode} "
            f"in {time.perf_counter() - t0:.1f}s",
            flush=True,
        )

    # Build report after all children exit.
    from benchmarks.quest_memory_probe.report import build_report

    build_report(out_dir)
    print(f"[main] report at {out_dir}/report.md", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

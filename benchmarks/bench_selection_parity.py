# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage 1 Task 7 — selection_impl parity + kernel-timing micro-benchmark.

Standalone (NO engine): exercises the three Quest block-selection impls
(torch oracle / triton / cuda) directly on small tensors at Llama-3.2-3B GQA
shapes (num_kv_heads=8, num_q_heads=24 -> G=3, head_size=128), sweeping the
candidate-block count B and top_k. For each (B, top_k) it:

  1. warms up + times N iters of each impl with torch.cuda.Event, and
  2. ASSERTS each non-oracle impl picks the IDENTICAL top-k set as torch.

Emits `data/stage1/<utc-timestamp>/selection_parity.json` reusing
`_run_meta` from benchmark_quest (same versioned, git-external data dir).

This is fully valid on the offload-bypassed baseline (pure selection, no KV
movement) and answers "which selection_impl for later stages".

Usage:
  export PATH=/usr/local/cuda-12.8/bin:$PATH CUDA_HOME=/usr/local/cuda-12.8
  .venv/bin/python benchmarks/bench_selection_parity.py --data-root /tmp/quest_data_t7
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

# Reuse the run-metadata helper from the main harness (lightweight import:
# benchmark_quest has no heavy vllm imports at module scope).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_quest import _run_meta  # noqa: E402

# Llama-3.2-3B GQA: 8 kv heads, 24 q heads -> G=3, head_size 128.
H_KV, G, D = 8, 3, 128
SWEEP_B = [9, 64, 256, 512]
SWEEP_TOP_K = [4, 16, 64]


def time_impl(fn, *, q, s, c, G, top_k, iters=200, warmup=20):
    for _ in range(warmup):
        fn(query=q, block_summary=s, candidate_ids=c, num_kv_groups=G, top_k=top_k)
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    ts = []
    for _ in range(iters):
        start.record()
        fn(query=q, block_summary=s, candidate_ids=c, num_kv_groups=G, top_k=top_k)
        end.record()
        end.synchronize()
        ts.append(start.elapsed_time(end) * 1000.0)  # us
    return ts


def _make_inputs(B, seed=0):
    torch.manual_seed(seed)
    q = torch.randn(H_KV * G, D, dtype=torch.float16, device="cuda")
    s = torch.randn(B, 2, H_KV, D, dtype=torch.float16, device="cuda")
    c = torch.arange(B, dtype=torch.int32, device="cuda")
    return q, s, c


def _topk_set(fn, *, q, s, c, top_k):
    return set(
        fn(query=q, block_summary=s, candidate_ids=c, num_kv_groups=G, top_k=top_k)
        .cpu()
        .tolist()
    )


def run_sweep(iters, warmup):
    """Returns a list of per-(B, top_k) result dicts. Asserts parity vs torch."""
    from vllm.v1.attention.ops.quest_selection_dispatch import (
        _resolve_selection_callable,
    )

    impls = {}
    for impl in ("torch", "triton", "cuda"):
        try:
            impls[impl] = _resolve_selection_callable(impl)
        except Exception as e:  # pragma: no cover - cuda unavailable host
            print(f"[warn] selection_impl {impl!r} unavailable: {e}")

    results = []
    for B in SWEEP_B:
        for top_k in SWEEP_TOP_K:
            if top_k > B:
                continue
            q, s, c = _make_inputs(B)
            # Oracle set first.
            ref_set = _topk_set(impls["torch"], q=q, s=s, c=c, top_k=top_k)
            point = {"B": B, "top_k": top_k, "impls": {}}
            for impl, fn in impls.items():
                got_set = _topk_set(fn, q=q, s=s, c=c, top_k=top_k)
                matches = got_set == ref_set
                # Hard parity gate for the non-oracle impls (a real kernel bug
                # if it ever trips). torch trivially matches itself.
                if impl != "torch":
                    assert matches, (
                        f"PARITY FAIL impl={impl} B={B} top_k={top_k}: "
                        f"{sorted(got_set)} vs torch {sorted(ref_set)}"
                    )
                ts = time_impl(fn, q=q, s=s, c=c, G=G, top_k=top_k,
                               iters=iters, warmup=warmup)
                point["impls"][impl] = {
                    "mean_us": statistics.mean(ts),
                    "median_us": statistics.median(ts),
                    "stdev_us": statistics.pstdev(ts) if len(ts) > 1 else 0.0,
                    "min_us": min(ts),
                    "iters": len(ts),
                    "top_k_set_matches_torch": matches,
                }
            results.append(point)
            row = "  ".join(
                f"{impl}={point['impls'][impl]['mean_us']:8.2f}us"
                f"({'ok' if point['impls'][impl]['top_k_set_matches_torch'] else 'MISMATCH'})"
                for impl in impls
            )
            print(f"B={B:4d} top_k={top_k:3d} | {row}")
    return results


def _stamp(iso: str) -> str:
    # Mirror benchmark_quest.write_versioned: compact ISO basic, UTC, trailing Z.
    stamp = iso.replace("-", "").replace(":", "").split(".")[0]
    stamp = stamp.replace("+0000", "Z").replace("+00:00", "Z")
    if not stamp.endswith("Z"):
        stamp += "Z"
    return stamp


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Quest selection_impl parity + kernel-timing micro-benchmark "
                    "(standalone, no engine)."
    )
    p.add_argument("--data-root", default="/home/xinyan/vLLM_quest/data",
                   help="git-external root; JSON lands in "
                        "<data-root>/stage1/<utc-timestamp>/selection_parity.json")
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--warmup", type=int, default=20)
    args = p.parse_args(argv)

    if not torch.cuda.is_available():
        print("CUDA not available; selection parity/timing needs a GPU.")
        return 1

    run_meta = _run_meta(sys.argv[1:] if argv is None else list(argv))
    results = run_sweep(iters=args.iters, warmup=args.warmup)

    all_match = all(
        pt["impls"][impl]["top_k_set_matches_torch"]
        for pt in results for impl in pt["impls"] if impl != "torch"
    )

    out_dir = Path(args.data_root) / "stage1" / _stamp(run_meta["utc"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "selection_parity.json"
    payload = {
        "meta": run_meta,
        "shapes": {"num_kv_heads": H_KV, "num_kv_groups": G,
                   "num_q_heads": H_KV * G, "head_size": D},
        "iters": args.iters,
        "warmup": args.warmup,
        "all_non_oracle_match_torch": all_match,
        "results": results,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nall_non_oracle_match_torch={all_match}")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

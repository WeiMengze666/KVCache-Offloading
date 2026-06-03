# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-sample aggregation of probe rows to summary records.

A "sample" is the time window between a sample_start and the matching
sample_end OR oom_at_sample marker. Sampling rows outside any sample window
are ignored.

Aggregates produced per sample:
- peak_*  -> max of the metric over sampling rows in window
- mean_*  -> median of the metric (legacy field name; we use median)
- oom     -> True if window ended with oom_at_sample
"""

from __future__ import annotations

import statistics
from collections.abc import Iterable
from typing import Any

_PEAK_FIELDS = (
    ("peak_nvml_used_bytes", "nvml.gpu_used_bytes"),
    ("peak_torch_allocated_bytes", "torch.allocated_bytes"),
    ("peak_torch_reserved_bytes", "torch.reserved_bytes"),
    ("peak_kv_useful_bytes", "vllm.gpu_kv_useful_bytes"),
    ("peak_actual_used_bytes", "vllm.actual_used_bytes"),
    ("peak_actual_used_peak_bytes", "vllm.actual_used_peak_bytes"),
    ("peak_engine_essential_peak_bytes", "vllm.engine_essential_peak_bytes"),
)
_MEDIAN_FIELDS = (
    ("mean_kv_slack_bytes", "vllm.kv_pool_slack_bytes"),
    ("mean_topk_hit_ratio", "quest.topk_hit_ratio"),
    ("mean_engine_essential_bytes", "vllm.engine_essential_bytes"),
    ("mean_kv_useful_bytes", "vllm.gpu_kv_useful_bytes"),
)


def _values(rows: list[dict], key: str) -> list[float]:
    out = []
    for r in rows:
        v = r.get(key)
        if v is None or v == "":
            continue
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            continue
    return out


def _peak(rows: list[dict], key: str) -> float:
    vals = _values(rows, key)
    return max(vals) if vals else 0


def _median(rows: list[dict], key: str):
    vals = _values(rows, key)
    return statistics.median(vals) if vals else None


def aggregate_samples(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = list(rows)
    samples: list[dict] = []
    current_start: dict | None = None
    current_sampling: list[dict] = []
    for r in rows:
        phase = r.get("phase")
        if phase == "sample_start":
            current_start = r
            current_sampling = []
        elif phase == "sampling":
            if current_start is not None:
                current_sampling.append(r)
        elif phase in ("sample_end", "oom_at_sample"):
            if current_start is None:
                continue
            agg = {
                "sample_id": current_start.get("sample_id"),
                "prompt_tokens": current_start.get("prompt_tokens"),
                "gen_tokens": r.get("gen_tokens"),
                "latency_s": r.get("latency_s"),
                "oom": phase == "oom_at_sample",
            }
            for out_key, src in _PEAK_FIELDS:
                agg[out_key] = _peak(current_sampling, src)
            for out_key, src in _MEDIAN_FIELDS:
                agg[out_key] = _median(current_sampling, src)
            samples.append(agg)
            current_start = None
            current_sampling = []
    return samples

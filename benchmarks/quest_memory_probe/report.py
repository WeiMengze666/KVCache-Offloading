# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Render plots + markdown report from per-config summary.json/samples.csv.

Imports matplotlib lazily and uses the 'Agg' backend so this works headlessly.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

_GIB = 1024**3


def _load_samples_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for r in reader:
            rows.append({k: (None if v == "" else v) for k, v in r.items()})
    return rows


def _f(v) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _setup_mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _plot_memory_timeline(cfg_name: str, rows: list[dict], out: Path) -> None:
    plt = _setup_mpl()
    sampling = [r for r in rows if r.get("phase") == "sampling"]
    if not sampling:
        return
    ts = [_f(r["ts_ms"]) for r in sampling]
    ts0 = ts[0] or 0
    t = [(x - ts0) / 1000.0 for x in ts]
    essential = [
        (_f(r.get("vllm.engine_essential_bytes")) or 0) / _GIB for r in sampling
    ]
    useful = [(_f(r.get("vllm.gpu_kv_useful_bytes")) or 0) / _GIB for r in sampling]
    slack = [(_f(r.get("vllm.kv_pool_slack_bytes")) or 0) / _GIB for r in sampling]
    nvml = [(_f(r.get("nvml.gpu_used_bytes")) or 0) / _GIB for r in sampling]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.stackplot(
        t,
        essential,
        useful,
        slack,
        labels=["engine_essential", "kv_useful", "kv_slack"],
        alpha=0.7,
    )
    ax.plot(t, nvml, "k--", label="nvml.gpu_used", linewidth=1.0)
    ax.set_xlabel("time since engine_init (s)")
    ax.set_ylabel("GiB")
    ax.set_title(f"GPU memory timeline — {cfg_name}")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)


def _plot_peak_bar(summaries: list[dict], out: Path) -> None:
    plt = _setup_mpl()
    names = [s["config"]["name"] for s in summaries]
    if not names:
        return
    nvml = [
        max((sample["peak_nvml_used_bytes"] for sample in s["samples"]), default=0)
        / _GIB
        for s in summaries
    ]
    torch_alloc = [
        max(
            (sample["peak_torch_allocated_bytes"] for sample in s["samples"]), default=0
        )
        / _GIB
        for s in summaries
    ]
    useful = [
        max((sample["peak_kv_useful_bytes"] for sample in s["samples"]), default=0)
        / _GIB
        for s in summaries
    ]

    import numpy as np

    x = np.arange(len(names))
    w = 0.27
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w, nvml, w, label="peak nvml.gpu_used")
    ax.bar(x, torch_alloc, w, label="peak torch.allocated")
    ax.bar(x + w, useful, w, label="peak kv_useful")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("GiB")
    ax.set_title("Peak GPU memory across configs")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)


def _plot_kv_pool_breakdown(summaries: list[dict], out: Path) -> None:
    plt = _setup_mpl()
    names = [s["config"]["name"] for s in summaries]
    if not names:
        return
    essential = [
        (s["samples"][0].get("mean_engine_essential_bytes") or 0) / _GIB
        if s["samples"]
        else 0
        for s in summaries
    ]
    useful = [
        (s["samples"][0].get("mean_kv_useful_bytes") or 0) / _GIB if s["samples"] else 0
        for s in summaries
    ]
    slack = [
        (s["samples"][0].get("mean_kv_slack_bytes") or 0) / _GIB if s["samples"] else 0
        for s in summaries
    ]
    import numpy as np

    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x, essential, label="engine_essential")
    ax.bar(x, useful, bottom=essential, label="kv_useful")
    bottoms = [a + b for a, b in zip(essential, useful)]
    ax.bar(x, slack, bottom=bottoms, label="kv_slack")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("GiB")
    ax.set_title("KV pool composition (steady-state median)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)


def _plot_topk_hit_ratio(cfg_name: str, summary: dict, out: Path) -> None:
    plt = _setup_mpl()
    samples = summary.get("samples", [])
    if not samples:
        return
    ratios = [s.get("mean_topk_hit_ratio") for s in samples]
    if all(r is None for r in ratios):
        return
    ratios = [0.0 if r is None else float(r) for r in ratios]
    ids = [s["sample_id"] for s in samples]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(len(ids)), ratios, marker="o")
    ax.set_xticks(range(len(ids)))
    ax.set_xticklabels(ids, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("mean top-k GPU hit ratio")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Top-k hit ratio per sample — {cfg_name}")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)


def _plot_oom_threshold(summaries: list[dict], out: Path) -> None:
    plt = _setup_mpl()
    fig, ax = plt.subplots(figsize=(10, 5))
    plotted = 0
    for s in summaries:
        samples = s.get("samples", [])
        if not samples:
            continue
        if not any(sample.get("oom") for sample in samples):
            continue
        xs = [sample["prompt_tokens"] for sample in samples]
        ys = [sample.get("peak_nvml_used_bytes", 0) / _GIB for sample in samples]
        colors = ["red" if sample.get("oom") else "green" for sample in samples]
        ax.scatter(xs, ys, c=colors, label=s["config"]["name"])
        plotted += 1
    if plotted == 0:
        plt.close(fig)
        return
    ax.set_xlabel("prompt_tokens")
    ax.set_ylabel("peak nvml.gpu_used (GiB)")
    ax.set_title("OOM threshold sweep — red=OOM, green=success")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)


def _takeaway(summaries: list[dict]) -> list[str]:
    out = []
    by_name = {s["config"]["name"]: s for s in summaries}
    dense_keys = [k for k in by_name if k.startswith("dense")]
    quest_keys = [k for k in by_name if "quest" in k]
    if dense_keys and quest_keys:
        d = by_name[dense_keys[0]]
        q = min(
            (by_name[k] for k in quest_keys),
            key=lambda s: s["config"].get("gpu_cache_blocks_per_seq", 1 << 30),
        )
        d_useful = max((sm["peak_kv_useful_bytes"] for sm in d["samples"]), default=0)
        q_useful = max((sm["peak_kv_useful_bytes"] for sm in q["samples"]), default=0)
        if d_useful > 0:
            ratio = q_useful / d_useful
            out.append(
                f"- 最小池 Quest 配置 (`{q['config']['name']}`) 的 "
                f"`peak_kv_useful` = {q_useful / _GIB:.2f} GiB，相对 Dense "
                f"(`{d['config']['name']}`, {d_useful / _GIB:.2f} GiB) 比值 "
                f"{ratio:.2f}（即 {ratio * 100:.0f}% / 节省 "
                f"{(1 - ratio) * 100:.0f}%）。"
            )
    if len(quest_keys) >= 2:
        qs = sorted(
            (by_name[k] for k in quest_keys),
            key=lambda s: s["config"].get("gpu_cache_blocks_per_seq", 0),
        )
        small, big = qs[0], qs[-1]
        small_slack = (
            small["samples"][0].get("mean_kv_slack_bytes") or 0
            if small["samples"]
            else 0
        )
        big_slack = (
            big["samples"][0].get("mean_kv_slack_bytes") or 0 if big["samples"] else 0
        )
        out.append(
            f"- 池从 {big['config'].get('gpu_cache_blocks_per_seq')} 缩到 "
            f"{small['config'].get('gpu_cache_blocks_per_seq')} 时，"
            f"`kv_slack` 中位数从 {big_slack / _GIB:.2f} → "
            f"{small_slack / _GIB:.2f} GiB。"
        )
    oom_summaries = [
        s for s in summaries if any(sm.get("oom") for sm in s.get("samples", []))
    ]
    for s in oom_summaries:
        ok = [sm for sm in s["samples"] if not sm.get("oom")]
        if ok:
            last_ok = max(ok, key=lambda x: x["prompt_tokens"])
            out.append(
                f"- `{s['config']['name']}` OOM 阈值前最长成功 prompt = "
                f"{last_ok['prompt_tokens']} tokens。"
            )
    return out or ["- 无可生成的 takeaway（数据不足）。"]


def build_report(out_dir: Path) -> None:
    out_dir = Path(out_dir)
    manifest = json.loads((out_dir / "manifest.json").read_text())
    cfg_names = [c["name"] for c in manifest["configs"]]
    summaries: list[dict] = []
    for name in cfg_names:
        sj = out_dir / name / "summary.json"
        if not sj.exists():
            print(f"[report] WARN missing {sj}")
            continue
        summaries.append(json.loads(sj.read_text()))

    plots = out_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    for s in summaries:
        name = s["config"]["name"]
        rows = _load_samples_csv(out_dir / name / "samples.csv")
        _plot_memory_timeline(name, rows, plots / f"memory_timeline_{name}.png")
        if s["config"].get("quest_enabled"):
            _plot_topk_hit_ratio(name, s, plots / f"topk_hit_ratio_{name}.png")
    _plot_peak_bar(summaries, plots / "memory_peak_bar.png")
    _plot_kv_pool_breakdown(summaries, plots / "kv_pool_breakdown.png")
    if manifest.get("subcommand") == "oom-sweep":
        _plot_oom_threshold(summaries, plots / "oom_threshold.png")

    md_lines = [
        f"# Quest 显存观测报告 — {manifest.get('timestamp', '')}",
        "",
        "## 实验参数",
        f"- subcommand: `{manifest.get('subcommand')}`",
        f"- commit: `{manifest.get('commit')}`",
        f"- argv: `{' '.join(manifest.get('argv', []))}`",
        "",
        "## 配置矩阵",
        "| name | quest | top_k | gpu_pool_blks |",
        "|---|---|---|---|",
    ]
    for c in manifest["configs"]:
        md_lines.append(
            f"| {c['name']} | {c.get('quest_enabled', False)} | "
            f"{c.get('top_k', '-')} | "
            f"{c.get('gpu_cache_blocks_per_seq', '-')} |"
        )
    md_lines += ["", "## 关键发现", *_takeaway(summaries), "", "## 时间序列"]
    for s in summaries:
        n = s["config"]["name"]
        md_lines.append(f"### {n}")
        md_lines.append(f"![memory_timeline_{n}](plots/memory_timeline_{n}.png)")
    md_lines += [
        "",
        "## 跨配置峰值对比",
        "![memory_peak_bar](plots/memory_peak_bar.png)",
        "",
        "## KV 池构成稳态对比",
        "![kv_pool_breakdown](plots/kv_pool_breakdown.png)",
        "",
        "## Top-k 命中率",
    ]
    for s in summaries:
        n = s["config"]["name"]
        if s["config"].get("quest_enabled"):
            md_lines.append(f"![topk_hit_ratio_{n}](plots/topk_hit_ratio_{n}.png)")
    if manifest.get("subcommand") == "oom-sweep":
        md_lines += ["", "## OOM 阈值", "![oom_threshold](plots/oom_threshold.png)"]
    md_lines += ["", "## 原始数据"]
    for s in summaries:
        n = s["config"]["name"]
        md_lines.append(f"- `{n}/samples.csv`、`{n}/summary.json`")

    (out_dir / "report.md").write_text("\n".join(md_lines))

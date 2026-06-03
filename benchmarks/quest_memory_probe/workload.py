# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Workload loader: LongBench-v2 with synthetic fallback.

LongBench loading is gated behind successful `datasets.load_dataset` call.
If anything fails (no network + no cache, missing dataset), we fall back to
a deterministic synthetic prompt of the requested length. The runner records
which path was used so the report makes it explicit.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

# Length buckets (token counts after tokenization).
_BUCKET_BOUNDARIES = {
    "short": (0, 4096),
    "medium": (4096, 16384),
    "long": (16384, 49152),
    "xlong": (49152, 1 << 30),
}
_VALID_BUCKETS = tuple(_BUCKET_BOUNDARIES.keys())
_VALID_SOURCES = ("longbench",)


@dataclass(frozen=True)
class Sample:
    sample_id: str
    prompt: str
    prompt_tokens: int
    bucket: str
    expected_max_tokens: int = 64


@dataclass(frozen=True)
class WorkloadSpec:
    source: str
    task: str
    buckets: tuple[str, ...]
    n: int


def parse_spec(spec: str) -> WorkloadSpec:
    """Parse 'longbench:narrativeqa:lengths=short,medium:n=2'."""
    parts = spec.split(":")
    if len(parts) < 4:
        raise ValueError(f"workload spec needs at least 4 colon-parts: {spec!r}")
    source, task, lengths_kv, n_kv = parts[0], parts[1], parts[2], parts[3]
    if source not in _VALID_SOURCES:
        raise ValueError(f"unknown workload source {source!r}; valid: {_VALID_SOURCES}")
    if not lengths_kv.startswith("lengths="):
        raise ValueError(f"third part must be 'lengths=...': {lengths_kv!r}")
    buckets = tuple(lengths_kv[len("lengths=") :].split(","))
    for b in buckets:
        if b not in _VALID_BUCKETS:
            raise ValueError(f"unknown length bucket {b!r}; valid: {_VALID_BUCKETS}")
    if not n_kv.startswith("n="):
        raise ValueError(f"fourth part must be 'n=...': {n_kv!r}")
    n = int(n_kv[len("n=") :])
    return WorkloadSpec(source=source, task=task, buckets=buckets, n=n)


def bucket_for_tokens(n_tokens: int) -> str:
    for bucket, (lo, hi) in _BUCKET_BOUNDARIES.items():
        if lo <= n_tokens < hi:
            return bucket
    raise ValueError(f"token count {n_tokens} outside known buckets")


# Synthetic prompt vocabulary — 10 short paragraphs.
# (will be tokenized differently per model but the order of magnitude is fine
# for length-bucket selection).
_PARAGRAPHS = [
    "The cartographer unfurled the parchment and traced the river's winding course "
    "through three forgotten kingdoms whose names had long since faded from memory.",
    "In the laboratory the spectrometer registered an unexpected absorption line near "
    "five hundred eighty nanometers, suggesting a previously uncataloged "
    "metal-organic complex.",
    "Pebbles clattered down the talus slope as the surveyor's hammer rang against "
    "an outcrop of unusually fine-grained basalt veined with pale green olivine.",
    "The auditor reviewed seven quarters of consolidated statements and flagged an "
    "inconsistency between the receivables aging schedule and the working capital "
    "reconciliation.",
    "Beneath the sail the navigator triangulated by sextant, accounting for "
    "refraction near the horizon and a steady three-knot current pulling the ship "
    "north by northwest.",
    "Inside the scriptorium the monk mixed iron gall with rainwater, careful to "
    "keep the proportion consistent so the lettering would not bleed through the "
    "vellum overnight.",
    "The clinician auscultated the patient's lungs, noted faint crackles at the "
    "right base, and ordered a chest radiograph and inflammatory marker assay "
    "before the next ward round.",
    "When the lecturer opened the manuscript the audience saw an annotated diagram "
    "explaining why prime gaps grow logarithmically on average yet exhibit "
    "unbounded irregular spikes.",
    "Across the threshold of the temple the archaeologist photographed weathered "
    "glyphs that appeared to record astronomical alignments rather than dynastic "
    "genealogies as previously assumed.",
    "Late in the trial the engineer admitted that the redundant safety interlock "
    "had been decommissioned during a controller firmware migration two years "
    "before the incident.",
]


def _tokens_for_prompt(prompt: str) -> int:
    """Conservative estimate: 4 chars per token (Llama-3.2 average ~3.5)."""
    return max(1, len(prompt) // 4)


def load_samples_synthetic(
    *,
    buckets: Sequence[str],
    n: int,
    seed: int = 1234,
) -> list[Sample]:
    """Fallback prompt generator: deterministic, no external deps.

    Each bucket targets the MIDDLE of its token range to keep the bucketing
    test predictable even with the rough char/token estimate.
    """
    targets = {
        "short": 2048,  # mid of [0, 4096)
        "medium": 10000,  # mid of [4096, 16384)
        "long": 32000,  # mid of [16384, 49152)
        "xlong": 80000,  # comfortably > 49152
    }
    out: list[Sample] = []
    for bucket in buckets:
        target_tokens = targets[bucket]
        target_chars = target_tokens * 4
        for i in range(n):
            # rotate paragraph order per (bucket, i) so blocks differ
            offset = (hash((bucket, i, seed)) & 0xFF) % len(_PARAGRAPHS)
            chunks = []
            chars = 0
            j = 0
            while chars < target_chars:
                p = _PARAGRAPHS[(offset + j) % len(_PARAGRAPHS)]
                # numbered to keep KV blocks distinguishable
                chunks.append(f"[{j + 1:04d}] {p}")
                chars += len(p) + 8
                j += 1
            prompt = "\n".join(chunks)
            tokens = _tokens_for_prompt(prompt)
            assert bucket_for_tokens(tokens) == bucket, (
                f"synthetic prompt for bucket={bucket} tokenized to {tokens}, "
                f"which falls outside that bucket; tighten _tokens_for_prompt."
            )
            out.append(
                Sample(
                    sample_id=f"synthetic/{bucket}/{i}",
                    prompt=prompt,
                    prompt_tokens=tokens,
                    bucket=bucket,
                )
            )
    return out


def _load_dataset(name: str, split: str):
    """Indirection for monkeypatching in tests."""
    from datasets import load_dataset

    return load_dataset(name, split=split)


def _read_template(name: str) -> str:
    longbench_dir = Path("/home/yijun/offload_attn/LongBench/prompts")
    return (longbench_dir / name).read_text(encoding="utf-8")


def _build_longbench_prompt(item: dict, template: str) -> str:
    """Mirror pred_quest_vllm.build_prompt without re-importing it (avoid
    pulling vllm into a no-GPU test path). Substitutes context, question,
    and four choices into the template."""
    return (
        template.replace("$DOC$", item.get("context", ""))
        .replace("$Q$", item.get("question", ""))
        .replace("$C_A$", item.get("choice_A", ""))
        .replace("$C_B$", item.get("choice_B", ""))
        .replace("$C_C$", item.get("choice_C", ""))
        .replace("$C_D$", item.get("choice_D", ""))
    )


def _tokenize_count(prompt: str, model: str) -> int:
    """Lazy tokenizer load — only when LongBench path is actually used."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    return len(tok(prompt, add_special_tokens=False)["input_ids"])


def _load_samples_by_indices(
    ds,
    indices_env: str,
    *,
    template: str,
    model: str,
    task: str,
) -> list[Sample]:
    """Pick specific row indices from the LongBench-v2 split, in env order.

    The index space here is the FILTERED list (rows with domain/sub_domain ==
    `task`), so callers can say "the 3rd, 17th, 42nd SDQA item" without
    knowing the row's position in the global 503-item dataset.
    """
    items = [
        it
        for it in ds
        if it.get("domain") == task or it.get("sub_domain") == task
    ]
    if not items:
        valid_domains = sorted({it.get("domain") for it in ds})
        raise RuntimeError(
            f"LongBench-v2 has no items with domain/sub_domain == "
            f"{task!r}. Valid domains: {valid_domains}."
        )
    indices = [int(x) for x in indices_env.split(",") if x.strip()]
    out: list[Sample] = []
    for ord_pos, idx in enumerate(indices):
        if not (0 <= idx < len(items)):
            raise RuntimeError(
                f"index {idx} out of range for task {task!r} "
                f"(len={len(items)})"
            )
        item = items[idx]
        prompt = _build_longbench_prompt(item, template)
        tokens = _tokenize_count(prompt, model=model)
        # Synthetic bucket label so downstream summary still groups sensibly.
        bucket = bucket_for_tokens(min(tokens, (1 << 30) - 1))
        out.append(
            Sample(
                sample_id=f"longbench/{task}/idx{idx}/pos{ord_pos}",
                prompt=prompt,
                prompt_tokens=tokens,
                bucket=bucket,
            )
        )
    return out


def load_samples(
    spec_str: str,
    *,
    model: str = "meta-llama/Llama-3.2-3B-Instruct",
    longbench_full: bool = False,
) -> list[Sample]:
    """Top-level entry. Tries LongBench, falls back to synthetic.

    QUEST_MEM_PROBE_FORCE_SYNTHETIC=1 in the env forces the fallback path
    (handy for unit tests and for cluster runs without HF Hub access).

    QUEST_MEM_PROBE_LONGBENCH_INDICES=12,34,56 in the env overrides the
    bucketing logic entirely and pulls those specific row indices from
    LongBench-v2 (still filtered by spec.task domain). Used when you need
    to hit precise token-length targets the bucket system can't express.

    longbench_full=True ignores the spec's `n=` cap and takes every item that
    falls into a requested bucket. Synthetic fallback ignores the flag.
    """
    spec = parse_spec(spec_str)
    if os.environ.get("QUEST_MEM_PROBE_FORCE_SYNTHETIC") == "1":
        print("[quest_memory_probe] forced synthetic workload", file=sys.stderr)
        return load_samples_synthetic(buckets=spec.buckets, n=spec.n)
    try:
        return _load_samples_longbench(spec, model=model, longbench_full=longbench_full)
    except Exception as e:
        print(
            f"[quest_memory_probe] WARN LongBench load failed ({e!r}); "
            "falling back to synthetic prompts",
            file=sys.stderr,
        )
        return load_samples_synthetic(buckets=spec.buckets, n=spec.n)


def _load_samples_longbench(
    spec: WorkloadSpec,
    *,
    model: str,
    longbench_full: bool = False,
) -> list[Sample]:
    template = _read_template("0shot.txt")
    ds = _load_dataset("THUDM/LongBench-v2", split="train")

    # Optional escape hatch: pin a list of row indices via env, bypassing
    # the bucket selection. Used to hit precise token-length targets that
    # the LongBench-v2 length labels (short/medium/long) can't express
    # — for example, picking exact 32k / 64k / 128k items for a
    # context-length sweep on Llama-3.2-3B (native 131072).
    indices_env = os.environ.get("QUEST_MEM_PROBE_LONGBENCH_INDICES")
    if indices_env:
        return _load_samples_by_indices(
            ds, indices_env, template=template, model=model, task=spec.task
        )

    # LongBench-v2 schema: domain ∈ {'Single-Document QA',
    # 'Multi-Document QA', 'Long In-context Learning',
    # 'Code Repository Understanding', 'Long-dialogue History Understanding',
    # 'Long Structured Data Understanding'}; length ∈ {'short','medium','long'}.
    # Accept either canonical domain or sub_domain. If task matches nothing,
    # fail fast — silently falling back to the full corpus tokenizes hundreds
    # of 100k+ token contexts and burns ~10 minutes per cfg.
    items = [
        it
        for it in ds
        if it.get("domain") == spec.task or it.get("sub_domain") == spec.task
    ]
    if not items:
        valid_domains = sorted({it.get("domain") for it in ds})
        raise RuntimeError(
            f"LongBench-v2 has no items with domain/sub_domain == "
            f"{spec.task!r}. Valid domains: {valid_domains}. "
            "Pass a real domain or set QUEST_MEM_PROBE_FORCE_SYNTHETIC=1."
        )

    # Pre-filter by LongBench-v2's own length field. v2 labels items as
    # short/medium/long; we trust those labels rather than re-bucketing on
    # tokenized lengths (v2 'short' includes items up to ~32k tokens, which
    # would land in our 'medium' bucket and leave 'short' empty). Items in
    # buckets the spec doesn't request are skipped here. xlong has no v2
    # equivalent, so xlong-only specs against LongBench yield 0 samples and
    # raise below — use synthetic for xlong workloads.
    requested_buckets = set(spec.buckets)
    items = [it for it in items if it.get("length") in requested_buckets]

    # When subsampling (longbench_full=False), pick the SHORTEST items per
    # bucket. v2's long bucket spans 167k–4M tokens; without this sort we'd
    # tokenize and try to prefill 4M-token Code-Repo dumps that no GPU can
    # hold. Char count is a monotonic proxy for tokens — sorting on chars
    # and tokenizing only the first N keeps cost bounded. longbench_full
    # still walks every item (callers asking for the full set accept the
    # cost).
    if not longbench_full:
        items = sorted(items, key=lambda it: len(it.get("context", "")))

    # Render + tokenize: tokenize is only for filling Sample.prompt_tokens
    # (informational; not used for bucketing).
    by_bucket: dict[str, list[Sample]] = {b: [] for b in spec.buckets}
    for idx, item in enumerate(items):
        if not longbench_full and all(len(v) >= spec.n for v in by_bucket.values()):
            break
        bucket = item.get("length")
        if bucket not in by_bucket:
            continue
        if not longbench_full and len(by_bucket[bucket]) >= spec.n:
            continue
        prompt = _build_longbench_prompt(item, template)
        tokens = _tokenize_count(prompt, model=model)
        by_bucket[bucket].append(
            Sample(
                sample_id=f"longbench/{spec.task}/{bucket}/{idx}",
                prompt=prompt,
                prompt_tokens=tokens,
                bucket=bucket,
            )
        )

    out: list[Sample] = []
    for bucket in spec.buckets:
        if not longbench_full and len(by_bucket[bucket]) < spec.n:
            raise RuntimeError(
                f"LongBench produced only {len(by_bucket[bucket])}/{spec.n} "
                f"samples for bucket={bucket!r} (task={spec.task!r}); "
                "spec is too tight for this dataset."
            )
        out.extend(by_bucket[bucket])
    return out

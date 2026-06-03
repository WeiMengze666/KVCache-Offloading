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


def load_samples(
    spec_str: str,
    *,
    model: str = "meta-llama/Llama-3.2-3B-Instruct",
    longbench_full: bool = False,
) -> list[Sample]:
    """Top-level entry. Tries LongBench, falls back to synthetic.

    QUEST_MEM_PROBE_FORCE_SYNTHETIC=1 in the env forces the fallback path
    (handy for unit tests and for cluster runs without HF Hub access).

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

    # Pre-filter by LongBench-v2's own length field. Our token-based buckets
    # may classify some items differently (v2 'short' can be 30k+ tokens),
    # but using v2 length as a candidate filter cuts tokenize work from ~503
    # items to <100. Skip when the spec asks for buckets v2 doesn't label
    # (e.g. xlong) or when --longbench-full is on.
    v2_length_for_buckets = {"short", "medium", "long"} & set(spec.buckets)
    if v2_length_for_buckets and not longbench_full:
        items = [it for it in items if it.get("length") in v2_length_for_buckets]

    # Render + tokenize. We do NOT keep all items in memory — bucket as we go.
    by_bucket: dict[str, list[Sample]] = {b: [] for b in spec.buckets}
    for idx, item in enumerate(items):
        if not longbench_full and all(len(v) >= spec.n for v in by_bucket.values()):
            break
        prompt = _build_longbench_prompt(item, template)
        tokens = _tokenize_count(prompt, model=model)
        try:
            bucket = bucket_for_tokens(tokens)
        except ValueError:
            continue
        if bucket not in by_bucket:
            continue
        if not longbench_full and len(by_bucket[bucket]) >= spec.n:
            continue
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

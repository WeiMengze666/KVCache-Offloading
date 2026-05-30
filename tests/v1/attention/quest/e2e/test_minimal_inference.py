# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E.1 #2: Quest with top_k=ALL ≈ dense FA on first-step logprobs.

Quest is enabled with a top_k larger than any plausible block count, so
select_blocks degenerates to picking *every* block. Output should be
numerically close to the dense FA path. We compare the top-N logprob
distribution at step 0 (greedy, max_tokens=1).

Cosine ≥ 0.999 is the contract; rationale lives in the spec under
§3.1 / §6 of `2026-05-30-vllm-quest-phase-e1-design.md`.
"""

from __future__ import annotations

import dataclasses
import math

import pytest

from vllm import SamplingParams

pytestmark = pytest.mark.real_model


_PROMPT = "The capital of France is"


def _logprob_dict_to_aligned_vectors(d_dense: dict, d_quest: dict):
    """Given two `{token_id: Logprob}` maps, return matched fp64 vectors over
    the *intersection* of their keys, sorted by token id for determinism.
    Logprobs are in log-space; we convert to probabilities for cosine.
    """
    common = sorted(set(d_dense) & set(d_quest))
    if not common:
        raise AssertionError("no overlap between dense and quest top-N tokens")
    dense_v = [math.exp(d_dense[t].logprob) for t in common]
    quest_v = [math.exp(d_quest[t].logprob) for t in common]
    return dense_v, quest_v


def _cosine(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb)


def test_quest_top_k_all_matches_dense_first_logprobs(
    dense_llm,
    baseline_quest_config,
    quest_llm_factory,
):
    cfg = dataclasses.replace(
        baseline_quest_config,
        top_k=10000,
        gpu_cache_blocks_per_seq=10240,
    )
    cfg.validate()
    quest_llm = quest_llm_factory(cfg)

    params = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20)
    out_dense = dense_llm.generate([_PROMPT], params, use_tqdm=False)
    out_quest = quest_llm.generate([_PROMPT], params, use_tqdm=False)

    dense_lp = out_dense[0].outputs[0].logprobs[0]
    quest_lp = out_quest[0].outputs[0].logprobs[0]

    dense_v, quest_v = _logprob_dict_to_aligned_vectors(dense_lp, quest_lp)
    cos = _cosine(dense_v, quest_v)
    assert cos >= 0.999, (
        f"dense vs quest(top_k=ALL) first-step logprob cosine={cos:.6f} "
        f"< 0.999 over {len(dense_v)} shared tokens"
    )

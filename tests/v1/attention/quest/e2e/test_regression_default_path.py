# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E.1 #1: default vLLM path runs end-to-end with Quest *not* enabled.

Three assertions:
  (a) generate() returns the expected number of tokens
  (b) sys.modules contains no `vllm.v1.attention.backends.quest.*` after the
      session-scoped dense_llm finishes setting up — i.e. constructing a
      non-Quest LLM does not import any Quest code path
  (c) two consecutive generate() calls on the same LLM instance return the
      same token-id sequence (deterministic within a single instance)
"""

from __future__ import annotations

import sys

import pytest

from vllm import SamplingParams

pytestmark = pytest.mark.real_model


def test_default_path_generates(dense_llm):
    params = SamplingParams(temperature=0.0, max_tokens=16)
    out = dense_llm.generate(["The capital of France is"], params, use_tqdm=False)
    assert len(out) == 1
    token_ids = list(out[0].outputs[0].token_ids)
    assert len(token_ids) == 16
    assert all(isinstance(t, int) for t in token_ids)


def test_default_path_does_not_import_quest_modules(dense_llm):
    leaked = [
        m for m in sys.modules if m.startswith("vllm.v1.attention.backends.quest")
    ]
    assert leaked == [], "default path leaked Quest modules: " + ", ".join(leaked)


def test_default_path_is_deterministic_within_instance(dense_llm):
    prompt = "The capital of France is"
    params = SamplingParams(temperature=0.0, max_tokens=8)
    out_a = dense_llm.generate([prompt], params, use_tqdm=False)
    out_b = dense_llm.generate([prompt], params, use_tqdm=False)
    ids_a = list(out_a[0].outputs[0].token_ids)
    ids_b = list(out_b[0].outputs[0].token_ids)
    assert ids_a == ids_b, f"non-deterministic within same LLM: {ids_a} vs {ids_b}"

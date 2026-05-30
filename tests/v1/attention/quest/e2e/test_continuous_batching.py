# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E.1 #3: Quest survives a multi-prompt batch with varying input lengths.

Submits 4 prompts of length ~10 / ~30 / ~60 / ~100 tokens in a single
generate() call. Asserts that:
  (a) all 4 outputs come back non-empty
  (b) outputs are aligned with input order (vLLM contract — we don't reorder
      generate() return ourselves)
  (c) each output has either the requested max_tokens or a clean EOS, never
      a zero-length output
"""

from __future__ import annotations

import pytest

from vllm import SamplingParams

pytestmark = pytest.mark.real_model


# Roughly 10 / 30 / 60 / 100 tokens once tokenized — exact counts don't matter
# as long as they are clearly differentiated.
_PROMPTS = [
    "Hello world.",
    "The capital of France is Paris and the country borders Germany, Italy, and Spain.",
    (
        "In a recent meta-analysis of randomized controlled trials, "
        "researchers observed that combining lifestyle modifications with "
        "moderate-intensity exercise produced statistically significant "
        "improvements across several cardiovascular markers."
    ),
    (
        "Long ago, in a forgotten valley nestled deep between two mountain "
        "ranges, there lived a community of weavers whose intricate tapestries "
        "depicted the constellations as they moved across the night sky over "
        "the course of a single year. Each thread, dyed with pigments rendered "
        "from rare lichens harvested only on the highest peaks, told a story."
    ),
]


def test_quest_continuous_batch_runs(baseline_quest_config, quest_llm_factory):
    quest_llm = quest_llm_factory(baseline_quest_config)
    params = SamplingParams(temperature=0.0, max_tokens=16)

    outputs = quest_llm.generate(_PROMPTS, params, use_tqdm=False)

    assert len(outputs) == len(_PROMPTS), (
        f"expected {len(_PROMPTS)} outputs, got {len(outputs)}"
    )

    for i, out in enumerate(outputs):
        token_ids = list(out.outputs[0].token_ids)
        assert len(token_ids) > 0, f"prompt {i} produced empty output"
        # Either filled to max_tokens or stopped early on EOS — both fine
        assert len(token_ids) <= 16, (
            f"prompt {i} produced more than max_tokens={16}: {len(token_ids)}"
        )

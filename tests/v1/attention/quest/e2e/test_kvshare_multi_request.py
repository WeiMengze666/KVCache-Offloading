# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage 2C-v2 regression: kvshare must GC stale per-request state between
requests.

Reproduces a real crash on ``experiment-2c`` HEAD (commit ``def5a97ce``).
Without cross-request GC the second ``generate()`` call dies with one of two
symptoms (depending on which leaked structure overflows first):

  1. ``ValueError: begin_evict requires ON_GPU at (layer,block), got ON_CPU``
     (``residency.py:46``, called from ``prefill_ingest_kvshare:455``) —
     residency rows mutated to ``ON_CPU`` by request A's spilled prompt
     blocks were never reset.

  2. ``AssertionError: kvshare prefill ingest must not overflow the arena``
     (``tier_manager.py:426``, called from ``prefill_ingest_kvshare:425``) —
     ``_slot_map`` entries keyed ``(seq_A, *)`` still occupy the arena,
     leaving no room for request B's keeps; the LRU then tries to evict
     a stale entry and the "no overflow" guard fires.

Both are the same root cause: ``prefill_ingest_kvshare`` mutates per-(layer,
block) residency, the ``_slot_map`` LRU, and ``_cpu_slots``/``_host_slots``,
but the cross-request GC (``_active_seqs`` diff → ``tm.free_request(old_seq)``
→ release slots + reset residency back to ``ON_GPU``) only runs from the
decode-time helper (``notify_filled_blocks_after_decode`` /
``kvshare_decode_write``). When request #2 starts a fresh prefill before any
decode step has run, the kvshare prefill path hits leftover state from
request #1 and crashes.

The 2A path's analogous bug was fixed in ``e962bd3b5`` ("use stable request
ids to prevent cross-request KV leak"), but that fix wired GC into the decode
helper only — fine for 2A (whose prefill helper is summary-only and never
mutates LRU/CPU/residency), wrong for 2C-v2 (whose prefill helper *does*
mutate them). This test guards the 2C-v2 prefill GC.

Concurrency=1 by design (matches the rest of the kvshare path).
"""
from __future__ import annotations

import dataclasses

import pytest

from vllm import SamplingParams

pytestmark = pytest.mark.real_model

# Long enough that the prompt's full-block count overflows the per-seq GPU
# arena (gpu_cache_blocks_per_seq=2 below) — that forces
# prefill_ingest_kvshare to spill prompt blocks (and therefore mutate
# residency to ON_CPU). Without overflow there is no spill and the bug
# doesn't fire. The conftest fixture builds the LLM with max_model_len=1024,
# so prompts must stay under that minus max_tokens; ~720 prompt tokens fits
# and gives 3 full block_size=256 blocks (overflows the cap=2 arena).
_LONG_PROMPT_A = (
    "In the spring of the year 1789, the assembly convened in Versailles "
    "to address grievances that had accumulated over decades of fiscal "
    "mismanagement and shifting alliances among the nobility. "
) * 20  # ~720 tokens, 2-3 full blocks

_LONG_PROMPT_B = (
    "The Roman Empire reached its greatest territorial extent under "
    "Emperor Trajan in 117 AD, spanning from Britain in the northwest "
    "to Mesopotamia in the southeast. Roman engineering produced "
    "aqueducts, paved roads, and concrete structures whose remains "
    "still stand. "
) * 12  # ~720 tokens, 2-3 full blocks


def test_quest_kvshare_survives_back_to_back_requests(
    baseline_quest_config,
    quest_llm_factory,
):
    """Two sequential generate() calls under footprint_kvshare must both
    succeed. The second call exercises the cross-request residency-reset
    contract: stale ON_CPU rows from the first request must be cleared before
    the second prefill calls ``begin_evict`` on overlapping logical block ids.
    """
    cfg = dataclasses.replace(
        baseline_quest_config,
        footprint_kvshare=True,
        # Tight pool so each prompt's full-block count > capacity → spill
        # fires → residency is mutated to ON_CPU → cross-request GC matters.
        # ~720-token prompt → 2-3 full blocks of block_size=256, so cap=2
        # forces at least one block to spill (and top_k <= cap-1 = 1).
        gpu_cache_blocks_per_seq=2,
        top_k=1,
    )
    cfg.validate()
    quest_llm = quest_llm_factory(cfg)

    params = SamplingParams(temperature=0.0, max_tokens=8, seed=1234)

    # First request — establishes ON_CPU residency for spilled prompt blocks.
    out_a = quest_llm.generate([_LONG_PROMPT_A], params, use_tqdm=False)
    assert len(out_a) == 1
    text_a = out_a[0].outputs[0].text
    tokens_a = list(out_a[0].outputs[0].token_ids)
    assert len(tokens_a) > 0, "first request produced no tokens"

    # Second request — today this crashes with `begin_evict requires ON_GPU,
    # got ON_CPU` from the kvshare prefill helper because the residency rows
    # mutated by request A's prefill_ingest_kvshare were never reset.
    out_b = quest_llm.generate([_LONG_PROMPT_B], params, use_tqdm=False)
    assert len(out_b) == 1
    text_b = out_b[0].outputs[0].text
    tokens_b = list(out_b[0].outputs[0].token_ids)
    assert len(tokens_b) > 0, "second request produced no tokens"

    # Sanity: the two prompts are different content, so greedy decoding under
    # the same seed must produce different outputs. Without GC, request B
    # would either crash (current bug) or, in a hypothetical bypass, inherit
    # request A's KV state and produce A's output.
    assert text_a != text_b, (
        f"both requests produced identical text {text_a!r} — request B may "
        f"have inherited stale state from request A"
    )

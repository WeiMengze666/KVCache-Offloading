# SPDX-License-Identifier: Apache-2.0
"""Triton selection vs. torch reference."""
from __future__ import annotations

import pytest
import torch


@pytest.fixture
def cuda():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")


@pytest.mark.parametrize(
    "H_kv, G, D, B, top_k",
    [
        (8, 4, 128, 64, 8),
        (8, 4, 128, 256, 32),
        (4, 8, 64, 128, 16),
        (32, 1, 128, 512, 64),
        (8, 4, 128, 64, 64),  # boundary: top_k == B
    ],
)
def test_triton_matches_torch_topk_set(cuda, H_kv, G, D, B, top_k):
    torch.manual_seed(42)
    from vllm.v1.attention.ops.quest_selection_torch import (
        quest_selection_torch,
    )
    from vllm.v1.attention.ops.quest_selection_triton import (
        quest_selection_triton,
    )

    query = torch.randn(H_kv * G, D, dtype=torch.float16, device="cuda")
    summary = torch.randn(B, 2, H_kv, D, dtype=torch.float16, device="cuda")
    cand = torch.arange(B, dtype=torch.int32, device="cuda")
    ref = quest_selection_torch(
        query=query, block_summary=summary, candidate_ids=cand,
        num_kv_groups=G, top_k=top_k,
    )
    got = quest_selection_triton(
        query=query, block_summary=summary, candidate_ids=cand,
        num_kv_groups=G, top_k=top_k,
    )
    # Ignore tie-break order; compare sets.
    assert set(got.cpu().tolist()) == set(ref.cpu().tolist())


def test_triton_subset_candidates(cuda):
    from vllm.v1.attention.ops.quest_selection_torch import (
        quest_selection_torch,
    )
    from vllm.v1.attention.ops.quest_selection_triton import (
        quest_selection_triton,
    )

    H_kv, G, D = 4, 2, 64
    summary = torch.randn(32, 2, H_kv, D, dtype=torch.float16, device="cuda")
    query = torch.randn(H_kv * G, D, dtype=torch.float16, device="cuda")
    cand = torch.tensor([2, 5, 7, 9, 11, 13, 15, 17],
                        dtype=torch.int32, device="cuda")
    ref = quest_selection_torch(
        query=query, block_summary=summary, candidate_ids=cand,
        num_kv_groups=G, top_k=3,
    )
    got = quest_selection_triton(
        query=query, block_summary=summary, candidate_ids=cand,
        num_kv_groups=G, top_k=3,
    )
    assert set(got.cpu().tolist()) == set(ref.cpu().tolist())

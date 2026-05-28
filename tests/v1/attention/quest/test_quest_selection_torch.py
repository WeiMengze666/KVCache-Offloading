# SPDX-License-Identifier: Apache-2.0
"""Reference (PyTorch) Quest selection: brute-force formula equivalence."""
from __future__ import annotations

import pytest
import torch


def _ref_score(query, summary):
    """Brute-force score: returns [num_candidates] scalar per block.

    query   : [num_kv_heads * G, head_size]
    summary : [num_candidates, 2, num_kv_heads, head_size]
    """
    G = query.shape[0] // summary.shape[2]
    H_kv = summary.shape[2]
    D = summary.shape[3]
    q = query.view(H_kv, G, D).float()
    k_max = summary[:, 0].float()  # [B, H_kv, D]
    k_min = summary[:, 1].float()
    # broadcast q over B
    qm = q.unsqueeze(0)            # [1, H_kv, G, D]
    km = k_max.unsqueeze(2)        # [B, H_kv, 1, D]
    kn = k_min.unsqueeze(2)
    elem = torch.maximum(qm * km, qm * kn)  # [B, H_kv, G, D]
    return elem.sum(dim=(1, 2, 3))


@pytest.fixture
def cuda():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")


def test_select_topk_matches_brute_force_topk(cuda):
    from vllm.v1.attention.ops.quest_selection_torch import (
        quest_selection_torch,
    )

    torch.manual_seed(0)
    H_kv, G, D = 8, 4, 128
    B = 64
    query = torch.randn(H_kv * G, D, dtype=torch.float16, device="cuda")
    summary = torch.randn(B, 2, H_kv, D, dtype=torch.float16, device="cuda")
    candidate_ids = torch.arange(B, dtype=torch.int32, device="cuda")
    top_k = 8

    got = quest_selection_torch(
        query=query,
        block_summary=summary,
        candidate_ids=candidate_ids,
        num_kv_groups=G,
        top_k=top_k,
    )
    assert got.shape == (top_k,)

    # Reference path: full score, top-k by score; ignore tie-break order.
    ref_scores = _ref_score(query, summary)
    ref_top = torch.topk(ref_scores, k=top_k, largest=True, sorted=False)[1]
    assert set(got.cpu().tolist()) == set(ref_top.cpu().tolist())


def test_candidate_ids_subset(cuda):
    """Selection must come from candidate_ids, not the global summary index."""
    from vllm.v1.attention.ops.quest_selection_torch import (
        quest_selection_torch,
    )

    torch.manual_seed(1)
    H_kv, G, D = 2, 2, 32
    B_total = 16
    summary = torch.randn(B_total, 2, H_kv, D,
                          dtype=torch.float16, device="cuda")
    query = torch.randn(H_kv * G, D, dtype=torch.float16, device="cuda")

    # Only ids 4-9 are candidates.
    candidate_ids = torch.tensor([4, 5, 6, 7, 8, 9],
                                 dtype=torch.int32, device="cuda")
    got = quest_selection_torch(
        query=query,
        block_summary=summary,
        candidate_ids=candidate_ids,
        num_kv_groups=G,
        top_k=3,
    )
    assert all(int(i) in {4, 5, 6, 7, 8, 9} for i in got.cpu().tolist())


def test_top_k_equals_num_candidates_returns_all(cuda):
    from vllm.v1.attention.ops.quest_selection_torch import (
        quest_selection_torch,
    )

    H_kv, G, D = 1, 1, 8
    summary = torch.randn(4, 2, H_kv, D, dtype=torch.float16, device="cuda")
    query = torch.randn(H_kv * G, D, dtype=torch.float16, device="cuda")
    cand = torch.arange(4, dtype=torch.int32, device="cuda")
    got = quest_selection_torch(
        query=query, block_summary=summary, candidate_ids=cand,
        num_kv_groups=G, top_k=4,
    )
    assert set(got.cpu().tolist()) == {0, 1, 2, 3}


def test_top_k_greater_than_num_candidates_raises(cuda):
    from vllm.v1.attention.ops.quest_selection_torch import (
        quest_selection_torch,
    )

    H_kv, G, D = 1, 1, 8
    summary = torch.randn(4, 2, H_kv, D, dtype=torch.float16, device="cuda")
    query = torch.randn(H_kv * G, D, dtype=torch.float16, device="cuda")
    cand = torch.arange(4, dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError, match="top_k"):
        quest_selection_torch(
            query=query, block_summary=summary, candidate_ids=cand,
            num_kv_groups=G, top_k=5,
        )

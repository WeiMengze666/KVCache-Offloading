# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA selection kernel vs. torch oracle (fp16 + bf16)."""

from __future__ import annotations

import pytest
import torch


@pytest.fixture(scope="module")
def cuda_and_C():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    try:
        import vllm._C  # noqa: F401
    except ImportError as exc:
        pytest.skip(f"vllm._C not built on this host: {exc}")


@pytest.mark.parametrize(
    "dtype",
    [torch.float16, torch.bfloat16],
    ids=["fp16", "bf16"],
)
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
def test_cuda_matches_torch_topk_set(
    cuda_and_C,
    H_kv,
    G,
    D,
    B,
    top_k,
    dtype,
):
    torch.manual_seed(42)
    from vllm.v1.attention.ops.quest_selection_torch import (
        quest_selection_torch,
    )
    from vllm.v1.attention.ops.quest_selection_cuda import (
        quest_selection_cuda,
    )

    query = torch.randn(H_kv * G, D, dtype=dtype, device="cuda")
    summary = torch.randn(B, 2, H_kv, D, dtype=dtype, device="cuda")
    cand = torch.arange(B, dtype=torch.int32, device="cuda")
    ref = quest_selection_torch(
        query=query,
        block_summary=summary,
        candidate_ids=cand,
        num_kv_groups=G,
        top_k=top_k,
    )
    got = quest_selection_cuda(
        query=query,
        block_summary=summary,
        candidate_ids=cand,
        num_kv_groups=G,
        top_k=top_k,
    )
    # Ignore tie-break order; compare sets.
    assert set(got.cpu().tolist()) == set(ref.cpu().tolist()), (
        f"cuda {sorted(got.cpu().tolist())} != torch {sorted(ref.cpu().tolist())}"
    )


def test_cuda_subset_candidates(cuda_and_C):
    """Selection must come from candidate_ids, not the global summary index."""
    from vllm.v1.attention.ops.quest_selection_torch import (
        quest_selection_torch,
    )
    from vllm.v1.attention.ops.quest_selection_cuda import (
        quest_selection_cuda,
    )

    H_kv, G, D = 4, 2, 64
    summary = torch.randn(32, 2, H_kv, D, dtype=torch.float16, device="cuda")
    query = torch.randn(H_kv * G, D, dtype=torch.float16, device="cuda")
    cand = torch.tensor([2, 5, 7, 9, 11, 13, 15, 17], dtype=torch.int32, device="cuda")
    ref = quest_selection_torch(
        query=query,
        block_summary=summary,
        candidate_ids=cand,
        num_kv_groups=G,
        top_k=3,
    )
    got = quest_selection_cuda(
        query=query,
        block_summary=summary,
        candidate_ids=cand,
        num_kv_groups=G,
        top_k=3,
    )
    assert set(got.cpu().tolist()) == set(ref.cpu().tolist())


def test_cuda_top_k_greater_than_num_candidates_raises(cuda_and_C):
    from vllm.v1.attention.ops.quest_selection_cuda import (
        quest_selection_cuda,
    )

    H_kv, G, D = 1, 1, 8
    summary = torch.randn(4, 2, H_kv, D, dtype=torch.float16, device="cuda")
    query = torch.randn(H_kv * G, D, dtype=torch.float16, device="cuda")
    cand = torch.arange(4, dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError, match="top_k"):
        quest_selection_cuda(
            query=query,
            block_summary=summary,
            candidate_ids=cand,
            num_kv_groups=G,
            top_k=5,
        )


def test_cuda_raises_when_C_missing(monkeypatch):
    """When _C failed to import, calling the wrapper raises RuntimeError
    with no silent fallback. Spec §10 R8."""
    import vllm.v1.attention.ops.quest_selection_cuda as mod

    monkeypatch.setattr(
        mod,
        "_C_IMPORT_ERROR",
        ImportError("simulated: vllm._C not built"),
    )
    with pytest.raises(RuntimeError, match="vllm._C"):
        mod.quest_selection_cuda(
            query=torch.empty(1, 1),
            block_summary=torch.empty(1, 2, 1, 1),
            candidate_ids=torch.empty(1, dtype=torch.int32),
            num_kv_groups=1,
            top_k=1,
        )

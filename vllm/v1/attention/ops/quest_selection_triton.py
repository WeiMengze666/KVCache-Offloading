# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton selection kernel — the Phase B middle stage.

Adapted from the legacy implementation
(transformers/.../submudule.py:_quest_selection_kernel) but rewritten with
the Phase B public surface: input is the global `block_summary` tensor
plus a `candidate_ids` index list; it does NOT assume per-seq contiguous
layout.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _quest_score_kernel(
    q_ptr, summary_ptr, cand_ptr, scores_ptr,
    q_stride_h, q_stride_d,
    s_stride_b, s_stride_2, s_stride_h, s_stride_d,
    G: tl.constexpr,
    H_KV: tl.constexpr,
    D: tl.constexpr,
):
    pid = tl.program_id(0)
    block_id = tl.load(cand_ptr + pid)
    acc = tl.zeros((D,), dtype=tl.float32)

    for h in tl.static_range(H_KV):
        max_off = (
            block_id * s_stride_b
            + 0 * s_stride_2
            + h * s_stride_h
            + tl.arange(0, D) * s_stride_d
        )
        min_off = (
            block_id * s_stride_b
            + 1 * s_stride_2
            + h * s_stride_h
            + tl.arange(0, D) * s_stride_d
        )
        k_max = tl.load(summary_ptr + max_off).to(tl.float32)
        k_min = tl.load(summary_ptr + min_off).to(tl.float32)
        for g in tl.static_range(G):
            q_off = (h * G + g) * q_stride_h + tl.arange(0, D) * q_stride_d
            q = tl.load(q_ptr + q_off).to(tl.float32)
            acc += tl.maximum(q * k_max, q * k_min)
    tl.store(scores_ptr + pid, tl.sum(acc))


def quest_selection_triton(
    *,
    query: torch.Tensor,
    block_summary: torch.Tensor,
    candidate_ids: torch.Tensor,
    num_kv_groups: int,
    top_k: int,
) -> torch.Tensor:
    """Same contract as quest_selection_torch."""
    num_candidates = candidate_ids.shape[0]
    if top_k > num_candidates:
        raise ValueError(
            f"top_k ({top_k}) > num_candidates ({num_candidates})"
        )

    H_kv = block_summary.shape[2]
    D = block_summary.shape[3]
    G = num_kv_groups

    scores = torch.empty(num_candidates,
                         dtype=torch.float32, device=query.device)
    grid = (num_candidates,)
    _quest_score_kernel[grid](
        q_ptr=query,
        summary_ptr=block_summary,
        cand_ptr=candidate_ids,
        scores_ptr=scores,
        q_stride_h=query.stride(0),
        q_stride_d=query.stride(1),
        s_stride_b=block_summary.stride(0),
        s_stride_2=block_summary.stride(1),
        s_stride_h=block_summary.stride(2),
        s_stride_d=block_summary.stride(3),
        G=G, H_KV=H_kv, D=D,
    )
    top_local = torch.topk(scores, k=top_k, largest=True, sorted=False)[1]
    return candidate_ids.index_select(0, top_local)

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reference (PyTorch) Quest block selection.

This is the **oracle** for the Triton (Phase B) and CUDA (Phase D)
implementations. Speed is irrelevant; correctness is everything.
"""
from __future__ import annotations

import torch


def quest_selection_torch(
    *,
    query: torch.Tensor,
    block_summary: torch.Tensor,
    candidate_ids: torch.Tensor,
    num_kv_groups: int,
    top_k: int,
) -> torch.Tensor:
    """Return the top-k block ids out of `candidate_ids` by Quest score.

    Args:
      query:          [num_kv_heads * num_kv_groups, head_size], decode
                      query (last token).
      block_summary:  [num_blocks_total, 2, num_kv_heads, head_size]
                      summary[*, 0] = K amax, summary[*, 1] = K amin.
      candidate_ids:  [num_candidates] global block ids in the summary.
      num_kv_groups:  GQA repeat factor (num_heads // num_kv_heads).
      top_k:          number of blocks to return (<= num_candidates).

    Returns:
      Tensor of shape [top_k] with block ids drawn from `candidate_ids`.
    """
    num_candidates = candidate_ids.shape[0]
    if top_k > num_candidates:
        raise ValueError(
            f"top_k ({top_k}) > num_candidates ({num_candidates})"
        )

    G = num_kv_groups
    H_kv = block_summary.shape[2]
    D = block_summary.shape[3]
    summary_sel = block_summary.index_select(0, candidate_ids.long())
    # Promote to fp32 for stable reduction; rationale matches the Triton
    # kernel in transformers/.../submudule.py:_quest_selection_kernel.
    q = query.view(H_kv, G, D).float()                    # [H_kv, G, D]
    k_max = summary_sel[:, 0].float()                     # [B, H_kv, D]
    k_min = summary_sel[:, 1].float()
    qm = q.unsqueeze(0)                                   # [1, H_kv, G, D]
    km = k_max.unsqueeze(2)                               # [B, H_kv, 1, D]
    kn = k_min.unsqueeze(2)
    elem = torch.maximum(qm * km, qm * kn)                # [B, H_kv, G, D]
    scores = elem.sum(dim=(1, 2, 3))                      # [B]
    top_local = torch.topk(scores, k=top_k, largest=True, sorted=False)[1]
    return candidate_ids.index_select(0, top_local)

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase D: CUDA implementation of Quest block selection.

This is a Task 3 stub — the real wrapper lands in Task 6 once the C++
op is registered. The stub raises NotImplementedError to make
accidental Task-3-era usage loud.
"""

from __future__ import annotations

import torch


def quest_selection_cuda(
    *,
    query: torch.Tensor,
    block_summary: torch.Tensor,
    candidate_ids: torch.Tensor,
    num_kv_groups: int,
    top_k: int,
) -> torch.Tensor:
    raise NotImplementedError(
        "quest_selection_cuda is a Phase D stub; implementation lands in Task 6."
    )

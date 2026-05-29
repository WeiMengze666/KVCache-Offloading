# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase D: CUDA implementation of Quest block selection.

Same public contract as quest_selection_torch / quest_selection_triton:
the wrapper validates inputs, allocates the per-candidate fp32 scores
tensor, calls the registered C++ op `_C.quest_score`, runs torch.topk
from Python, and returns the selected block ids drawn from
candidate_ids.

When `vllm._C` failed to import on this host, the wrapper raises a
single clean `RuntimeError` on its first call. There is no silent
fallback to torch / triton — caller asked explicitly for cuda
(spec §10 R8).
"""

from __future__ import annotations

import torch

try:
    import vllm._C  # noqa: F401  -- side effect: registers torch.ops._C.*

    _C_IMPORT_ERROR: BaseException | None = None
except ImportError as exc:
    _C_IMPORT_ERROR = exc


def quest_selection_cuda(
    *,
    query: torch.Tensor,
    block_summary: torch.Tensor,
    candidate_ids: torch.Tensor,
    num_kv_groups: int,
    top_k: int,
) -> torch.Tensor:
    """CUDA Quest block selection. Same contract as quest_selection_torch.

    Raises:
      RuntimeError: when vllm._C is not built on this host. Caller must
                    have selection_impl="cuda" explicitly to reach this
                    point; this error means the build is missing.
      ValueError:   when top_k > num_candidates.
    """
    if _C_IMPORT_ERROR is not None:
        raise RuntimeError(
            "quest_selection_cuda requires vllm._C, which failed to "
            "import on this host: "
            f"{type(_C_IMPORT_ERROR).__name__}: {_C_IMPORT_ERROR}. "
            "Rebuild vllm with `uv pip install -e . --torch-backend=auto`,"
            " or set QuestConfig.selection_impl='triton' / 'torch'."
        )

    num_candidates = candidate_ids.shape[0]
    if top_k > num_candidates:
        raise ValueError(f"top_k ({top_k}) > num_candidates ({num_candidates})")

    scores = torch.empty(
        num_candidates,
        dtype=torch.float32,
        device=query.device,
    )
    torch.ops._C.quest_score(
        query,
        block_summary,
        candidate_ids,
        scores,
        int(num_kv_groups),
    )
    top_local = torch.topk(scores, k=top_k, largest=True, sorted=False)[1]
    return candidate_ids.index_select(0, top_local)

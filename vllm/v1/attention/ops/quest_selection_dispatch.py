# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase D: dispatch table for Quest block-selection implementations.

The single runtime callsite (`run_sparse_decode` in
`vllm/v1/attention/backends/quest/impl_helpers.py`) calls
`_resolve_selection_callable(quest_config.selection_impl)` once at engine
init (via `bind_runtime`) and stashes the resolved callable on each Quest
layer. Per-step dispatch is then a single attribute read.

The "cuda" branch lazy-imports `quest_selection_cuda` so that this module
remains importable on hosts where `vllm._C` did not build. Calling the
returned cuda callable on such a host raises `RuntimeError` from the
wrapper itself — this dispatcher does NOT silently fall back to torch
or triton (spec §10 R8).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import torch

SelectionImpl = Literal["torch", "triton", "cuda"]


def _resolve_selection_callable(
    impl: SelectionImpl,
) -> Callable[..., torch.Tensor]:
    """Return the public Python wrapper for the requested impl.

    Args:
      impl: one of "torch", "triton", "cuda".

    Returns:
      The selection callable. Same signature across all three impls:
        fn(*, query, block_summary, candidate_ids, num_kv_groups, top_k)
        -> top_block_ids: torch.Tensor[top_k]

    Raises:
      ValueError: when `impl` is not one of the three accepted strings.
    """
    if impl == "torch":
        from vllm.v1.attention.ops.quest_selection_torch import (
            quest_selection_torch,
        )

        return quest_selection_torch
    if impl == "triton":
        from vllm.v1.attention.ops.quest_selection_triton import (
            quest_selection_triton,
        )

        return quest_selection_triton
    if impl == "cuda":
        from vllm.v1.attention.ops.quest_selection_cuda import (
            quest_selection_cuda,
        )

        return quest_selection_cuda
    raise ValueError(
        f"selection_impl must be 'torch', 'triton', or 'cuda', got {impl!r}"
    )

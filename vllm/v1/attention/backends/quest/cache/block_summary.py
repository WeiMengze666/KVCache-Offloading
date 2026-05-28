# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-block summary store (max/min along the sequence axis)."""
from __future__ import annotations

import torch


class BlockSummaryStore:
    """Holds [num_layers, max_blocks, 2, num_kv_heads, head_size] tensor.

    `summary[L, B, 0]` is `amax_over_block(K)`, `summary[L, B, 1]` is
    `amin_over_block(K)`. Used by quest_selection as the only visible part
    of evicted blocks.
    """

    def __init__(
        self,
        *,
        num_layers: int,
        max_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        dtype: torch.dtype,
        device: str | torch.device = "cuda",
    ) -> None:
        if num_layers <= 0:
            raise ValueError(f"num_layers must be > 0, got {num_layers}")
        if max_blocks <= 0:
            raise ValueError(f"max_blocks must be > 0, got {max_blocks}")

        self.num_layers = num_layers
        self.max_blocks = max_blocks
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size

        self.summary = torch.zeros(
            (num_layers, max_blocks, 2, num_kv_heads, head_size),
            dtype=dtype, device=device,
        )

    def on_block_filled(
        self,
        layer_idx: int,
        block_id: int,
        k_block: torch.Tensor,
    ) -> None:
        """k_block shape: [block_size, num_kv_heads, head_size]."""
        if k_block.shape != (
            self.block_size,
            self.num_kv_heads,
            self.head_size,
        ):
            raise ValueError(
                f"k_block shape {tuple(k_block.shape)} != "
                f"({self.block_size}, {self.num_kv_heads}, {self.head_size})"
            )
        self.summary[layer_idx, block_id, 0] = k_block.amax(dim=0)
        self.summary[layer_idx, block_id, 1] = k_block.amin(dim=0)

    def gather(
        self,
        layer_idx: int,
        block_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Returns summaries indexed by `block_ids` in order.

        Shape: [len(block_ids), 2, num_kv_heads, head_size].
        """
        return self.summary[layer_idx].index_select(0, block_ids.long())

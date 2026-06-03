# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-block summary store (Quest min/max or ArkVale cuboid-mean digest)."""

from __future__ import annotations

from typing import Literal

import torch

DigestMode = Literal["quest_minmax", "arkvale_cuboid_mean"]


class BlockSummaryStore:
    """Holds [num_layers, max_blocks, 2, num_kv_heads, head_size] tensor.

    `summary[L, B, 0]` and `summary[L, B, 1]` are the two digest endpoints
    used by quest_selection. The formula that fills them depends on
    `digest_mode`:
      - 'quest_minmax' (default): slot 0 = K amax, slot 1 = K amin
      - 'arkvale_cuboid_mean': slot 0 = center + radius_mean,
                                slot 1 = center - radius_mean,
        where center = (true_max + true_min)/2 and
        radius_mean = mean(|K - center|) over the block-token axis.

    Downstream selection ops are agnostic to the formula — they read
    only the two slot values.
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
        digest_mode: DigestMode = "quest_minmax",
    ) -> None:
        if num_layers <= 0:
            raise ValueError(f"num_layers must be > 0, got {num_layers}")
        if max_blocks <= 0:
            raise ValueError(f"max_blocks must be > 0, got {max_blocks}")
        if digest_mode not in ("quest_minmax", "arkvale_cuboid_mean"):
            raise ValueError(
                f"digest_mode must be 'quest_minmax' or "
                f"'arkvale_cuboid_mean', got {digest_mode!r}"
            )

        self.num_layers = num_layers
        self.max_blocks = max_blocks
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.digest_mode: DigestMode = digest_mode

        self.summary = torch.zeros(
            (num_layers, max_blocks, 2, num_kv_heads, head_size),
            dtype=dtype,
            device=device,
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
        if self.digest_mode == "arkvale_cuboid_mean":
            k_max = k_block.amax(dim=0)
            k_min = k_block.amin(dim=0)
            center = (k_max + k_min) * 0.5
            radius = (k_block - center).abs().mean(dim=0)
            self.summary[layer_idx, block_id, 0] = center + radius
            self.summary[layer_idx, block_id, 1] = center - radius
        else:  # 'quest_minmax'
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

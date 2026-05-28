# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pinned CPU pool for evicted Quest KV blocks."""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class CpuStoreStats:
    alloc_count: int = 0
    free_count: int = 0
    store_calls: int = 0
    load_calls: int = 0


class CpuKvBackingStore:
    """One pinned CPU tensor pair (K, V) of shape
    [num_layers, blocks_per_layer, block_size, num_kv_heads, head_size].

    Per-layer free-list manages slot allocation. Phase B uses synchronous
    copies; Phase C can subclass / wrap to make them stream-aware without
    changing the API surface.
    """

    def __init__(
        self,
        *,
        num_layers: int,
        blocks_per_layer: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        dtype: torch.dtype,
    ) -> None:
        if num_layers <= 0 or blocks_per_layer <= 0:
            raise ValueError(
                f"num_layers and blocks_per_layer must be positive; "
                f"got {num_layers}, {blocks_per_layer}"
            )
        shape = (num_layers, blocks_per_layer, block_size,
                 num_kv_heads, head_size)
        self.k = torch.empty(shape, dtype=dtype, pin_memory=True)
        self.v = torch.empty(shape, dtype=dtype, pin_memory=True)

        self.num_layers = num_layers
        self.blocks_per_layer = blocks_per_layer
        # Per-layer free list (LIFO so recent slots are warm in CPU cache).
        self._free: list[list[int]] = [
            list(reversed(range(blocks_per_layer)))
            for _ in range(num_layers)
        ]
        self._stats = CpuStoreStats()

    def alloc(self, layer_idx: int) -> int:
        free = self._free[layer_idx]
        if not free:
            raise RuntimeError(
                f"layer {layer_idx} CPU pool is full "
                f"({self.blocks_per_layer} blocks)"
            )
        self._stats.alloc_count += 1
        return free.pop()

    def free(self, layer_idx: int, cpu_slot: int) -> None:
        self._free[layer_idx].append(cpu_slot)
        self._stats.free_count += 1

    def store_block(
        self,
        layer_idx: int,
        cpu_slot: int,
        k_block: torch.Tensor,
        v_block: torch.Tensor,
    ) -> None:
        # Synchronous D2H (Phase B). Phase C will switch to non_blocking=True
        # with stream-aware ordering.
        self.k[layer_idx, cpu_slot].copy_(k_block, non_blocking=False)
        self.v[layer_idx, cpu_slot].copy_(v_block, non_blocking=False)
        self._stats.store_calls += 1

    def load_block(
        self,
        layer_idx: int,
        cpu_slot: int,
        k_dst: torch.Tensor,
        v_dst: torch.Tensor,
    ) -> None:
        # Synchronous H2D (Phase B).
        k_dst.copy_(self.k[layer_idx, cpu_slot], non_blocking=False)
        v_dst.copy_(self.v[layer_idx, cpu_slot], non_blocking=False)
        self._stats.load_calls += 1

    def stats(self) -> CpuStoreStats:
        return self._stats

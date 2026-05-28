# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per (layer, block_id) residency state machine."""
from __future__ import annotations

import enum

import numpy as np
import torch


class ResidencyState(enum.IntEnum):
    ON_GPU = 0
    EVICTING = 1   # D2H started, not yet observable in CPU pool
    ON_CPU = 2
    LOADING = 3    # H2D started, not yet observable in GPU slot


class BlockResidency:
    """Backend-private mirror of (layer, block_id) state.

    Stored as a 2D int8 tensor on host so transitions are cheap and the
    Python orchestration code can read them. A device-side mask is computed
    on demand for kernel-side filtering (e.g., is_on_gpu_mask used by
    quest_selection).
    """

    def __init__(self, num_layers: int, max_blocks: int) -> None:
        self.num_layers = num_layers
        self.max_blocks = max_blocks
        self._states = np.zeros(
            (num_layers, max_blocks), dtype=np.int8
        )  # ON_GPU = 0 by construction; will be overwritten on first write

    def state(self, layer_idx: int, block_id: int) -> ResidencyState:
        return ResidencyState(int(self._states[layer_idx, block_id]))

    def mark_on_gpu(self, layer_idx: int, block_id: int) -> None:
        self._states[layer_idx, block_id] = ResidencyState.ON_GPU

    def begin_evict(self, layer_idx: int, block_id: int) -> None:
        cur = self.state(layer_idx, block_id)
        if cur == ResidencyState.EVICTING:
            return
        if cur != ResidencyState.ON_GPU:
            raise ValueError(
                f"begin_evict requires ON_GPU at ({layer_idx},{block_id}), "
                f"got {cur.name}"
            )
        self._states[layer_idx, block_id] = ResidencyState.EVICTING

    def complete_evict(self, layer_idx: int, block_id: int) -> None:
        cur = self.state(layer_idx, block_id)
        if cur != ResidencyState.EVICTING:
            raise ValueError(
                f"cannot complete_evict at ({layer_idx},{block_id}): "
                f"current state {cur.name}"
            )
        self._states[layer_idx, block_id] = ResidencyState.ON_CPU

    def begin_load(self, layer_idx: int, block_id: int) -> None:
        cur = self.state(layer_idx, block_id)
        if cur != ResidencyState.ON_CPU:
            raise ValueError(
                f"begin_load requires ON_CPU at ({layer_idx},{block_id}), "
                f"got {cur.name}"
            )
        self._states[layer_idx, block_id] = ResidencyState.LOADING

    def complete_load(self, layer_idx: int, block_id: int) -> None:
        cur = self.state(layer_idx, block_id)
        if cur != ResidencyState.LOADING:
            raise ValueError(
                f"cannot complete_load at ({layer_idx},{block_id}): "
                f"current state {cur.name}"
            )
        self._states[layer_idx, block_id] = ResidencyState.ON_GPU

    def is_on_gpu_mask(
        self, layer_idx: int, block_ids: torch.Tensor
    ) -> torch.Tensor:
        """Returns a bool tensor on the same device as block_ids."""
        host_mask = (
            self._states[layer_idx, block_ids.cpu().numpy()]
            == int(ResidencyState.ON_GPU)
        )
        return torch.from_numpy(host_mask).to(block_ids.device)

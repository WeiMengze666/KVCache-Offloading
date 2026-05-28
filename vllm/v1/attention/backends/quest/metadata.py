# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Quest attention metadata + builder.

Phase B: QuestAttentionMetadata is now a real subclass that adds
- sparse_block_table: per-step top-k logical block ids per request
  (populated by impl just before flash_attn call)
- quest_layer_indices: maps global layer_idx -> quest layer slot, or
  -1 for full-KV layers
- is_full_kv_layer: bool tensor over global layer_idx
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionMetadata,
    FlashAttentionMetadataBuilder,
)


@dataclass
class QuestAttentionMetadata(FlashAttentionMetadata):
    sparse_block_table: torch.Tensor | None = None
    """Per-request top-k logical block ids (int32, shape (num_reqs, top_k)).

    None on prefill steps (full attention). Populated by
    QuestSparseOffloadImpl.forward right before the kernel call."""

    quest_layer_indices: torch.Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.int32),
    )
    """Maps global layer index -> quest slot (>=0) or -1 for full-KV."""

    is_full_kv_layer: torch.Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.bool),
    )
    """Bool tensor of length num_layers; True for full-KV layers."""


class QuestMetadataBuilder(FlashAttentionMetadataBuilder):
    """Delegates standard FA metadata then promotes to QuestAttentionMetadata.

    Quest-only fields are populated once at engine init (they are static
    over a model's lifetime) and copied into every metadata object.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._fa_builder = self  # Phase B: build via super().build
        self._quest_layer_indices: torch.Tensor | None = None

    def set_quest_layer_indices(self, indices: torch.Tensor) -> None:
        self._quest_layer_indices = indices.to(torch.int32)

    def build(self, *args, **kwargs) -> QuestAttentionMetadata:
        fa_md = super().build(*args, **kwargs)
        return self._promote(fa_md)

    def build_for_test(self) -> QuestAttentionMetadata:
        """Light path used by unit tests that don't have a real common
        metadata fixture set up. Phase B-only."""
        fa_md = self._fa_builder.build()
        return self._promote(fa_md)

    def _promote(
        self, fa_md: FlashAttentionMetadata
    ) -> QuestAttentionMetadata:
        idx = self._quest_layer_indices
        if idx is None:
            idx = torch.empty(0, dtype=torch.int32)
        is_full = (idx < 0)
        # Use only fields actually present on FlashAttentionMetadata to avoid
        # version skew breaking us.
        return QuestAttentionMetadata(
            **{f: getattr(fa_md, f)
               for f in fa_md.__dataclass_fields__},
            sparse_block_table=None,
            quest_layer_indices=idx,
            is_full_kv_layer=is_full,
        )

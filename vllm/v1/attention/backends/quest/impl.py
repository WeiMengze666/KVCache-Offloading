# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""QuestSparseOffloadImpl — Phase B real forward path."""
from __future__ import annotations

import torch

from vllm.v1.attention.backend import AttentionImpl, AttentionType
from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl


class QuestSparseOffloadImpl(AttentionImpl):
    """Forward strategy:
      - prefill (max_query_len > 1) OR layer is in full_kv_layers:
          delegate to FlashAttentionImpl as in Phase A.
      - decode of a Quest layer:
          1. write current K/V to GPU cache (reshape_and_cache_flash) —
             same as FA does, so this happens via the FA forward we still
             call for the cache update step.
          2. on each newly-completed block (slot_mapping spans a block
             boundary), tier_manager.on_block_filled.
          3. quest_selection over candidate_ids = ON_GPU + ON_CPU blocks.
          4. tier_manager.ensure_resident(top_ids) — sync H2D for missing.
          5. build sparse_block_table from top_ids -> physical slots.
          6. flash_attn_with_kvcache(block_table=sparse_block_table, ...).

    Phase B operates per layer; cross-layer state lives on the
    forward_context (set up by the worker once at engine init).
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
    ) -> None:
        self.kv_cache_dtype = kv_cache_dtype
        self._fa_impl = FlashAttentionImpl(
            num_heads=num_heads,
            head_size=head_size,
            scale=scale,
            num_kv_heads=num_kv_heads,
            alibi_slopes=alibi_slopes,
            sliding_window=sliding_window,
            kv_cache_dtype=kv_cache_dtype,
            logits_soft_cap=logits_soft_cap,
            attn_type=attn_type,
            kv_sharing_target_layer_name=kv_sharing_target_layer_name,
        )

    def forward(
        self,
        layer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        is_prefill = attn_metadata.max_query_len > 1
        if is_prefill or self._is_full_kv_layer(layer, attn_metadata):
            # Prefill always runs full attention (spec §1). Full-KV layers
            # always delegate, regardless of phase. Both share the standard
            # FA forward — including reshape_and_cache_flash for KV write.
            out = self._fa_impl.forward(
                layer, query, key, value, kv_cache, attn_metadata, output,
                output_scale=output_scale,
                output_block_scale=output_block_scale,
            )
            # During prefill, hand newly completed blocks to the tier manager
            # so the working set + summaries are up to date for the upcoming
            # decode steps.
            self._notify_filled_blocks_after_prefill(
                layer, key, value, attn_metadata,
            )
            return out

        # Decode of a Quest layer: write KV first via the standard helper,
        # then run the sparse path.
        self._write_kv_via_fa_helper(
            layer, key, value, kv_cache, attn_metadata,
        )
        self._notify_filled_blocks_after_decode(
            layer, kv_cache, attn_metadata,
        )
        return self._forward_sparse_decode(
            layer, query, kv_cache, attn_metadata, output,
        )

    # ----- private helpers (see Task 14 for full bodies) -----

    def _is_full_kv_layer(self, layer, attn_metadata) -> bool:
        idx = getattr(attn_metadata, "quest_layer_indices", None)
        if idx is None or idx.numel() == 0:
            return True  # safe default = behave like FA
        return bool(idx[layer.layer_idx].item() < 0)

    def _write_kv_via_fa_helper(self, layer, key, value, kv_cache, md):
        from vllm._custom_ops import reshape_and_cache_flash
        n = md.num_actual_tokens
        reshape_and_cache_flash(
            key[:n], value[:n],
            kv_cache[:, 0], kv_cache[:, 1],
            md.slot_mapping[:n],
            self.kv_cache_dtype,
            layer._k_scale, layer._v_scale,
        )

    def _notify_filled_blocks_after_prefill(self, layer, key, value, md):
        # Implementation in Task 14 hooks tier_manager via forward_context.
        from vllm.v1.attention.backends.quest.impl_helpers import (
            notify_filled_blocks_after_prefill,
        )
        notify_filled_blocks_after_prefill(layer, key, value, md)

    def _notify_filled_blocks_after_decode(self, layer, kv_cache, md):
        from vllm.v1.attention.backends.quest.impl_helpers import (
            notify_filled_blocks_after_decode,
        )
        notify_filled_blocks_after_decode(layer, kv_cache, md)

    def _forward_sparse_decode(self, layer, query, kv_cache, md, output):
        from vllm.v1.attention.backends.quest.impl_helpers import (
            run_sparse_decode,
        )
        return run_sparse_decode(self, layer, query, kv_cache, md, output)

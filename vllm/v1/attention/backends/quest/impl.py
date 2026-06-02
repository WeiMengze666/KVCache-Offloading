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
          1. KV for this step is already written into the GPU cache by the
             engine via do_kv_cache_update (forward_includes_kv_cache_update
             is False), which runs BEFORE this forward on every path.
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
        if attn_metadata is None:
            # Dummy forward (profiling / capture warmup). Delegate to FA which
            # also no-ops on None metadata. Quest sparse path requires real
            # metadata + a populated tier_manager state, neither of which
            # exists during the profiling pass.
            return self._fa_impl.forward(
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                output_scale=output_scale,
                output_block_scale=output_block_scale,
            )
        is_prefill = attn_metadata.max_query_len > 1
        full_kv = self._is_full_kv_layer(layer, attn_metadata)
        # Quest sparse decode needs at least one fully-filled block per seq
        # (block_size tokens registered to tier_manager). If any seq in the
        # batch is still inside its first block, candidates would include an
        # unfilled block and tier_manager.ensure_resident would fail. Fall
        # back to dense for the whole batch in that case.
        block_size = (
            layer.tier_manager.gpu_k.shape[1]
            if getattr(layer, "tier_manager", None) is not None
            else 0
        )
        any_seq_too_short = block_size > 0 and bool(
            (attn_metadata.seq_lens < block_size).any().item()
        )
        if is_prefill or full_kv or any_seq_too_short:
            # Prefill always runs full attention (spec §1). Full-KV layers
            # always delegate, regardless of phase. KV was already written by
            # the engine via do_kv_cache_update (forward_includes_kv_cache_update
            # is False) before this forward ran, so FA forward only reads.
            #
            # Stage 2C-v2 (footprint_kvshare): under kv-share EVERY Quest layer
            # (the scratch layer AND the layers sharing to it) loses its own
            # per-layer engine cache — they all alias ONE scratch tensor that is
            # overwritten by each subsequent Quest layer within the same forward
            # pass. So the 2A path (defer arena population to a decode-time trim
            # that reads the engine cache) is wrong for ALL of them: by decode
            # the scratch holds the LAST Quest layer's KV. Every Quest layer must
            # therefore offload from its OWN key/value at prefill time.
            #
            # Only the SHARED (non-scratch) layers additionally need us to WRITE
            # the scratch before FA prefill — the engine skips do_kv_cache_update
            # for them (kv_sharing_target guard). The scratch layer's own engine
            # write already happened (it has no share target), so we must NOT
            # double-write it.
            kvshare_layer = (
                not full_kv
                and self._is_footprint_kvshare_layer(layer)
            )
            kvshare_write = (
                kvshare_layer
                and is_prefill
                and self._is_kvshare_shared_layer(layer)
            )
            if kvshare_write:
                self._kvshare_write_scratch(layer, key, value, kv_cache,
                                            attn_metadata)
            out = self._fa_impl.forward(
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                output_scale=output_scale,
                output_block_scale=output_block_scale,
            )
            if kvshare_layer and is_prefill:
                # Quest-owned per-layer offload, sourced from key/value (NOT the
                # scratch, which the next layer overwrites). Establishes the
                # bounded working set + summaries at prefill time. Applies to the
                # scratch layer too (its scratch slot is overwritten by later
                # layers before decode).
                self._kvshare_prefill_offload(layer, key, value, attn_metadata)
            else:
                # During prefill, hand newly completed blocks to the tier manager
                # so the working set + summaries are up to date for the upcoming
                # decode steps.
                self._notify_filled_blocks_after_prefill(
                    layer,
                    key,
                    value,
                    attn_metadata,
                )
            return out

        # Decode of a Quest layer. KV for this step was already written by the
        # engine via do_kv_cache_update before this forward ran, so we go
        # straight to notifying filled blocks (reads from kv_cache) and the
        # sparse path — no manual reshape_and_cache_flash here (that would
        # double-write).
        #
        # Stage 2C-v2: for EVERY kv-share Quest layer (scratch + shared) the
        # arena live block must be filled from this step's key/value, not read
        # back from the scratch (which later Quest layers overwrite this same
        # step). The sparse gather still reads only the arena, so only the WRITE
        # side changes.
        if not full_kv and self._is_footprint_kvshare_layer(layer):
            self._kvshare_decode_write(layer, key, value, attn_metadata)
        else:
            self._notify_filled_blocks_after_decode(
                layer,
                kv_cache,
                attn_metadata,
            )
        return self._forward_sparse_decode(
            layer,
            query,
            kv_cache,
            attn_metadata,
            output,
        )

    def do_kv_cache_update(
        self,
        layer,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        """Engine-driven KV cache write (new "KV write is a separate op"
        contract; see Attention.forward -> unified_kv_cache_update). Runs
        BEFORE forward on every path. Delegate to FlashAttentionImpl so the
        write is byte-identical to the dense backend (reshape_and_cache_flash
        with the layer's k/v scales). This is the single KV write per step
        per layer — the sparse decode path must NOT write KV again.
        """
        self._fa_impl.do_kv_cache_update(
            layer,
            key,
            value,
            kv_cache,
            slot_mapping,
        )

    # ----- private helpers (see Task 14 for full bodies) -----

    def _is_full_kv_layer(self, layer, attn_metadata) -> bool:
        idx = getattr(attn_metadata, "quest_layer_indices", None)
        if idx is None or idx.numel() == 0:
            return True  # safe default = behave like FA
        return bool(idx[layer.layer_idx].item() < 0)

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

    # ----- Stage 2C-v2 (footprint_kvshare): Quest-owned KV write -----

    def _is_kvshare_shared_layer(self, layer) -> bool:
        """True iff this layer is a non-full-KV Quest layer that has been routed
        out of HMA via kv-share (footprint_kvshare on AND a share target set at
        construction). For these layers the engine skips the KV write, so Quest
        owns it. False (the 2A/2B path) when footprint_kvshare is off or the
        layer keeps its own KV (full-KV layers, the scratch layer)."""
        if getattr(layer, "kv_sharing_target_layer_name", None) is None:
            return False
        qc = getattr(layer, "_quest_config_ref", None)
        return bool(qc is not None and getattr(qc, "footprint_kvshare", False))

    def _is_footprint_kvshare_layer(self, layer) -> bool:
        """True iff footprint_kvshare is on AND this is a Quest layer (has a
        tier_manager). This is BROADER than _is_kvshare_shared_layer: it ALSO
        includes the scratch layer (the kv-share target, whose own
        kv_sharing_target is None). Under kv-share the scratch layer loses its
        per-layer engine cache too — all Quest layers alias one scratch tensor
        overwritten within each forward pass — so EVERY Quest layer must do the
        key/value-sourced offload, not the 2A engine-cache-sourced path. Only
        the WRITE of the scratch (kvshare_write_scratch) is shared-layer-only."""
        if getattr(layer, "tier_manager", None) is None:
            return False
        qc = getattr(layer, "_quest_config_ref", None)
        return bool(qc is not None and getattr(qc, "footprint_kvshare", False))

    def _kvshare_write_scratch(self, layer, key, value, kv_cache, md):
        from vllm.v1.attention.backends.quest.impl_helpers import (
            kvshare_write_scratch,
        )

        kvshare_write_scratch(self, layer, key, value, kv_cache, md)

    def _kvshare_prefill_offload(self, layer, key, value, md):
        from vllm.v1.attention.backends.quest.impl_helpers import (
            kvshare_prefill_offload,
        )

        kvshare_prefill_offload(layer, key, value, md)

    def _kvshare_decode_write(self, layer, key, value, md):
        from vllm.v1.attention.backends.quest.impl_helpers import (
            kvshare_decode_write,
        )

        kvshare_decode_write(layer, key, value, md)

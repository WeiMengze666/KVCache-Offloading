# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""QuestSparseOffloadBackend: vLLM v1 attention backend (Phase A skeleton).

Phase A registers and routes; it does not change attention semantics.
Forward is delegated to FlashAttentionImpl (see impl.py). Phase B will swap
the forward implementation in place — this file should not need changes
beyond updating supports_* flags.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import torch

from vllm.v1.attention.backend import AttentionBackend, AttentionType

if TYPE_CHECKING:
    from vllm.config.cache import CacheDType
    from vllm.v1.attention.backend import (
        AttentionImpl,
        AttentionMetadataBuilder,
    )


class QuestSparseOffloadBackend(AttentionBackend):
    """Sparse + KV-offload backend driven by Quest block selection.

    Phase A: identical behavior to FlashAttention.
    Phase B+: real sparse path (see implementation plan / spec).
    """

    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16,
    ]
    supported_kv_cache_dtypes: ClassVar[list["CacheDType"]] = [
        "auto",
        "float16",
        "bfloat16",
    ]

    @staticmethod
    def get_name() -> str:
        return "QUEST_SPARSE_OFFLOAD"

    @staticmethod
    def get_impl_cls() -> type["AttentionImpl"]:
        from vllm.v1.attention.backends.quest.impl import QuestSparseOffloadImpl

        return QuestSparseOffloadImpl

    @staticmethod
    def get_builder_cls() -> type["AttentionMetadataBuilder"]:
        from vllm.v1.attention.backends.quest.metadata import QuestMetadataBuilder

        return QuestMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        # Match FlashAttention's layout exactly so that delegation in Phase A
        # is binary-identical and Phase B can pass the same kv_cache tensor
        # to flash_attn_varlen_func with a custom block_table.
        if block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        return (num_blocks, 2, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend

        return FlashAttentionBackend.get_kv_cache_stride_order(
            include_num_layers_dimension=include_num_layers_dimension
        )

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @classmethod
    def is_mla(cls) -> bool:
        return False

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend

        return FlashAttentionBackend.supports_head_size(head_size)

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER

    @classmethod
    def validate_quest_configuration(
        cls,
        *,
        model_config,
        cache_config,
        quest_config,
    ) -> list[str]:
        """Return [] when this configuration is acceptable, else a list of
        human-readable reasons. Phase B helper for unit-testable validation;
        Phase E will pin this onto the actual selector wiring.
        """
        from vllm.v1.attention.backends.quest.compatibility import (
            check_model_compat,
        )

        if quest_config is None or not quest_config.enabled:
            return []

        errors: list[str] = []

        if cache_config.block_size % 256 != 0:
            errors.append(
                f"cache_config.block_size={cache_config.block_size} is not a "
                "multiple of 256. flash_attn paged kernels (FA2/FA3) require "
                "block_size % 256 == 0. Set --block-size 256 or larger."
            )

        if quest_config.top_k > quest_config.gpu_cache_blocks_per_seq:
            errors.append(
                f"top_k ({quest_config.top_k}) > gpu_cache_blocks_per_seq "
                f"({quest_config.gpu_cache_blocks_per_seq}); the working set "
                "must fit the selected blocks."
            )

        compat = check_model_compat(model_config)
        if compat:
            if quest_config.unsupported_model_policy == "error":
                errors.extend(compat)
            # else 'fallback': selector will pick the default backend; we
            # silently refuse this one without surfacing errors.

        return errors

    @classmethod
    def init_runtime_state(
        cls,
        *,
        layers,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        max_blocks_total: int,
        dtype: torch.dtype,
        quest_config,
        kv_caches: dict[str, torch.Tensor] | None = None,
    ) -> None:
        """Construct the shared BlockSummaryStore + CpuKvBackingStore + per-
        layer TierManager objects, attach a `tier_manager` attribute to each
        Quest layer (full-KV layers are left untouched)."""
        from vllm.v1.attention.backends.quest.cache.block_summary import (
            BlockSummaryStore,
        )
        from vllm.v1.attention.backends.quest.cache.cpu_backing_store import (
            CpuKvBackingStore,
        )
        from vllm.v1.attention.backends.quest.cache.residency import (
            BlockResidency,
        )
        from vllm.v1.attention.backends.quest.cache.tier_manager import (
            TierManager,
        )

        if not quest_config.enabled:
            return

        full_set = set(quest_config.full_kv_layers)
        quest_layers = [l for l in layers if l.layer_idx not in full_set]
        if not quest_layers:
            return

        num_quest = len(quest_layers)
        page_bytes = (
            2 * block_size * num_kv_heads * head_size
            * torch.tensor([], dtype=dtype).element_size()
        )
        cpu_blocks = quest_config.resolve_cpu_blocks_per_layer(
            page_size_bytes=page_bytes, num_quest_layers=num_quest,
        )

        summary = BlockSummaryStore(
            num_layers=num_quest,
            max_blocks=max_blocks_total,
            block_size=block_size, num_kv_heads=num_kv_heads,
            head_size=head_size, dtype=dtype, device="cuda",
        )
        cpu_store = CpuKvBackingStore(
            num_layers=num_quest,
            blocks_per_layer=cpu_blocks,
            block_size=block_size, num_kv_heads=num_kv_heads,
            head_size=head_size, dtype=dtype,
        )
        residency = BlockResidency(
            num_layers=num_quest, max_blocks=max_blocks_total,
        )

        # GPU paged buffers: prefer vLLM-allocated tensor when supplied
        # (Phase E hook). Fall back to fresh allocation for unit tests.
        for slot, layer in enumerate(quest_layers):
            layer_name = getattr(layer, "layer_name", None)
            if kv_caches is not None and layer_name in kv_caches:
                full = kv_caches[layer_name]
                # FA layout: (num_blocks, 2, block_size, num_kv_heads, head_size)
                # Zero-copy slice views into the vLLM-allocated tensor.
                gpu_k = full[:, 0]
                gpu_v = full[:, 1]
                gpu_budget = full.shape[0]
            else:
                gpu_k = torch.empty(
                    (quest_config.gpu_cache_blocks_per_seq, block_size,
                     num_kv_heads, head_size),
                    dtype=dtype, device="cuda",
                )
                gpu_v = torch.empty_like(gpu_k)
                gpu_budget = quest_config.gpu_cache_blocks_per_seq
            layer.tier_manager = TierManager(
                layer_idx=slot,
                gpu_budget=gpu_budget,
                gpu_k=gpu_k, gpu_v=gpu_v,
                summary_store=summary,
                residency=residency,
                cpu_store=cpu_store,
            )

    @classmethod
    def bind_runtime(
        cls,
        *,
        vllm_config,
        kv_cache_config,
        kv_caches: dict[str, torch.Tensor],
        layers: dict[str, object],
    ) -> None:
        """Single Phase E entry point called from GPUModelRunner.

        1. No-op when quest_config is disabled.
        2. Run validate_quest_configuration; raise ValueError on failure.
        3. Filter layers to those bound to QuestSparseOffloadBackend.
        4. Compute (block_size, num_kv_heads, head_size, dtype) from the
           Quest layers' KV cache spec — guaranteed homogeneous because
           QuestKVCacheSpec.merge enforces equality.
        5. Call init_runtime_state with the kv_caches dict so each
           TierManager points into the vLLM-allocated tensor.
        """
        quest_config = getattr(vllm_config, "quest_config", None)
        if quest_config is None or not quest_config.enabled:
            return

        errors = cls.validate_quest_configuration(
            model_config=vllm_config.model_config,
            cache_config=vllm_config.cache_config,
            quest_config=quest_config,
        )
        if errors:
            raise ValueError(
                "QuestSparseOffloadBackend configuration is invalid:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

        quest_layers_list = [
            layer for layer in layers.values()
            if getattr(layer, "attn_backend", None) is cls
            and layer.layer_idx not in set(quest_config.full_kv_layers)
        ]
        if not quest_layers_list:
            return

        sample = quest_layers_list[0]
        block_size = vllm_config.cache_config.block_size

        cls.init_runtime_state(
            layers=quest_layers_list,
            block_size=block_size,
            num_kv_heads=sample.num_kv_heads,
            head_size=sample.head_size,
            max_blocks_total=kv_cache_config.num_blocks,
            dtype=sample.kv_cache_torch_dtype,
            quest_config=quest_config,
            kv_caches=kv_caches,
        )

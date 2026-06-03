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

from vllm.logger import init_logger
from vllm.v1.attention.backend import AttentionBackend, AttentionType

logger = init_logger(__name__)

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

    .. warning::

       Phase E constraint: this backend's TierManager maintains its own
       per-layer LRU mapping ``(seq_id, logical_block_id) -> gpu_slot``.
       vLLM's KVCacheManager separately allocates / evicts / reuses
       blocks based on its own scheduler logic. **Prefix caching MUST be
       disabled** when Quest is enabled — otherwise the KV manager may
       reuse a slot that TierManager still believes is bound to a logical
       block, producing silent corruption. This will be addressed in
       Phase F by making TierManager subscribe to KV manager block
       reclamation events.
    """

    # vLLM's "KV write is a separate op" contract: when False, the engine
    # (Attention.forward -> unified_kv_cache_update -> impl.do_kv_cache_update)
    # performs the KV cache write BEFORE impl.forward runs, on every path.
    # Mirror FlashAttentionBackend (flash_attn.py:96), whose forward no longer
    # writes KV. We delegate the write to FlashAttentionImpl.do_kv_cache_update
    # via QuestSparseOffloadImpl.do_kv_cache_update.
    forward_includes_kv_cache_update: bool = False

    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16,
    ]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
    ]

    @staticmethod
    def get_name() -> str:
        # Returns the AttentionBackendEnum slot name. Quest backend is
        # registered against AttentionBackendEnum.CUSTOM (see
        # `vllm/v1/attention/backends/quest/registration.py`); upstream
        # `Attention.__init__` does `AttentionBackendEnum[get_name()]`,
        # so the literal must match a real enum member.
        # The human-friendly identifier "QUEST_SPARSE_OFFLOAD" lives on
        # `QuestConfig.backend_name` and is what shows up in logs.
        return "CUSTOM"

    @staticmethod
    def get_impl_cls() -> type[AttentionImpl]:
        from vllm.v1.attention.backends.quest.impl import QuestSparseOffloadImpl

        return QuestSparseOffloadImpl

    @staticmethod
    def get_builder_cls() -> type[AttentionMetadataBuilder]:
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

        # Stage 2C-v2 (Task A3): under footprint_kvshare the non-full-KV Quest
        # layers are aliased to ONE physical scratch tensor (kv-share). Prefix
        # caching would let vLLM's block manager reuse those blocks behind the
        # TierManager's back → silent corruption. This is a TEMPORARY constraint
        # of the kv-share-eviction design, NOT a vLLM invariant; guard it loudly
        # so a future change (or a user flag) fails instead of corrupting.
        if getattr(quest_config, "footprint_kvshare", False) and getattr(
            cache_config, "enable_prefix_caching", False
        ):
            errors.append(
                "footprint_kvshare requires prefix caching to be OFF "
                "(enable_prefix_caching=False). The shared Quest layers reuse "
                "one physical scratch tensor, so prefix-cache reuse of those "
                "blocks is silent corruption. This is a TEMPORARY constraint of "
                "the kv-share-eviction design, not a vLLM invariant. Pass "
                "--no-enable-prefix-caching."
            )

        if quest_config.top_k > quest_config.gpu_cache_blocks_per_seq - 1:
            errors.append(
                f"top_k ({quest_config.top_k}) must be <= "
                f"gpu_cache_blocks_per_seq - 1 "
                f"({quest_config.gpu_cache_blocks_per_seq - 1}); one arena slot "
                "is reserved for the live decode block."
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
        max_model_len: int | None = None,
        max_num_seqs: int | None = None,
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
        from vllm.v1.attention.ops.quest_selection_dispatch import (
            _resolve_selection_callable,
        )

        if not quest_config.enabled:
            return

        full_set = set(quest_config.full_kv_layers)
        # Filter is also applied by bind_runtime; kept here as defense-in-depth
        # for direct callers (unit tests).
        quest_layers = [l for l in layers if l.layer_idx not in full_set]
        if not quest_layers:
            return

        selection_callable = _resolve_selection_callable(
            quest_config.selection_impl,
        )

        num_quest = len(quest_layers)
        page_bytes = (
            2
            * block_size
            * num_kv_heads
            * head_size
            * torch.tensor([], dtype=dtype).element_size()
        )
        cpu_blocks = quest_config.resolve_cpu_blocks_per_layer(
            page_size_bytes=page_bytes,
            num_quest_layers=num_quest,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            block_size=block_size,
        )

        summary = BlockSummaryStore(
            num_layers=num_quest,
            max_blocks=max_blocks_total,
            block_size=block_size,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            dtype=dtype,
            device="cuda",
            digest_mode=quest_config.digest_mode,
        )
        cpu_store = CpuKvBackingStore(
            num_layers=num_quest,
            blocks_per_layer=cpu_blocks,
            block_size=block_size,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            dtype=dtype,
        )
        residency = BlockResidency(
            num_layers=num_quest,
            max_blocks=max_blocks_total,
        )

        # Phase C: optional async transfer infrastructure. None when the
        # config gate is off — TierManager falls back to Phase B sync path.
        stream_pool = None
        if quest_config.enable_async_prefetch:
            from vllm.v1.attention.backends.quest.async_transfer import (
                QuestStreamPool,
            )

            stream_pool = QuestStreamPool()

        # GPU paged buffers: prefer vLLM-allocated tensor when supplied
        # (Phase E hook). Fall back to fresh allocation for unit tests.
        for slot, layer in enumerate(quest_layers):
            layer_name = getattr(layer, "layer_name", None)
            if kv_caches is not None and layer_name in kv_caches:
                full = kv_caches[layer_name]
                # Stage 2A: do NOT alias the full engine cache. Allocate a private
                # bounded arena of gpu_cache_blocks_per_seq blocks; the engine
                # tensor is kept ONLY as the SOURCE for the prefill->decode trim
                # (TierManager.trim_to_working_set, added in a later task).
                # FA layout: full.shape = (num_blocks, 2, block_size, num_kv_heads, head_size)
                cap = quest_config.gpu_cache_blocks_per_seq
                gpu_k = torch.empty(
                    cap,
                    full.shape[2],
                    full.shape[3],
                    full.shape[4],
                    dtype=full.dtype,
                    device=full.device,
                )
                gpu_v = torch.empty_like(gpu_k)
                gpu_budget = cap
                pool_aliases_kv_cache = False
                engine_kv_for_layer = full
            else:
                if kv_caches is not None:
                    logger.warning(
                        "QuestSparseOffloadBackend: layer %r has "
                        "layer_name=%r but kv_caches has no entry for it. "
                        "Falling back to fresh allocation; vLLM-allocated "
                        "tensor will be unused for this layer.",
                        layer,
                        layer_name,
                    )
                gpu_k = torch.empty(
                    (
                        quest_config.gpu_cache_blocks_per_seq,
                        block_size,
                        num_kv_heads,
                        head_size,
                    ),
                    dtype=dtype,
                    device="cuda",
                )
                gpu_v = torch.empty_like(gpu_k)
                gpu_budget = quest_config.gpu_cache_blocks_per_seq
                # Private buffer: on_block_filled copies KV in and the decode
                # read path reads via logical_to_slot (unit-test path).
                pool_aliases_kv_cache = False
                engine_kv_for_layer = None
            layer.tier_manager = TierManager(
                layer_idx=slot,
                gpu_budget=gpu_budget,
                gpu_k=gpu_k,
                gpu_v=gpu_v,
                summary_store=summary,
                residency=residency,
                cpu_store=cpu_store,
                stream_pool=stream_pool,
                enable_event_timing=quest_config.enable_debug_counters,
                enable_overlap_capture=quest_config.enable_debug_counters,
                gpu_pool_aliases_kv_cache=pool_aliases_kv_cache,
                engine_kv_cache=engine_kv_for_layer,
                enable_write_through=quest_config.enable_write_through,
                # Stage 3 fields live only on QuestConfig; sibling configs
                # (e.g. ArkValeConfig) reuse this path and never set them, so
                # default to the byte-identical lru/False behavior via getattr.
                block_ordering=getattr(quest_config, "block_ordering", "lru"),
                prefetch_touch=getattr(quest_config, "prefetch_touch", False),
            )
            layer._quest_selection_callable_ref = selection_callable
            # Stash the config on every quest layer so impl.forward can read
            # footprint_kvshare (Stage 2C-v2) without a global lookup. (Mode 2
            # also sets this below for the registry path; harmless to set here.)
            layer._quest_config_ref = quest_config

        # Mode 2 layer-registry: only when async is on AND prefetch window
        # is non-zero. Without these refs, run_sparse_decode's helpers
        # return None and Mode 2 is inert.
        # Stage 3: the cross-layer registry is needed for any non-lru ordering
        # (prefetch/mixture), not only the legacy window>0 trigger. lru never
        # needs cross-layer refs.
        if (
            stream_pool is not None
            and getattr(quest_config, "block_ordering", "lru") != "lru"
        ):
            tm_registry: dict[int, TierManager] = {
                l.layer_idx: l.tier_manager for l in quest_layers
            }
            indices_view = sorted(tm_registry.keys())
            for l in quest_layers:
                l._quest_config_ref = quest_config
                l._quest_layer_tm_registry = tm_registry
                l._quest_layer_indices_view = indices_view

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
        from vllm.config import get_active_sparse_cfg

        quest_config = get_active_sparse_cfg(vllm_config)
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
            layer
            for layer in layers.values()
            if getattr(layer, "attn_backend", None) is cls
            and layer.layer_idx not in set(quest_config.full_kv_layers)
        ]
        if not quest_layers_list:
            return

        sample = quest_layers_list[0]
        block_size = vllm_config.cache_config.block_size
        # Stage 2B Q1: host-pool sizing needs the longest sequence and the
        # concurrency so write-through can back every logical block.
        max_model_len = getattr(
            vllm_config.model_config, "max_model_len", None
        )
        max_num_seqs = getattr(
            getattr(vllm_config, "scheduler_config", None),
            "max_num_seqs",
            None,
        )

        cls.init_runtime_state(
            layers=quest_layers_list,
            block_size=block_size,
            num_kv_heads=sample.num_kv_heads,
            head_size=sample.head_size,
            max_blocks_total=kv_cache_config.num_blocks,
            dtype=sample.kv_cache_torch_dtype,
            quest_config=quest_config,
            kv_caches=kv_caches,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
        )

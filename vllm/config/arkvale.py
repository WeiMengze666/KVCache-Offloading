# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ArkValeConfig: configuration for the ArkVale (cuboid_mean) sparse selector.

ArkVale shares the entire QuestSparseOffloadBackend, BlockSummaryStore,
and KV tiering / CPU offload stack with Quest. The only algorithmic
difference is the page-digest formula, controlled by `digest_mode`
(default 'arkvale_cuboid_mean'). All other fields mirror QuestConfig.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

EvictionPolicy = Literal["lru", "arc"]
SelectionImpl = Literal["torch", "triton", "cuda"]
UnsupportedModelPolicy = Literal["error", "fallback"]
DigestMode = Literal["quest_minmax", "arkvale_cuboid_mean"]


@dataclass
class ArkValeConfig:
    enabled: bool = False
    backend_name: str = "ARKVALE_SPARSE_OFFLOAD"

    # Algorithm
    block_size: int = 32
    top_k: int = 64
    full_kv_layers: list[int] = field(default_factory=lambda: [0, 1])

    # GPU/CPU tiering
    gpu_cache_blocks_per_seq: int = 256
    cpu_cache_blocks: int = 65536
    cpu_cache_gib: int | None = None
    eviction_policy: EvictionPolicy = "lru"

    # Async (Phase C)
    enable_async_prefetch: bool = False
    enable_double_buffering: bool = False
    num_h2d_streams: int = 1
    num_d2h_streams: int = 1
    prefetch_window_blocks: int = 0

    # Kernel
    selection_impl: SelectionImpl = "torch"

    # Digest formula. ArkVale defaults to cuboid-mean.
    digest_mode: DigestMode = "arkvale_cuboid_mean"

    # Debug
    enable_debug_counters: bool = False

    # Compatibility
    unsupported_model_policy: UnsupportedModelPolicy = "error"

    def validate(self) -> None:
        if self.top_k <= 0:
            raise ValueError(f"top_k must be positive, got {self.top_k}")
        if self.top_k > self.gpu_cache_blocks_per_seq:
            raise ValueError(
                f"top_k ({self.top_k}) must be <= "
                f"gpu_cache_blocks_per_seq ({self.gpu_cache_blocks_per_seq})"
            )
        if self.block_size <= 0:
            raise ValueError(f"block_size must be positive, got {self.block_size}")
        if self.cpu_cache_blocks < 0:
            raise ValueError(
                f"cpu_cache_blocks must be >= 0, got {self.cpu_cache_blocks}"
            )
        if self.cpu_cache_gib is not None and self.cpu_cache_gib <= 0:
            raise ValueError(
                f"cpu_cache_gib must be positive when set, got {self.cpu_cache_gib}"
            )
        if self.eviction_policy not in ("lru", "arc"):
            raise ValueError(
                f"eviction_policy must be 'lru' or 'arc', got {self.eviction_policy!r}"
            )
        if self.selection_impl not in ("torch", "triton", "cuda"):
            raise ValueError(
                f"selection_impl must be 'torch', 'triton', or 'cuda', "
                f"got {self.selection_impl!r}"
            )
        if self.unsupported_model_policy not in ("error", "fallback"):
            raise ValueError(
                f"unsupported_model_policy must be 'error' or 'fallback', "
                f"got {self.unsupported_model_policy!r}"
            )
        if not isinstance(self.full_kv_layers, list) or not all(
            isinstance(x, int) for x in self.full_kv_layers
        ):
            raise ValueError(
                f"full_kv_layers must be a list of int, got {self.full_kv_layers!r}"
            )
        if self.prefetch_window_blocks < 0:
            raise ValueError(
                f"prefetch_window_blocks must be >= 0, "
                f"got {self.prefetch_window_blocks}"
            )
        if self.prefetch_window_blocks > 0 and not self.enable_async_prefetch:
            raise ValueError(
                "prefetch_window_blocks > 0 requires enable_async_prefetch=True."
            )
        if self.digest_mode not in ("quest_minmax", "arkvale_cuboid_mean"):
            raise ValueError(
                f"digest_mode must be 'quest_minmax' or "
                f"'arkvale_cuboid_mean', got {self.digest_mode!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ArkValeConfig:
        return cls(**data)

    def resolve_cpu_blocks_per_layer(
        self,
        *,
        page_size_bytes: int,
        num_quest_layers: int,
    ) -> int:
        if num_quest_layers <= 0:
            return 0
        legacy_cap = self.cpu_cache_blocks
        if self.cpu_cache_gib is None:
            return legacy_cap
        gib_cap = self.cpu_cache_gib * (1024**3) // page_size_bytes // num_quest_layers
        return min(legacy_cap, gib_cap)

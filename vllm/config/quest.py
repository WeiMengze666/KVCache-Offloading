# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""QuestConfig: configuration for the Quest sparse offload attention backend.

Phase A only carries plumbing. Tiering / async / kernel fields are present so
later phases can flip them without re-introducing config-shape churn — but
they are validated and have safe defaults that keep Phase A behavior equal to
FlashAttention with the gate flipped.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

EvictionPolicy = Literal["lru", "arc"]
SelectionImpl = Literal["torch", "triton", "cuda"]
UnsupportedModelPolicy = Literal["error", "fallback"]


@dataclass
class QuestConfig:
    enabled: bool = False
    backend_name: str = "QUEST_SPARSE_OFFLOAD"

    # Quest algorithm (Phase B activates these).
    block_size: int = 32
    top_k: int = 64
    full_kv_layers: list[int] = field(default_factory=lambda: [0, 1])

    # GPU/CPU tiering (Phase B activates these).
    gpu_cache_blocks_per_seq: int = 256
    cpu_cache_blocks: int = 65536
    cpu_cache_gib: int | None = None
    """Total pinned CPU pool budget in GiB across ALL Quest layers.

    When set, the runtime computes `floor(cpu_cache_gib * 1024**3 /
    page_size_bytes / num_quest_layers)` and takes the min with
    `cpu_cache_blocks` (the legacy per-layer ceiling). When None, only the
    legacy ceiling applies. Tighter constraint always wins.

    Set this when host RAM is the binding constraint. The legacy ceiling
    is kept for backwards compatibility with the Transformers-side
    configuration."""
    eviction_policy: EvictionPolicy = "lru"

    # Async (Phase C activates these).
    enable_async_prefetch: bool = False
    """Phase C gate. When True, ensure_resident issues non_blocking=True H2D
    on a dedicated h2d_stream and returns an event for the compute stream
    to wait on before the kernel runs (Mode 1). When False, all transfers
    are synchronous (Phase B behavior). Default False; flip to True to opt
    in to async transfers."""

    enable_double_buffering: bool = False
    """Phase C reserved. Currently unused — the Phase C design uses a single
    h2d/d2h stream pair without staging buffers (each Quest layer has its
    own GPU pool, so layer-N forward and H2D into layer-N+1 don't conflict).
    Reserved for future expansion."""

    num_h2d_streams: int = 1
    """Phase C reserved. Currently fixed at 1; multi-stream H2D is deferred."""

    num_d2h_streams: int = 1
    """Phase C reserved. Currently fixed at 1."""

    prefetch_window_blocks: int = 0
    """Mode 2 toggle. When > 0 and enable_async_prefetch=True, after layer N's
    forward we speculatively prefetch layer N's top_ids into layer N+1's GPU
    pool on the h2d_stream. Layer N+1's forward waits on the prefetch event
    before starting.

    .. warning::

       Mode 2 carries a structural LRU-thrash risk. When the GPU pool is
       full (steady state) and the speculative prefetch picks differ from
       layer N+1's actual selection, every wrong prefetch evicts an LRU
       block to CPU, and the actual selection then has to refetch it. In
       the worst case (zero overlap between speculation and reality),
       Mode 2 can be 2x slower than Mode 1.

       Quest's cross-layer top-k overlap is workload-dependent and has
       not been measured for this project. **Do not enable Mode 2
       (prefetch_window_blocks > 0) in production without first
       benchmarking the overlap fraction on your model.** Phase D may
       add an overlap-threshold gate; until then, Mode 2 is best left
       at 0.
    """

    # Kernel dispatch (Phase D activates "cuda").
    selection_impl: SelectionImpl = "torch"

    # Debug.
    enable_debug_counters: bool = False

    # Compatibility behavior when the loaded model isn't whitelisted.
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
            raise ValueError(
                f"block_size must be positive, got {self.block_size}"
            )
        if self.cpu_cache_blocks < 0:
            raise ValueError(
                f"cpu_cache_blocks must be >= 0, got {self.cpu_cache_blocks}"
            )
        if self.cpu_cache_gib is not None and self.cpu_cache_gib <= 0:
            raise ValueError(
                f"cpu_cache_gib must be positive when set, "
                f"got {self.cpu_cache_gib}"
            )
        if self.eviction_policy not in ("lru", "arc"):
            raise ValueError(
                f"eviction_policy must be 'lru' or 'arc', "
                f"got {self.eviction_policy!r}"
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
                f"full_kv_layers must be a list of int, "
                f"got {self.full_kv_layers!r}"
            )
        if self.prefetch_window_blocks < 0:
            raise ValueError(
                f"prefetch_window_blocks must be >= 0, "
                f"got {self.prefetch_window_blocks}"
            )
        if self.prefetch_window_blocks > 0 and not self.enable_async_prefetch:
            raise ValueError(
                "prefetch_window_blocks > 0 (Mode 2) requires "
                "enable_async_prefetch=True (Mode 1)."
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QuestConfig":
        return cls(**data)

    def resolve_cpu_blocks_per_layer(
        self, *, page_size_bytes: int, num_quest_layers: int,
    ) -> int:
        if num_quest_layers <= 0:
            return 0
        legacy_cap = self.cpu_cache_blocks
        if self.cpu_cache_gib is None:
            return legacy_cap
        gib_cap = (
            self.cpu_cache_gib * (1024 ** 3) // page_size_bytes
            // num_quest_layers
        )
        return min(legacy_cap, gib_cap)

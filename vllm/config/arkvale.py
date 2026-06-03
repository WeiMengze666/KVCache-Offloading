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
    digest_mode: DigestMode = "arkvale_cuboid_mean"
    """Page digest formula. Differs from QuestConfig: defaults to 'arkvale_cuboid_mean'.
    'quest_minmax' = true K amax/amin (Quest, default).
    'arkvale_cuboid_mean' = center +/- mean(|K - center|) where
    center = (true_max + true_min) / 2 (ArkVale cuboid-mean variant).

    The digest tensor layout is identical for both modes — selection ops
    and CPU offload are unaware of which formula produced the values."""

    # GPU/CPU tiering
    gpu_cache_blocks_per_seq: int = 256
    cpu_cache_blocks: int = 65536
    cpu_cache_gib: int | None = None
    """Total pinned CPU pool budget in GiB across ALL ArkVale layers.

    When set, the runtime computes `floor(cpu_cache_gib * 1024**3 /
    page_size_bytes / num_quest_layers)` and takes the min with
    `cpu_cache_blocks` (the legacy per-layer ceiling). When None, only the
    legacy ceiling applies. Tighter constraint always wins.

    Set this when host RAM is the binding constraint. The legacy ceiling
    is kept for backwards compatibility with the Transformers-side
    configuration."""
    eviction_policy: EvictionPolicy = "lru"

    # Stage 2B: write-through D2H. Mirrors QuestConfig.enable_write_through
    # so the shared backend treats the two configs as interchangeable. When
    # True, every full block is mirrored to a durable pinned-host slot at
    # fill time; eviction then just drops the GPU slot and a miss is H2D-only.
    # Requires the host pool to hold every logical block of the sequence —
    # see resolve_cpu_blocks_per_layer's max_model_len sizing. Opt-in;
    # False keeps the 2A write-back path byte-for-byte.
    enable_write_through: bool = False

    # Stage 2C-v2: footprint reduction via kv-share. Mirrors
    # QuestConfig.footprint_kvshare. When True, the non-full-KV ArkVale
    # layers are routed OUT of the HMA KV-cache groups by pointing each at
    # the first ArkVale layer ("scratch") via vLLM's kv-sharing channel.
    # Requires prefix caching OFF. Opt-in; False keeps the 2A/2B path
    # byte-for-byte.
    footprint_kvshare: bool = False

    # Async (Phase C)
    enable_async_prefetch: bool = False
    """Phase C gate. When True, ensure_resident issues non_blocking=True H2D
    on a dedicated h2d_stream and returns an event for the compute stream
    to wait on before the kernel runs (Mode 1). When False, all transfers
    are synchronous (Phase B behavior). Default False; flip to True to opt
    in to async transfers."""

    enable_double_buffering: bool = False
    """Phase C reserved. Currently unused — the Phase C design uses a single
    h2d/d2h stream pair without staging buffers (each ArkVale layer has its
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

       ArkVale's cross-layer top-k overlap is workload-dependent and has
       not been measured for this project. **Do not enable Mode 2
       (prefetch_window_blocks > 0) in production without first
       benchmarking the overlap fraction on your model.** Phase D may
       add an overlap-threshold gate; until then, Mode 2 is best left
       at 0.
    """

    # Kernel
    selection_impl: SelectionImpl = "torch"

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
        if self.digest_mode not in ("quest_minmax", "arkvale_cuboid_mean"):
            raise ValueError(
                f"digest_mode must be 'quest_minmax' or "
                f"'arkvale_cuboid_mean', got {self.digest_mode!r}"
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
        if not isinstance(self.enable_write_through, bool):
            raise ValueError(
                f"enable_write_through must be a bool, "
                f"got {self.enable_write_through!r}"
            )
        if not isinstance(self.footprint_kvshare, bool):
            raise ValueError(
                f"footprint_kvshare must be a bool, "
                f"got {self.footprint_kvshare!r}"
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
        max_model_len: int | None = None,
        max_num_seqs: int | None = None,
        block_size: int | None = None,
    ) -> int:
        """Per-layer pinned-host block count.

        Mirrors ``QuestConfig.resolve_cpu_blocks_per_layer`` so the shared
        backend (``init_runtime_state``) can call either config object with
        the same kwargs. The ceiling is the tighter of ``cpu_cache_blocks``
        and the optional ``cpu_cache_gib`` byte budget. When the sizing
        kwargs are given, the pool sizes UP to ``need = cdiv(max_model_len,
        block_size) * max_num_seqs`` but never above the ceiling. If the
        ceiling is below ``need`` AND write-through is enabled, that's
        unsatisfiable (an evicted block would have no host backup → silent
        corruption), so raise loudly. With write-through off, the
        under-provisioned ceiling is tolerated.
        """
        if num_quest_layers <= 0:
            return 0
        legacy_cap = self.cpu_cache_blocks
        if self.cpu_cache_gib is None:
            ceiling = legacy_cap
        else:
            gib_cap = (
                self.cpu_cache_gib * (1024**3)
                // page_size_bytes
                // num_quest_layers
            )
            ceiling = min(legacy_cap, gib_cap)

        if max_model_len is None or max_num_seqs is None or block_size is None:
            return ceiling

        need = -(-max_model_len // block_size) * max_num_seqs  # cdiv * seqs
        if need > ceiling:
            if self.enable_write_through:
                raise RuntimeError(
                    f"ArkVale write-through needs the host pool to back "
                    f"the whole sequence: need {need} blocks/layer "
                    f"(cdiv({max_model_len},{block_size}) * {max_num_seqs}) "
                    f"but the configured ceiling is only {ceiling} "
                    f"(cpu_cache_blocks={self.cpu_cache_blocks}, "
                    f"cpu_cache_gib={self.cpu_cache_gib}). Raise the ceiling "
                    f"or lower max_model_len/max_num_seqs."
                )
            return ceiling
        return need

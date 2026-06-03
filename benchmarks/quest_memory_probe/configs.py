# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RunConfig: a single experiment configuration sent to a subprocess runner.

Pure-Python, no GPU/vLLM imports. Allows unit tests without a CUDA environment.
"""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class RunConfig:
    name: str
    # Engine-level
    model: str = "meta-llama/Llama-3.2-3B-Instruct"
    block_size: int = 256
    gpu_memory_utilization: float = 0.55
    max_model_len: int = 65536
    enforce_eager: bool = True
    seed: int = 1234
    dtype: str = "float16"
    # Workload
    workload_spec: str = "longbench:narrativeqa:lengths=short:n=1"
    max_tokens: int = 64
    # When True, the LongBench loader ignores `n=` and takes every item that
    # falls into a requested bucket. Synthetic fallback ignores this flag.
    longbench_full: bool = False
    # Quest
    quest_enabled: bool = False
    top_k: int = 0
    gpu_cache_blocks_per_seq: int = 0
    cpu_cache_blocks: int = 8192
    cpu_cache_gib: float = 24.0
    selection_impl: str = "torch"
    full_kv_layers: tuple[int, ...] = field(default_factory=tuple)
    # Probe
    probe_interval_ms: int = 250

    def validate(self) -> None:
        if self.block_size != 256:
            raise ValueError(
                f"block_size must be 256 (got {self.block_size}); "
                "flash_attn 2.8.3 SM89 requires this."
            )
        if not self.quest_enabled:
            return
        if self.top_k <= 0:
            raise ValueError("top_k must be > 0 when quest_enabled=True")
        if self.gpu_cache_blocks_per_seq <= 0:
            raise ValueError(
                "gpu_cache_blocks_per_seq must be > 0 when quest_enabled=True"
            )
        if self.gpu_cache_blocks_per_seq % self.top_k != 0:
            warnings.warn(
                f"gpu_cache_blocks_per_seq ({self.gpu_cache_blocks_per_seq}) "
                f"is not a multiple of top_k ({self.top_k}); the working-set "
                "won't tile the GPU pool cleanly, so probe curves between "
                "integer-multiple points may be hard to interpret.",
                stacklevel=2,
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RunConfig:
        # tuple round-trips via asdict as list
        d = dict(d)
        if "full_kv_layers" in d and isinstance(d["full_kv_layers"], list):
            d["full_kv_layers"] = tuple(d["full_kv_layers"])
        return cls(**d)


def _make(name: str, **overrides: Any) -> RunConfig:
    cfg = RunConfig(name=name, **overrides)
    cfg.validate()
    return cfg


def expand_dense_vs_quest(
    *,
    workload_spec: str,
    top_k: int,
    quest_pool: int,
) -> list[RunConfig]:
    """Subcommand A: dense baseline vs Quest enabled (no offload pressure)."""
    return [
        _make(
            name="dense",
            workload_spec=workload_spec,
            quest_enabled=False,
        ),
        _make(
            name=f"quest_pool{quest_pool}_topk{top_k}",
            workload_spec=workload_spec,
            quest_enabled=True,
            top_k=top_k,
            gpu_cache_blocks_per_seq=quest_pool,
        ),
    ]


def expand_pool_size(
    *,
    workload_spec: str,
    top_k: int,
    pool_sizes: list[int],
) -> list[RunConfig]:
    """Subcommand B: same Quest config, sweep gpu_cache_blocks_per_seq.

    Each pool_size that isn't a multiple of top_k will trigger a warning
    (the working-set no longer tiles the GPU pool cleanly), but is still
    accepted. Validation runs in RunConfig.validate(); we just construct +
    validate here.
    """
    out: list[RunConfig] = []
    for p in pool_sizes:
        out.append(
            _make(
                name=f"quest_pool{p}_topk{top_k}",
                workload_spec=workload_spec,
                quest_enabled=True,
                top_k=top_k,
                gpu_cache_blocks_per_seq=p,
            )
        )
    return out


def expand_oom_sweep(
    *,
    workload_spec: str,
    top_k: int,
    quest_pool: int,
) -> list[RunConfig]:
    """Subcommand C: dense vs one Quest config; runner orders samples by
    prompt_tokens ascending and walks until OOM."""
    return [
        _make(
            name="dense_oom",
            workload_spec=workload_spec,
            quest_enabled=False,
        ),
        _make(
            name=f"quest_pool{quest_pool}_topk{top_k}_oom",
            workload_spec=workload_spec,
            quest_enabled=True,
            top_k=top_k,
            gpu_cache_blocks_per_seq=quest_pool,
        ),
    ]

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ArkValeConfig dataclass tests (mirrors test_quest_config.py)."""

import pytest

from vllm.config.arkvale import ArkValeConfig


def test_arkvale_config_defaults():
    cfg = ArkValeConfig()
    assert cfg.enabled is False
    assert cfg.backend_name == "ARKVALE_SPARSE_OFFLOAD"
    assert cfg.block_size == 32
    assert cfg.top_k == 64
    assert cfg.full_kv_layers == [0, 1]
    assert cfg.gpu_cache_blocks_per_seq == 256
    assert cfg.cpu_cache_blocks == 65536
    assert cfg.cpu_cache_gib is None
    assert cfg.eviction_policy == "lru"
    assert cfg.enable_async_prefetch is False
    assert cfg.prefetch_window_blocks == 0
    assert cfg.selection_impl == "torch"
    assert cfg.unsupported_model_policy == "error"
    assert cfg.digest_mode == "arkvale_cuboid_mean"


def test_arkvale_config_validate_ok():
    cfg = ArkValeConfig(enabled=True)
    cfg.validate()


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"top_k": 0}, "top_k"),
        ({"top_k": 1000, "gpu_cache_blocks_per_seq": 4}, "gpu_cache_blocks_per_seq"),
        ({"block_size": 0}, "block_size"),
        ({"cpu_cache_blocks": -1}, "cpu_cache_blocks"),
        ({"cpu_cache_gib": 0}, "cpu_cache_gib"),
        ({"eviction_policy": "fifo"}, "eviction_policy"),
        ({"selection_impl": "cuda2"}, "selection_impl"),
        ({"unsupported_model_policy": "warn"}, "unsupported_model_policy"),
        ({"prefetch_window_blocks": -1}, "prefetch_window_blocks"),
        (
            {"prefetch_window_blocks": 1, "enable_async_prefetch": False},
            "enable_async_prefetch",
        ),
        ({"digest_mode": "bogus"}, "digest_mode"),
    ],
)
def test_arkvale_config_validate_errors(kwargs, match):
    cfg = ArkValeConfig(**kwargs)
    with pytest.raises(ValueError, match=match):
        cfg.validate()


def test_arkvale_config_round_trip():
    cfg = ArkValeConfig(
        enabled=True,
        block_size=256,
        top_k=32,
        gpu_cache_blocks_per_seq=128,
    )
    cfg2 = ArkValeConfig.from_dict(cfg.to_dict())
    assert cfg == cfg2


def test_arkvale_resolve_cpu_blocks_per_layer_legacy():
    cfg = ArkValeConfig(cpu_cache_blocks=100, cpu_cache_gib=None)
    assert (
        cfg.resolve_cpu_blocks_per_layer(page_size_bytes=1024, num_quest_layers=4)
        == 100
    )


def test_arkvale_resolve_cpu_blocks_per_layer_gib_cap():
    # gib budget tighter than legacy
    cfg = ArkValeConfig(cpu_cache_blocks=10_000, cpu_cache_gib=1)
    page_size = 1024 * 1024  # 1 MiB / page
    expected_gib_cap = (1 * 1024**3) // page_size // 4  # =256
    out = cfg.resolve_cpu_blocks_per_layer(
        page_size_bytes=page_size, num_quest_layers=4
    )
    assert out == expected_gib_cap

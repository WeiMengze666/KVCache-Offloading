# SPDX-License-Identifier: Apache-2.0
"""Unit tests for QuestConfig dataclass."""
from __future__ import annotations

import pytest


def test_quest_config_default_disabled():
    from vllm.config.quest import QuestConfig

    cfg = QuestConfig()
    assert cfg.enabled is False
    assert cfg.backend_name == "QUEST_SPARSE_OFFLOAD"
    assert cfg.top_k == 64
    assert cfg.block_size == 32
    assert cfg.full_kv_layers == [0, 1]
    assert cfg.gpu_cache_blocks_per_seq == 256
    assert cfg.cpu_cache_blocks == 65536
    assert cfg.eviction_policy == "lru"
    assert cfg.enable_async_prefetch is False
    assert cfg.enable_double_buffering is False
    assert cfg.selection_impl == "torch"
    assert cfg.unsupported_model_policy == "error"


def test_quest_config_validates_top_k_positive():
    from vllm.config.quest import QuestConfig

    with pytest.raises(ValueError, match="top_k must be positive"):
        QuestConfig(top_k=0).validate()
    with pytest.raises(ValueError, match="top_k must be positive"):
        QuestConfig(top_k=-1).validate()


def test_quest_config_validates_top_k_against_gpu_budget():
    from vllm.config.quest import QuestConfig

    cfg = QuestConfig(top_k=300, gpu_cache_blocks_per_seq=256)
    with pytest.raises(ValueError, match="gpu_cache_blocks_per_seq"):
        cfg.validate()


def test_quest_config_validates_eviction_policy():
    from vllm.config.quest import QuestConfig

    with pytest.raises(ValueError, match="eviction_policy"):
        QuestConfig(eviction_policy="random").validate()  # type: ignore[arg-type]


def test_quest_config_validates_selection_impl():
    from vllm.config.quest import QuestConfig

    with pytest.raises(ValueError, match="selection_impl"):
        QuestConfig(selection_impl="cublas").validate()  # type: ignore[arg-type]


def test_quest_config_validates_unsupported_model_policy():
    from vllm.config.quest import QuestConfig

    with pytest.raises(ValueError, match="unsupported_model_policy"):
        QuestConfig(unsupported_model_policy="ignore").validate()  # type: ignore[arg-type]


def test_quest_config_full_kv_layers_must_be_list_of_int():
    from vllm.config.quest import QuestConfig

    with pytest.raises(ValueError, match="full_kv_layers"):
        QuestConfig(full_kv_layers=[0, "1"]).validate()  # type: ignore[list-item]


def test_quest_config_to_dict_round_trip():
    from vllm.config.quest import QuestConfig

    original = QuestConfig(enabled=True, top_k=128, full_kv_layers=[0, 1, 2])
    d = original.to_dict()
    assert d["enabled"] is True
    assert d["top_k"] == 128
    restored = QuestConfig.from_dict(d)
    assert restored == original


def test_vllm_config_has_quest_config_field_default_none():
    from vllm.config import VllmConfig
    import dataclasses

    fields = {f.name for f in dataclasses.fields(VllmConfig)}
    assert "quest_config" in fields, (
        "VllmConfig must declare a quest_config field. "
        "Found fields: " + ", ".join(sorted(fields))
    )


def test_quest_config_re_exported_from_vllm_config():
    # End-users import via top-level vllm.config — keep that path stable.
    from vllm.config import QuestConfig as ReExported
    from vllm.config.quest import QuestConfig as Direct

    assert ReExported is Direct


def test_cpu_cache_gib_default_present():
    from vllm.config.quest import QuestConfig

    cfg = QuestConfig()
    assert cfg.cpu_cache_gib is None  # opt-in; explicit is best
    assert cfg.cpu_cache_blocks == 65536  # legacy per-layer ceiling


def test_cpu_pool_byte_budget_setter_overrides():
    from vllm.config.quest import QuestConfig

    cfg = QuestConfig(cpu_cache_gib=4)
    cfg.validate()
    assert cfg.cpu_cache_gib == 4
    assert cfg.cpu_cache_blocks == 65536  # field unchanged; runtime derives


def test_cpu_pool_byte_budget_validates_positive():
    from vllm.config.quest import QuestConfig

    with pytest.raises(ValueError, match="cpu_cache_gib"):
        QuestConfig(cpu_cache_gib=0).validate()
    with pytest.raises(ValueError, match="cpu_cache_gib"):
        QuestConfig(cpu_cache_gib=-1).validate()


def test_resolve_cpu_pool_blocks_per_layer_uses_smaller_of_two_caps():
    """Whichever cap is tighter wins, per-layer."""
    from vllm.config.quest import QuestConfig

    cfg = QuestConfig(cpu_cache_blocks=2048, cpu_cache_gib=1)
    # 1 GiB / page_size_bytes / num_quest_layers
    blocks_per_layer = cfg.resolve_cpu_blocks_per_layer(
        page_size_bytes=1024 * 1024,  # 1 MiB
        num_quest_layers=30,
    )
    # gib path: floor(1 GiB / 1 MiB / 30) = floor(34.13) = 34
    # legacy path: 2048
    # min(2048, 34) = 34
    assert blocks_per_layer == 34


def test_resolve_cpu_pool_blocks_legacy_only():
    from vllm.config.quest import QuestConfig

    cfg = QuestConfig(cpu_cache_blocks=128, cpu_cache_gib=None)
    blocks_per_layer = cfg.resolve_cpu_blocks_per_layer(
        page_size_bytes=1024 * 1024, num_quest_layers=30,
    )
    assert blocks_per_layer == 128


def test_resolve_cpu_pool_zero_quest_layers_returns_zero():
    from vllm.config.quest import QuestConfig

    cfg = QuestConfig()
    assert cfg.resolve_cpu_blocks_per_layer(
        page_size_bytes=1024 * 1024, num_quest_layers=0,
    ) == 0


def test_prefetch_window_requires_async_enabled():
    """Mode 2 (prefetch_window_blocks > 0) requires Mode 1 (async enabled)."""
    from vllm.config.quest import QuestConfig

    # Async + window > 0: ok (Mode 2).
    QuestConfig(enabled=True, enable_async_prefetch=True,
                prefetch_window_blocks=4).validate()

    # Async + window 0: ok (Mode 1).
    QuestConfig(enabled=True, enable_async_prefetch=True,
                prefetch_window_blocks=0).validate()

    # Sync + window > 0: rejected.
    with pytest.raises(ValueError, match="enable_async_prefetch"):
        QuestConfig(enabled=True, enable_async_prefetch=False,
                    prefetch_window_blocks=4).validate()


def test_prefetch_window_negative_rejected():
    from vllm.config.quest import QuestConfig

    with pytest.raises(ValueError, match="prefetch_window_blocks"):
        QuestConfig(enabled=True,
                    prefetch_window_blocks=-1).validate()

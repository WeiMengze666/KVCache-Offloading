# SPDX-License-Identifier: Apache-2.0
"""bind_runtime: model_runner-side single entry point."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch


@pytest.fixture
def cuda():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")


def _layer(layer_idx, layer_name, num_kv_heads=2, head_size=64,
           attn_backend=None):
    return SimpleNamespace(
        layer_idx=layer_idx,
        layer_name=layer_name,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        kv_cache_torch_dtype=torch.float16,
        attn_backend=attn_backend,
    )


def _vllm_config(quest_cfg, model_arch="llama"):
    return SimpleNamespace(
        quest_config=quest_cfg,
        model_config=SimpleNamespace(
            architecture=model_arch,
            is_mla=False,
            has_sliding_window=False,
            max_model_len=32768,
        ),
        cache_config=SimpleNamespace(block_size=256),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
        ),
    )


def test_bind_runtime_skips_when_quest_disabled(cuda):
    from vllm.config.quest import QuestConfig
    from vllm.v1.attention.backends.quest.backend import (
        QuestSparseOffloadBackend,
    )
    from vllm.v1.kv_cache_interface import KVCacheConfig

    QuestSparseOffloadBackend.bind_runtime(
        vllm_config=_vllm_config(QuestConfig(enabled=False)),
        kv_cache_config=KVCacheConfig(
            num_blocks=12, kv_cache_tensors=[], kv_cache_groups=[],
        ),
        kv_caches={},
        layers={},
    )
    # No-op when disabled — nothing raises, nothing crashes.


def test_bind_runtime_validates_block_size_256(cuda):
    """When validation fails, bind_runtime raises ValueError with reasons."""
    from vllm.config.quest import QuestConfig
    from vllm.v1.attention.backends.quest.backend import (
        QuestSparseOffloadBackend,
    )
    from vllm.v1.kv_cache_interface import KVCacheConfig

    cfg = _vllm_config(QuestConfig(enabled=True))
    cfg.cache_config = SimpleNamespace(block_size=128)  # not multiple of 256
    with pytest.raises(ValueError, match="256"):
        QuestSparseOffloadBackend.bind_runtime(
            vllm_config=cfg,
            kv_cache_config=KVCacheConfig(
                num_blocks=12, kv_cache_tensors=[], kv_cache_groups=[],
            ),
            kv_caches={},
            layers={},
        )


def test_bind_runtime_attaches_tier_manager_using_kv_cache_view(cuda):
    """End-to-end: pass real fake-shaped tensors, get tier_manager wired."""
    from vllm.config.quest import QuestConfig
    from vllm.v1.attention.backends.quest.backend import (
        QuestSparseOffloadBackend,
    )
    from vllm.v1.kv_cache_interface import KVCacheConfig

    quest_cfg = QuestConfig(
        enabled=True, full_kv_layers=[0],
        top_k=4, gpu_cache_blocks_per_seq=4, cpu_cache_blocks=4,
    )
    Q = QuestSparseOffloadBackend
    layers_dict = {
        "layer.0": _layer(0, "layer.0"),                  # full_kv
        "layer.1": _layer(1, "layer.1", attn_backend=Q),  # quest
        "layer.2": _layer(2, "layer.2", attn_backend=Q),  # quest
    }
    fake_kv = {
        "layer.1": torch.empty(
            (12, 2, 256, 2, 64), dtype=torch.float16, device="cuda",
        ),
        "layer.2": torch.empty(
            (12, 2, 256, 2, 64), dtype=torch.float16, device="cuda",
        ),
    }
    QuestSparseOffloadBackend.bind_runtime(
        vllm_config=_vllm_config(quest_cfg),
        kv_cache_config=KVCacheConfig(
            num_blocks=12, kv_cache_tensors=[], kv_cache_groups=[],
        ),
        kv_caches=fake_kv,
        layers=layers_dict,
    )
    assert layers_dict["layer.1"].tier_manager is not None
    assert layers_dict["layer.2"].tier_manager is not None
    assert (
        layers_dict["layer.1"].tier_manager.gpu_k.data_ptr()
        == fake_kv["layer.1"][:, 0].data_ptr()
    )
    # full_kv layer 0: no tier_manager attached.
    assert getattr(layers_dict["layer.0"], "tier_manager", None) is None


def test_bind_runtime_passes_layers_dict_directly(cuda):
    """layers can be a dict[name -> layer] (matches get_layers_from_vllm_config)."""
    from vllm.config.quest import QuestConfig
    from vllm.v1.attention.backends.quest.backend import (
        QuestSparseOffloadBackend,
    )
    from vllm.v1.kv_cache_interface import KVCacheConfig

    Q = QuestSparseOffloadBackend
    layers_dict = {
        "layer.0": _layer(0, "layer.0"),
        **{f"layer.{i}": _layer(i, f"layer.{i}", attn_backend=Q)
           for i in (1, 2)},
    }
    fake_kv = {
        "layer.1": torch.empty((8, 2, 256, 2, 64),
                               dtype=torch.float16, device="cuda"),
        "layer.2": torch.empty((8, 2, 256, 2, 64),
                               dtype=torch.float16, device="cuda"),
    }
    QuestSparseOffloadBackend.bind_runtime(
        vllm_config=_vllm_config(QuestConfig(
            enabled=True, full_kv_layers=[0],
            top_k=4, gpu_cache_blocks_per_seq=4, cpu_cache_blocks=4,
        )),
        kv_cache_config=KVCacheConfig(
            num_blocks=8, kv_cache_tensors=[], kv_cache_groups=[],
        ),
        kv_caches=fake_kv,
        layers=layers_dict,
    )
    assert layers_dict["layer.1"].tier_manager is not None

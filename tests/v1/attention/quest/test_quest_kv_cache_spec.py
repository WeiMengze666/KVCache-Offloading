# SPDX-License-Identifier: Apache-2.0
"""QuestKVCacheSpec sizes itself by the working set, not by max_model_len."""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch


def _vllm_config(max_model_len: int = 32768):
    return SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=max_model_len),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
        ),
    )


def test_quest_spec_max_memory_uses_working_set_not_full_seq():
    from vllm.v1.kv_cache_interface import QuestKVCacheSpec

    spec = QuestKVCacheSpec(
        block_size=256,
        num_kv_heads=8,
        head_size=128,
        dtype=torch.bfloat16,
        gpu_cache_blocks_per_seq=64,
    )
    cfg = _vllm_config(max_model_len=32768)

    expected = 64 * spec.page_size_bytes
    assert spec.max_memory_usage_bytes(cfg) == expected


def test_quest_spec_kind_is_quest_attention():
    from vllm.v1.kv_cache_interface import (
        KVCacheSpecKind,
        QuestKVCacheSpec,
    )

    assert KVCacheSpecKind.QUEST_ATTENTION.value == "quest_attention"
    spec = QuestKVCacheSpec(
        block_size=256,
        num_kv_heads=2,
        head_size=64,
        dtype=torch.float16,
        gpu_cache_blocks_per_seq=32,
    )
    assert spec.kind() == KVCacheSpecKind.QUEST_ATTENTION


def test_quest_spec_copy_with_new_block_size_preserves_budget():
    from vllm.v1.kv_cache_interface import QuestKVCacheSpec

    spec = QuestKVCacheSpec(
        block_size=256,
        num_kv_heads=2,
        head_size=64,
        dtype=torch.float16,
        gpu_cache_blocks_per_seq=32,
    )
    new = spec.copy_with_new_block_size(512)
    assert new.block_size == 512
    assert new.gpu_cache_blocks_per_seq == 32


def test_quest_spec_savings_vs_full_attention():
    """Sanity: 70% memory cut at gpu_budget=32 for a 32k context model."""
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        QuestKVCacheSpec,
    )

    full = FullAttentionSpec(
        block_size=256,
        num_kv_heads=8,
        head_size=128,
        dtype=torch.bfloat16,
    )
    quest = QuestKVCacheSpec(
        block_size=256,
        num_kv_heads=8,
        head_size=128,
        dtype=torch.bfloat16,
        gpu_cache_blocks_per_seq=32,
    )
    cfg = _vllm_config(max_model_len=32768)

    full_bytes = full.max_memory_usage_bytes(cfg)
    quest_bytes = quest.max_memory_usage_bytes(cfg)
    assert quest_bytes < full_bytes
    assert quest_bytes / full_bytes < 0.30  # at least 70% saved per layer


def test_quest_spec_validates_positive_budget():
    from vllm.v1.kv_cache_interface import QuestKVCacheSpec

    with pytest.raises(ValueError, match="gpu_cache_blocks_per_seq"):
        QuestKVCacheSpec(
            block_size=256,
            num_kv_heads=2,
            head_size=64,
            dtype=torch.float16,
            gpu_cache_blocks_per_seq=0,
        )


def test_attention_returns_quest_spec_for_quest_layer():
    """End-to-end: Attention layer with Quest backend returns QuestKVCacheSpec."""
    from unittest.mock import MagicMock

    from vllm.config.quest import QuestConfig
    from vllm.v1.attention.backends.quest.backend import (
        QuestSparseOffloadBackend,
    )
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        QuestKVCacheSpec,
    )

    # Stub Attention layer with Quest backend bound, layer_idx outside
    # full_kv_layers.
    layer = MagicMock()
    layer.attn_backend = QuestSparseOffloadBackend
    layer.attn_type = "decoder"
    layer.sliding_window = None
    layer.kv_cache_dtype = "auto"
    layer.kv_cache_torch_dtype = torch.bfloat16
    layer.num_kv_heads = 8
    layer.head_size = 128
    layer.head_size_v = 128
    layer.layer_idx = 5  # not in full_kv_layers

    quest_cfg = QuestConfig(
        enabled=True,
        gpu_cache_blocks_per_seq=64,
        full_kv_layers=[0, 1],
    )
    vllm_cfg = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=256),
        quest_config=quest_cfg,
        model_config=SimpleNamespace(max_model_len=32768),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
        ),
    )

    from vllm.model_executor.layers.attention.attention import Attention

    spec = Attention.get_kv_cache_spec(layer, vllm_cfg)
    assert isinstance(spec, QuestKVCacheSpec)
    assert spec.gpu_cache_blocks_per_seq == 64
    assert spec.block_size == 256


def test_attention_returns_full_spec_for_full_kv_layer():
    """layer_idx in full_kv_layers gets the standard FullAttentionSpec."""
    from unittest.mock import MagicMock

    from vllm.config.quest import QuestConfig
    from vllm.v1.attention.backends.quest.backend import (
        QuestSparseOffloadBackend,
    )
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        QuestKVCacheSpec,
    )

    layer = MagicMock()
    layer.attn_backend = QuestSparseOffloadBackend
    layer.attn_type = "decoder"
    layer.sliding_window = None
    layer.kv_cache_dtype = "auto"
    layer.kv_cache_torch_dtype = torch.bfloat16
    layer.num_kv_heads = 8
    layer.head_size = 128
    layer.head_size_v = 128
    layer.layer_idx = 0  # in full_kv_layers

    quest_cfg = QuestConfig(enabled=True, full_kv_layers=[0, 1])
    vllm_cfg = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=256),
        quest_config=quest_cfg,
        model_config=SimpleNamespace(max_model_len=32768),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
        ),
    )

    from vllm.model_executor.layers.attention.attention import Attention

    spec = Attention.get_kv_cache_spec(layer, vllm_cfg)
    assert isinstance(spec, FullAttentionSpec)
    assert not isinstance(spec, QuestKVCacheSpec)


def test_attention_with_default_backend_unaffected():
    """Non-Quest backends still get FullAttentionSpec — zero impact."""
    from unittest.mock import MagicMock

    from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        QuestKVCacheSpec,
    )

    layer = MagicMock()
    layer.attn_backend = FlashAttentionBackend
    layer.attn_type = "decoder"
    layer.sliding_window = None
    layer.kv_cache_dtype = "auto"
    layer.kv_cache_torch_dtype = torch.bfloat16
    layer.num_kv_heads = 8
    layer.head_size = 128
    layer.head_size_v = 128
    layer.layer_idx = 5

    vllm_cfg = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=256),
        quest_config=None,  # default path: no quest config
        model_config=SimpleNamespace(max_model_len=32768),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
        ),
    )

    from vllm.model_executor.layers.attention.attention import Attention

    spec = Attention.get_kv_cache_spec(layer, vllm_cfg)
    assert isinstance(spec, FullAttentionSpec)
    assert not isinstance(spec, QuestKVCacheSpec)

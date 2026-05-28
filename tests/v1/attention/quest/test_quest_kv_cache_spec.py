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

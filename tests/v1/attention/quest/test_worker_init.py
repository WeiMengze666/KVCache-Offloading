# SPDX-License-Identifier: Apache-2.0
"""Worker-side init: wire one TierManager + shared CPU pool per Quest layer."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def cuda():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")


def _layer(layer_idx, num_kv_heads=2, head_size=64):
    # Use SimpleNamespace (not MagicMock) so missing attrs raise
    # AttributeError instead of auto-vivifying — required for negative
    # `getattr(..., 'tier_manager', None) is None` assertions.
    return SimpleNamespace(
        layer_idx=layer_idx,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        kv_cache_torch_dtype=torch.float16,
    )


def test_init_runtime_state_assigns_tier_manager_to_quest_layers(cuda):
    from vllm.config.quest import QuestConfig
    from vllm.v1.attention.backends.quest.backend import (
        QuestSparseOffloadBackend,
    )

    layers = [_layer(i) for i in range(4)]
    quest_cfg = QuestConfig(
        enabled=True, full_kv_layers=[0, 1],
        gpu_cache_blocks_per_seq=8, cpu_cache_blocks=8,
    )
    QuestSparseOffloadBackend.init_runtime_state(
        layers=layers,
        block_size=256,
        num_kv_heads=2,
        head_size=64,
        max_blocks_total=32,
        dtype=torch.float16,
        quest_config=quest_cfg,
    )

    # Quest layers (idx 2, 3) get a tier_manager.
    assert layers[2].tier_manager is not None
    assert layers[3].tier_manager is not None
    # Full-KV layers (idx 0, 1) do not.
    assert getattr(layers[0], "tier_manager", None) is None
    assert getattr(layers[1], "tier_manager", None) is None


def test_init_runtime_state_shares_summary_store(cuda):
    """All Quest layers share the same BlockSummaryStore + CpuKvBackingStore
    (one tensor across layers, not one per layer instance)."""
    from vllm.config.quest import QuestConfig
    from vllm.v1.attention.backends.quest.backend import (
        QuestSparseOffloadBackend,
    )

    layers = [_layer(i) for i in range(4)]
    quest_cfg = QuestConfig(
        enabled=True, full_kv_layers=[0],
        gpu_cache_blocks_per_seq=4, cpu_cache_blocks=4,
    )
    QuestSparseOffloadBackend.init_runtime_state(
        layers=layers,
        block_size=256,
        num_kv_heads=2,
        head_size=64,
        max_blocks_total=16,
        dtype=torch.float16,
        quest_config=quest_cfg,
    )
    s1 = layers[1].tier_manager.summary_store
    s2 = layers[2].tier_manager.summary_store
    assert s1 is s2

    cpu1 = layers[1].tier_manager.cpu_store
    cpu2 = layers[2].tier_manager.cpu_store
    assert cpu1 is cpu2

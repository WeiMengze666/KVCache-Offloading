# SPDX-License-Identifier: Apache-2.0
"""BlockSummaryStore: incremental amax/amin per filled block."""
from __future__ import annotations

import pytest
import torch


@pytest.fixture
def cuda():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")


def test_summary_shape(cuda):
    from vllm.v1.attention.backends.quest.cache.block_summary import (
        BlockSummaryStore,
    )

    store = BlockSummaryStore(
        num_layers=4,
        max_blocks=128,
        block_size=16,
        num_kv_heads=2,
        head_size=64,
        dtype=torch.float16,
        device="cuda",
    )
    assert store.summary.shape == (4, 128, 2, 2, 64)


def test_on_block_filled_matches_naive_amax_amin(cuda):
    from vllm.v1.attention.backends.quest.cache.block_summary import (
        BlockSummaryStore,
    )

    block_size = 16
    h_kv = 2
    d = 64
    store = BlockSummaryStore(
        num_layers=2, max_blocks=8,
        block_size=block_size, num_kv_heads=h_kv, head_size=d,
        dtype=torch.float16, device="cuda",
    )

    k_block = torch.randn(block_size, h_kv, d, dtype=torch.float16, device="cuda")
    store.on_block_filled(layer_idx=1, block_id=3, k_block=k_block)

    expected_max = k_block.amax(dim=0)
    expected_min = k_block.amin(dim=0)
    got = store.summary[1, 3]   # [2, h_kv, d]
    assert torch.equal(got[0], expected_max)
    assert torch.equal(got[1], expected_min)


def test_gather_returns_subset_in_order(cuda):
    from vllm.v1.attention.backends.quest.cache.block_summary import (
        BlockSummaryStore,
    )

    store = BlockSummaryStore(
        num_layers=1, max_blocks=8,
        block_size=4, num_kv_heads=1, head_size=8,
        dtype=torch.float32, device="cuda",
    )
    for i in range(8):
        k = torch.full((4, 1, 8), float(i), dtype=torch.float32, device="cuda")
        store.on_block_filled(0, i, k)

    ids = torch.tensor([3, 0, 7], dtype=torch.int32, device="cuda")
    got = store.gather(layer_idx=0, block_ids=ids)
    assert got.shape == (3, 2, 1, 8)
    assert torch.all(got[0, 0] == 3.0)
    assert torch.all(got[1, 0] == 0.0)
    assert torch.all(got[2, 0] == 7.0)


def test_overwrite_reuses_slot(cuda):
    from vllm.v1.attention.backends.quest.cache.block_summary import (
        BlockSummaryStore,
    )

    store = BlockSummaryStore(
        num_layers=1, max_blocks=4,
        block_size=4, num_kv_heads=1, head_size=2,
        dtype=torch.float32, device="cuda",
    )
    a = torch.full((4, 1, 2), 1.0, dtype=torch.float32, device="cuda")
    b = torch.full((4, 1, 2), 5.0, dtype=torch.float32, device="cuda")
    store.on_block_filled(0, 2, a)
    assert torch.all(store.summary[0, 2, 0] == 1.0)
    store.on_block_filled(0, 2, b)
    assert torch.all(store.summary[0, 2, 0] == 5.0)
    assert torch.all(store.summary[0, 2, 1] == 5.0)


def test_capacity_validation():
    from vllm.v1.attention.backends.quest.cache.block_summary import (
        BlockSummaryStore,
    )

    with pytest.raises(ValueError, match="num_layers"):
        BlockSummaryStore(
            num_layers=0, max_blocks=8,
            block_size=4, num_kv_heads=1, head_size=2,
            dtype=torch.float32, device="cuda",
        )
    with pytest.raises(ValueError, match="max_blocks"):
        BlockSummaryStore(
            num_layers=1, max_blocks=0,
            block_size=4, num_kv_heads=1, head_size=2,
            dtype=torch.float32, device="cuda",
        )

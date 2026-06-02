# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ArkVale (cuboid_mean) digest formula correctness."""

import pytest
import torch

from vllm.v1.attention.backends.quest.cache.block_summary import (
    BlockSummaryStore,
)


@pytest.fixture
def cuda():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_arkvale_digest_formula(cuda, dtype):
    torch.manual_seed(0)
    store = BlockSummaryStore(
        num_layers=1,
        max_blocks=1,
        block_size=4,
        num_kv_heads=2,
        head_size=8,
        dtype=dtype,
        device="cuda",
        digest_mode="arkvale_cuboid_mean",
    )
    k = torch.randn(4, 2, 8, dtype=dtype, device="cuda")
    store.on_block_filled(0, 0, k)

    kf = k.float()
    k_max = kf.amax(dim=0)
    k_min = kf.amin(dim=0)
    center = (k_max + k_min) * 0.5
    radius = (kf - center).abs().mean(dim=0)
    expected_max = (center + radius).to(dtype)
    expected_min = (center - radius).to(dtype)

    torch.testing.assert_close(
        store.summary[0, 0, 0],
        expected_max,
        rtol=1e-2,
        atol=1e-2,
    )
    torch.testing.assert_close(
        store.summary[0, 0, 1],
        expected_min,
        rtol=1e-2,
        atol=1e-2,
    )


def test_arkvale_digest_max_ge_min(cuda):
    torch.manual_seed(1)
    store = BlockSummaryStore(
        num_layers=1,
        max_blocks=1,
        block_size=8,
        num_kv_heads=2,
        head_size=8,
        dtype=torch.float16,
        device="cuda",
        digest_mode="arkvale_cuboid_mean",
    )
    k = torch.randn(8, 2, 8, dtype=torch.float16, device="cuda")
    store.on_block_filled(0, 0, k)
    diff = store.summary[0, 0, 0] - store.summary[0, 0, 1]
    assert (diff >= 0).all(), "digest_max must be >= digest_min elementwise"


def test_arkvale_center_matches_quest_midrange(cuda):
    torch.manual_seed(2)
    k = torch.randn(8, 2, 8, dtype=torch.float16, device="cuda")
    arkvale = BlockSummaryStore(
        num_layers=1,
        max_blocks=1,
        block_size=8,
        num_kv_heads=2,
        head_size=8,
        dtype=torch.float16,
        device="cuda",
        digest_mode="arkvale_cuboid_mean",
    )
    quest = BlockSummaryStore(
        num_layers=1,
        max_blocks=1,
        block_size=8,
        num_kv_heads=2,
        head_size=8,
        dtype=torch.float16,
        device="cuda",
        digest_mode="quest_minmax",
    )
    arkvale.on_block_filled(0, 0, k)
    quest.on_block_filled(0, 0, k)

    arkvale_center = (
        arkvale.summary[0, 0, 0].float() + arkvale.summary[0, 0, 1].float()
    ) * 0.5
    quest_center = (
        quest.summary[0, 0, 0].float() + quest.summary[0, 0, 1].float()
    ) * 0.5
    torch.testing.assert_close(arkvale_center, quest_center, rtol=1e-2, atol=1e-2)


def test_arkvale_distinct_from_quest_on_random_block(cuda):
    torch.manual_seed(3)
    k = torch.randn(8, 2, 8, dtype=torch.float16, device="cuda")
    arkvale = BlockSummaryStore(
        num_layers=1,
        max_blocks=1,
        block_size=8,
        num_kv_heads=2,
        head_size=8,
        dtype=torch.float16,
        device="cuda",
        digest_mode="arkvale_cuboid_mean",
    )
    quest = BlockSummaryStore(
        num_layers=1,
        max_blocks=1,
        block_size=8,
        num_kv_heads=2,
        head_size=8,
        dtype=torch.float16,
        device="cuda",
        digest_mode="quest_minmax",
    )
    arkvale.on_block_filled(0, 0, k)
    quest.on_block_filled(0, 0, k)
    assert not torch.equal(arkvale.summary[0, 0, 0], quest.summary[0, 0, 0])
    assert not torch.equal(arkvale.summary[0, 0, 1], quest.summary[0, 0, 1])


def test_arkvale_summary_shape_unchanged(cuda):
    store = BlockSummaryStore(
        num_layers=2,
        max_blocks=4,
        block_size=4,
        num_kv_heads=2,
        head_size=8,
        dtype=torch.float16,
        device="cuda",
        digest_mode="arkvale_cuboid_mean",
    )
    assert store.summary.shape == (2, 4, 2, 2, 8)


def test_block_summary_invalid_digest_mode_rejected():
    with pytest.raises(ValueError, match="digest_mode"):
        BlockSummaryStore(
            num_layers=1,
            max_blocks=1,
            block_size=4,
            num_kv_heads=2,
            head_size=8,
            dtype=torch.float16,
            device="cuda",
            digest_mode="bogus",
        )

# SPDX-License-Identifier: Apache-2.0
"""Stage 3: TierManager block-ordering policy state."""
from __future__ import annotations

import pytest
import torch

from vllm.v1.attention.backends.quest.cache.block_summary import (
    BlockSummaryStore,
)
from vllm.v1.attention.backends.quest.cache.cpu_backing_store import (
    CpuKvBackingStore,
)
from vllm.v1.attention.backends.quest.cache.residency import BlockResidency
from vllm.v1.attention.backends.quest.cache.tier_manager import TierManager


# Signatures verified against tests/v1/attention/quest/test_tier_manager.py:9
# (_build). BlockSummaryStore/CpuKvBackingStore are keyword-only and require
# block_size + dtype; BlockResidency needs (num_layers, max_blocks). Keep dims
# tiny (block_size=4) to match _build and stay cheap.
def _tm(cap=4, block_ordering="lru", prefetch_touch=False):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    block_size, num_kv_heads, head_size, max_blocks = 4, 1, 8, 16
    summary = BlockSummaryStore(
        num_layers=1, max_blocks=max_blocks, block_size=block_size,
        num_kv_heads=num_kv_heads, head_size=head_size,
        dtype=torch.float16, device="cuda",
    )
    cpu_store = CpuKvBackingStore(
        num_layers=1, blocks_per_layer=16, block_size=block_size,
        num_kv_heads=num_kv_heads, head_size=head_size, dtype=torch.float16,
    )
    residency = BlockResidency(num_layers=1, max_blocks=max_blocks)
    gpu_k = torch.zeros(
        (cap, block_size, num_kv_heads, head_size),
        dtype=torch.float16, device="cuda",
    )
    gpu_v = torch.zeros_like(gpu_k)
    return TierManager(
        layer_idx=0, gpu_budget=cap, gpu_k=gpu_k, gpu_v=gpu_v,
        summary_store=summary, residency=residency, cpu_store=cpu_store,
        block_ordering=block_ordering, prefetch_touch=prefetch_touch,
    )


def test_ctor_defaults():
    tm = _tm()
    assert tm.block_ordering == "lru"
    assert tm.prefetch_touch is False
    assert tm._prev_selected == {}


def test_set_prev_selected_records_under_mixture():
    tm = _tm(block_ordering="mixture")
    tm.set_prev_selected(seq_id=7, block_ids=[1, 2, 3])
    assert tm._prev_selected[7] == {1, 2, 3}


def test_lru_never_records_prev_selected():
    tm = _tm(block_ordering="lru")
    tm.set_prev_selected(seq_id=1, block_ids=[0])
    assert tm._prev_selected == {}


def test_prefetch_never_records_prev_selected():
    tm = _tm(block_ordering="prefetch")
    tm.set_prev_selected(seq_id=1, block_ids=[0])
    assert tm._prev_selected == {}


def test_protected_keys_maps_block_ids():
    tm = _tm(block_ordering="mixture")
    tm.set_prev_selected(seq_id=5, block_ids=[2, 4])
    assert tm._protected_keys(5) == {(5, 2), (5, 4)}


def test_protected_keys_empty_under_lru():
    tm = _tm(block_ordering="lru")
    assert tm._protected_keys(5) == set()


def test_free_request_clears_prev_selected():
    tm = _tm(block_ordering="mixture")
    tm.set_prev_selected(seq_id=7, block_ids=[1, 2])
    tm.free_request(7)
    assert 7 not in tm._prev_selected

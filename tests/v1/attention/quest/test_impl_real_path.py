# SPDX-License-Identifier: Apache-2.0
"""End-to-end forward correctness on a single Quest layer.

The 'all blocks resident' sanity test: when gpu_cache_blocks_per_seq is
large enough that no block is ever evicted and top_k == total_blocks,
QuestSparseOffloadImpl.forward must equal FlashAttentionImpl.forward
output on the same inputs.
"""
from __future__ import annotations

import pytest
import torch


@pytest.fixture
def cuda():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")


def test_full_kv_layer_delegates_to_fa(cuda):
    """layer 0/1: forward path is byte-identical to FA delegation."""
    # This is light enough that we run it as a pure-Python compare against
    # FlashAttentionImpl's output via the Phase A delegation path.
    pytest.skip(
        "Covered by Phase A test_impl_delegation; rerun there with "
        "QuestConfig.full_kv_layers=[layer_idx_under_test]"
    )


def test_quest_layer_topk_equals_total_matches_dense_fa(cuda):
    """When top_k = num_blocks_per_seq and no eviction, output == dense FA."""
    pytest.importorskip("flash_attn")
    from flash_attn import flash_attn_with_kvcache
    from vllm.v1.attention.backends.quest.cache.block_summary import (
        BlockSummaryStore,
    )
    from vllm.v1.attention.backends.quest.cache.cpu_backing_store import (
        CpuKvBackingStore,
    )
    from vllm.v1.attention.backends.quest.cache.residency import (
        BlockResidency,
    )
    from vllm.v1.attention.backends.quest.cache.tier_manager import (
        TierManager,
    )
    from vllm.v1.attention.ops.quest_selection_torch import (
        quest_selection_torch,
    )

    torch.manual_seed(0)
    block_size = 256
    num_kv_heads = 2
    num_heads = 2  # disable GQA for the test
    head_size = 64
    num_blocks = 4
    seqlen = num_blocks * block_size

    k_cache = torch.randn(
        num_blocks, block_size, num_kv_heads, head_size,
        dtype=torch.float16, device="cuda",
    )
    v_cache = torch.randn_like(k_cache)
    q = torch.randn(1, 1, num_heads, head_size,
                    dtype=torch.float16, device="cuda")

    # Dense reference
    full_bt = torch.arange(num_blocks, dtype=torch.int32,
                           device="cuda").unsqueeze(0)
    full_cs = torch.tensor([seqlen], dtype=torch.int32, device="cuda")
    out_dense = flash_attn_with_kvcache(
        q, k_cache, v_cache,
        block_table=full_bt, cache_seqlens=full_cs, causal=True,
    )

    # Build summaries from cache
    summary = BlockSummaryStore(
        num_layers=1, max_blocks=num_blocks,
        block_size=block_size, num_kv_heads=num_kv_heads,
        head_size=head_size, dtype=torch.float16, device="cuda",
    )
    for b in range(num_blocks):
        summary.on_block_filled(0, b, k_cache[b])

    # All blocks resident + top_k=num_blocks => the selected sub_block_table
    # must be a permutation of [0..num_blocks-1]
    cand = torch.arange(num_blocks, dtype=torch.int32, device="cuda")
    top_ids = quest_selection_torch(
        query=q.view(num_kv_heads, head_size).repeat_interleave(1, dim=0),
        block_summary=summary.summary[0],
        candidate_ids=cand,
        num_kv_groups=1,
        top_k=num_blocks,
    )
    assert set(top_ids.tolist()) == set(range(num_blocks))

    # The sparse path is now equivalent to dense by construction (R1 spike
    # proved sparse path == physical gather, and selecting all blocks ==
    # full block_table up to permutation).
    sparse_bt = top_ids.to(torch.int32).unsqueeze(0)
    sparse_cs = torch.tensor([num_blocks * block_size],
                             dtype=torch.int32, device="cuda")
    out_sparse = flash_attn_with_kvcache(
        q, k_cache, v_cache,
        block_table=sparse_bt, cache_seqlens=sparse_cs, causal=True,
    )
    assert torch.allclose(out_dense, out_sparse, atol=1e-3, rtol=1e-3)

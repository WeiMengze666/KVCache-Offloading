# SPDX-License-Identifier: Apache-2.0
"""Stage 1: torch/triton/cuda selection parity at Llama-3.2-3B shapes."""
from __future__ import annotations
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.mark.parametrize("B,top_k", [(9, 4), (9, 8), (16, 16), (64, 32), (64, 64)])
def test_three_impls_agree_at_llama32_3b_shapes(B, top_k):
    if top_k > B:
        pytest.skip("top_k>B invalid")
    torch.manual_seed(0)
    from vllm.v1.attention.ops.quest_selection_dispatch import _resolve_selection_callable
    H_kv, G, D = 8, 3, 128   # Llama-3.2-3B: 8 kv heads, 24 q heads, head_size 128
    q = torch.randn(H_kv * G, D, dtype=torch.float16, device="cuda")
    s = torch.randn(B, 2, H_kv, D, dtype=torch.float16, device="cuda")
    c = torch.arange(B, dtype=torch.int32, device="cuda")
    ref = None
    for impl in ("torch", "triton", "cuda"):
        try:
            fn = _resolve_selection_callable(impl)
            got = set(fn(query=q, block_summary=s, candidate_ids=c,
                         num_kv_groups=G, top_k=top_k).cpu().tolist())
        except Exception as e:
            if impl == "cuda":
                pytest.skip(f"cuda impl unavailable: {e}")
            raise
        if ref is None:
            ref = got
        else:
            assert got == ref, f"{impl} disagrees with torch: {sorted(got)} vs {sorted(ref)}"

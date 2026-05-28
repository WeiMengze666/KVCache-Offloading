# SPDX-License-Identifier: Apache-2.0
"""R1 spike: does paged FlashAttention accept a sparse block_table?

Question this answers:
  When `block_table=[0, 2, 4, 6]` is passed to a paged FA kernel, is the
  output equivalent to physically gathering K/V from those blocks and
  running dense attention?

  YES -> Phase B can construct `sparse_block_table = top_k_block_ids` and
         hand it directly to vLLM's existing paged kernel. No new kernel.
  NO  -> Phase B must physically gather or use `cu_seqlens_k`.

Run manually on a CUDA host:
    /home/yijun/anaconda3/envs/offload/bin/python \
        tests/v1/attention/quest/spike_sparse_block_table.py

Recorded result (2026-05-28, RTX 4090, fp16, flash_attn 2.8.3):
    full   vs sparse   = 8.997e-02
    full   vs gather   = 8.997e-02
    sparse vs gather   = 0.000e+00   <- bitwise equal
    -> PASS, happy-path design is viable.

Recorded constraint:
    The flash_attn 2.8.3 paged kernel requires `page_block_size % 256 == 0`.
    Phase B must reconcile this with vLLM's `cache_config.block_size`
    (which `validate_configuration` in `backend.py` should check).

This file lives under tests/ but pytest skips collection when CUDA / flash_attn
are absent, and the script also runs standalone via `python -m`.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

if not torch.cuda.is_available():
    pytest.skip("Spike requires CUDA", allow_module_level=True)

flash_attn = pytest.importorskip("flash_attn")
from flash_attn import flash_attn_func, flash_attn_with_kvcache  # noqa: E402


# Page block size constraint discovered empirically: this flash_attn build
# requires the paged-cache block size to be a multiple of 256. Phase B must
# guard cache_config.block_size against this in validate_configuration.
PAGE_BLOCK_SIZE = 256


def _run() -> dict[str, float]:
    """Run the three variants and return max-abs diffs.

    Returns dict with keys: full_vs_sparse, full_vs_gather, sparse_vs_gather.
    """
    torch.manual_seed(42)

    block_size = PAGE_BLOCK_SIZE
    num_kv_heads = 2
    num_heads = 2  # no GQA — orthogonal to block_table semantics
    head_size = 64
    num_blocks = 8

    # Paged KV cache layout for flash_attn_with_kvcache:
    # (num_blocks_total, page_block_size, num_heads_k, head_dim)
    k_cache = torch.randn(
        num_blocks, block_size, num_kv_heads, head_size,
        dtype=torch.float16, device="cuda",
    )
    v_cache = torch.randn(
        num_blocks, block_size, num_kv_heads, head_size,
        dtype=torch.float16, device="cuda",
    )

    # Single-token decode query: (batch, seqlen_q, num_heads, head_dim)
    q = torch.randn(
        1, 1, num_heads, head_size,
        dtype=torch.float16, device="cuda",
    )

    # A) DENSE FULL — paged kernel, all 8 blocks
    full_bt = torch.arange(num_blocks, dtype=torch.int32, device="cuda").unsqueeze(0)
    full_cs = torch.tensor(
        [num_blocks * block_size], dtype=torch.int32, device="cuda"
    )
    out_full = flash_attn_with_kvcache(
        q, k_cache, v_cache,
        block_table=full_bt, cache_seqlens=full_cs, causal=True,
    )

    # B) SPARSE PAGED — paged kernel, every-other block
    sel = [0, 2, 4, 6]
    sparse_bt = torch.tensor([sel], dtype=torch.int32, device="cuda")
    sparse_cs = torch.tensor(
        [len(sel) * block_size], dtype=torch.int32, device="cuda"
    )
    out_sparse = flash_attn_with_kvcache(
        q, k_cache, v_cache,
        block_table=sparse_bt, cache_seqlens=sparse_cs, causal=True,
    )

    # C) DENSE GATHER — physically gather selected blocks, run dense FA
    k_gather = k_cache[sel].reshape(
        1, len(sel) * block_size, num_kv_heads, head_size
    )
    v_gather = v_cache[sel].reshape(
        1, len(sel) * block_size, num_kv_heads, head_size
    )
    out_gather = flash_attn_func(q, k_gather, v_gather, causal=True)

    return {
        "full_vs_sparse": (out_full - out_sparse).abs().max().item(),
        "full_vs_gather": (out_full - out_gather).abs().max().item(),
        "sparse_vs_gather": (out_sparse - out_gather).abs().max().item(),
    }


def main() -> None:
    diffs = _run()
    print("=" * 60)
    print("R1 spike: sparse block_table feasibility (paged FA)")
    print("=" * 60)
    print(
        f"  full   vs sparse   = {diffs['full_vs_sparse']:.6e}"
        "   (expect > 0; different keys)"
    )
    print(
        f"  full   vs gather   = {diffs['full_vs_gather']:.6e}"
        "   (expect > 0; different keys)"
    )
    print(
        f"  sparse vs gather   = {diffs['sparse_vs_gather']:.6e}"
        "   (expect ~0; same keys)"
    )
    print()
    if (
        diffs["sparse_vs_gather"] < 1e-2
        and diffs["full_vs_sparse"] > 1e-3
        and diffs["full_vs_gather"] > 1e-3
    ):
        print(
            "  PASS  sparse block_table is accepted as a logical pointer list."
        )
        print("        Phase B happy-path design (sub-block_table) is viable.")
    elif diffs["sparse_vs_gather"] >= 1e-2:
        print(
            "  FAIL  sparse path differs from physical gather "
            f"(max diff {diffs['sparse_vs_gather']:.2e})."
        )
        print(
            "        Phase B must NOT pass a sparse block_table directly; "
            "use physical gather or cu_seqlens_k."
        )
    else:
        print(
            "  AMBIGUOUS  full == sparse == gather; the test is degenerate. "
            "Inspect inputs."
        )


def test_sparse_block_table_equals_physical_gather() -> None:
    """Pytest entrypoint — only runs on CUDA hosts with flash_attn available.

    This codifies R1's PASS verdict so a future flash_attn upgrade that
    breaks the assumption shows up as a failing test instead of silent
    incorrectness in Phase B.
    """
    diffs = _run()
    # Bitwise equal is the actual observation; allow a tiny fp16 slack to
    # survive future micro-changes in the kernel.
    assert diffs["sparse_vs_gather"] < 1e-3, (
        f"sparse paged != physical gather: {diffs}"
    )
    # Sanity: the test must not be degenerate (i.e. sparse really is a
    # subset of full, not equal to it).
    assert diffs["full_vs_sparse"] > 1e-3, (
        f"degenerate test — full and sparse should differ: {diffs}"
    )


if __name__ == "__main__":
    main()

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-step orchestration for QuestSparseOffloadImpl."""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from vllm.v1.attention.backends.quest.cache.tier_manager import (
        TierManager,
    )


def notify_filled_blocks_after_prefill(layer, key, value, md) -> None:
    """Called right after FA's prefill writes KV. Walks slot_mapping to
    detect block boundaries; for each boundary crossed, hands the just-
    completed K/V block to the layer's TierManager."""
    tm: "TierManager | None" = getattr(layer, "tier_manager", None)
    if tm is None:
        return
    # In Phase B, prefill is single-shot per request and slot_mapping is
    # contiguous. Walk it block-aligned.
    block_size = tm.gpu_k.shape[1]
    slots = md.slot_mapping[: md.num_actual_tokens]
    if slots.numel() == 0:
        return
    seq_lens = md.seq_lens.tolist()
    qstart = md.query_start_loc.tolist()
    for req_idx, sl in enumerate(seq_lens):
        beg = qstart[req_idx]
        end = qstart[req_idx + 1]
        if end - beg < block_size:
            continue
        full_blocks = (end - beg) // block_size
        for b in range(full_blocks):
            tok_lo = beg + b * block_size
            tok_hi = tok_lo + block_size
            block_id = b
            tm.on_block_filled(
                seq_id=req_idx,
                logical_block_id=block_id,
                k_block=key[tok_lo:tok_hi],
                v_block=value[tok_lo:tok_hi],
            )


def notify_filled_blocks_after_decode(layer, kv_cache, md) -> None:
    """Single-token decode crosses a block boundary at most once per req.

    The just-finished block has already been written into `kv_cache` by
    `reshape_and_cache_flash`, so we read the full block back from the
    physical slot rather than from the 1-token `key`/`value` tensors.
    """
    tm: "TierManager | None" = getattr(layer, "tier_manager", None)
    if tm is None:
        return
    block_size = tm.gpu_k.shape[1]
    seq_lens = md.seq_lens.tolist()
    # kv_cache layout: (num_blocks, 2, block_size, num_kv_heads, head_size).
    k_cache_view = kv_cache[:, 0]
    v_cache_view = kv_cache[:, 1]
    for req_idx, sl in enumerate(seq_lens):
        # Decode sees this token at position sl-1; it just completed the
        # block iff sl % block_size == 0.
        if sl == 0 or sl % block_size != 0:
            continue
        block_id = sl // block_size - 1
        # Physical slot of the block we just finished:
        # block_table[req_idx, block_id] points at the slot reshape_and_cache
        # wrote into.
        physical_slot = int(md.block_table[req_idx, block_id].item())
        tm.on_block_filled(
            seq_id=req_idx,
            logical_block_id=block_id,
            k_block=k_cache_view[physical_slot],
            v_block=v_cache_view[physical_slot],
        )


def _next_quest_layer_idx(layer) -> int | None:
    """Return the next Quest layer's global index, or None if `layer` is
    the last Quest layer in the model. Reads the indices view stashed by
    bind_runtime; if missing, returns None (Mode 2 inert).
    """
    indices = getattr(layer, "_quest_layer_indices_view", None)
    if indices is None:
        return None
    cur = layer.layer_idx
    after = [i for i in indices if i > cur]
    return after[0] if after else None


def _prefetch_window(layer) -> int:
    """Read prefetch_window_blocks off the layer's quest_config (cached
    by bind_runtime). Returns 0 when not set or when Mode 2 is disabled."""
    qc = getattr(layer, "_quest_config_ref", None)
    if qc is None:
        return 0
    return int(getattr(qc, "prefetch_window_blocks", 0))


def _quest_layer_tier_manager(layer, target_layer_idx: int):
    """Resolve the TierManager for a target layer index. Reads from the
    forward-context registry stashed by bind_runtime."""
    registry = getattr(layer, "_quest_layer_tm_registry", None)
    if registry is None:
        return None
    return registry.get(target_layer_idx)


def run_sparse_decode(impl, layer, query, kv_cache, md, output) -> torch.Tensor:
    """Decode-step sparse path. Must equal dense FA when top_k >= num_blocks
    and no eviction has happened (proved by R1 spike)."""
    from flash_attn import flash_attn_with_kvcache
    from vllm.v1.attention.ops.quest_selection_torch import (
        quest_selection_torch,
    )

    tm: "TierManager" = layer.tier_manager
    seq_lens = md.seq_lens
    block_size = tm.gpu_k.shape[1]
    num_reqs = seq_lens.shape[0]
    top_k = int(getattr(md, "quest_top_k", 64))

    # Mode 2 preamble: if a previous layer scheduled a prefetch into this
    # layer's pool, wait on it before ensure_resident decides which extra
    # blocks to fetch. The wait is no-op when no event was registered
    # (Mode 1, or layer 0 of a fresh seq).
    pool = getattr(tm, "stream_pool", None)
    if pool is not None:
        for req_idx in range(num_reqs):
            prefetch_event = pool.pop_prefetch_event(
                seq_id=req_idx, target_layer_idx=layer.layer_idx,
            )
            if prefetch_event is not None:
                torch.cuda.current_stream().wait_event(prefetch_event)

    per_req_top_ids: list[torch.Tensor] = []
    out_chunks = []
    for req_idx in range(num_reqs):
        sl = int(seq_lens[req_idx].item())
        num_blocks = (sl + block_size - 1) // block_size
        cand = torch.arange(num_blocks, dtype=torch.int32,
                            device=query.device)
        # build [num_kv_heads * G, head_size] view for the last query token
        q_token = query[req_idx]              # [num_heads, head_size]
        # Score using the global summary row.
        summary_layer = tm.summary_store.summary[layer.layer_idx]
        top_ids = quest_selection_torch(
            query=q_token.reshape(layer.num_heads, layer.head_size),
            block_summary=summary_layer,
            candidate_ids=cand,
            num_kv_groups=layer.num_heads // layer.num_kv_heads,
            top_k=min(top_k, num_blocks),
        )
        per_req_top_ids.append(top_ids)
        # Wait on H2D completion before kernel reads the slots. Sync mode
        # returns None (no wait); async mode returns an Event we must
        # serialize the compute stream against.
        h2d_event = tm.ensure_resident(
            seq_id=req_idx, logical_block_ids=top_ids,
        )
        if h2d_event is not None:
            torch.cuda.current_stream().wait_event(h2d_event)
        # Translate logical block ids to physical GPU slots.
        slots = torch.tensor(
            [tm.logical_to_slot(seq_id=req_idx, logical_block_id=int(b))
             for b in top_ids.tolist()],
            dtype=torch.int32, device=query.device,
        ).unsqueeze(0)
        sub_seq_len = torch.tensor(
            [slots.numel() * block_size],
            dtype=torch.int32, device=query.device,
        )
        # kv_cache layout (num_blocks, 2, block_size, h_kv, head_size)
        k_view = kv_cache[:, 0]
        v_view = kv_cache[:, 1]
        out_req = flash_attn_with_kvcache(
            query[req_idx: req_idx + 1].unsqueeze(1),
            k_view, v_view,
            block_table=slots, cache_seqlens=sub_seq_len, causal=True,
        )
        out_chunks.append(out_req.squeeze(1))

    out = torch.cat(out_chunks, dim=0)
    output.copy_(out.reshape_as(output))

    # Mode 2 postamble: speculatively prefetch the same top_ids into the
    # next layer's pool. Window > 0 gates Mode 2; window == 0 keeps
    # Mode 1 (no speculation).
    if pool is not None:
        next_layer_idx = _next_quest_layer_idx(layer)
        if next_layer_idx is not None:
            window = _prefetch_window(layer)
            if window > 0:
                next_tm = _quest_layer_tier_manager(layer, next_layer_idx)
                if next_tm is not None:
                    for req_idx, top_ids in enumerate(per_req_top_ids):
                        # Limit how many ids we prefetch to bound the
                        # LRU-thrash exposure (see QuestConfig docstring).
                        ids = top_ids[: window]
                        next_tm.prefetch_top_ids(
                            seq_id=req_idx,
                            logical_block_ids=ids,
                        )

    return output

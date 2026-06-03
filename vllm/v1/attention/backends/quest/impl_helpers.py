# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-step orchestration for QuestSparseOffloadImpl."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from vllm.v1.attention.backends.quest.cache.tier_manager import (
        TierManager,
    )


# Stage 2C-v2 INVARIANT-1 tripwire: maps scratch tensor data_ptr ->
# [last_writer_layer_idx, offload_done_bool]. Set when a shared layer writes the
# scratch (kvshare_write_scratch), flipped True when that layer's per-layer
# offload completes (kvshare_prefill_offload). A non-serial executor would write
# the scratch again before the prior offload finished — caught loudly. Only
# touched on the footprint_kvshare prefill path; empty/unused otherwise.
_SCRATCH_OWNER: dict[int, list] = {}


@contextlib.contextmanager
def _nvtx_range(name: str, enabled: bool):
    """NVTX range, no-op unless enabled (clean pass / default path stay clean).
    torch.cuda.nvtx is always importable; range_push/pop are ~free but we still
    gate so the clean perf pass has zero added calls."""
    if not enabled:
        yield
        return
    import torch

    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def _seq_id_for(md, req_idx: int):
    """Stable per-request key for TierManager state (LRU, _trimmed, _cpu_slots).

    Uses ``md.request_ids[req_idx]`` (vLLM's input_batch.req_ids, populated by
    QuestMetadataBuilder.set_request_ids per forward) when available. Falls
    back to ``req_idx`` ONLY for unit-test metadata fixtures that bypass the
    builder — production paths must always carry request_ids. Without this,
    the TierManager arena keys on batch-position and a second generate() call
    aliases the first call's KV state.
    """
    rids = getattr(md, "request_ids", ())
    if rids and req_idx < len(rids):
        return rids[req_idx]
    return req_idx


def notify_filled_blocks_after_prefill(layer, key, value, md) -> None:
    """Called right after FA's prefill writes KV. Walks slot_mapping to
    detect block boundaries; for each boundary crossed, registers the just-
    completed block's SUMMARY with the layer's TierManager.

    Summary-only by design: prefill leaves all prompt blocks in the engine's
    paged cache (full residency); the bounded Quest arena is populated only by
    the one-shot trim_to_working_set at the prefill->decode boundary. Copying
    blocks into the arena here would overflow it for prompts longer than `cap`
    blocks (and pre-spill blocks the trim then double-evicts). See
    TierManager.register_prefill_summary.
    """
    tm: TierManager | None = getattr(layer, "tier_manager", None)
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
        seq_id = _seq_id_for(md, req_idx)
        for b in range(full_blocks):
            tok_lo = beg + b * block_size
            tok_hi = tok_lo + block_size
            block_id = b
            tm.register_prefill_summary(
                seq_id=seq_id,
                logical_block_id=block_id,
                k_block=key[tok_lo:tok_hi],
            )


def notify_filled_blocks_after_decode(layer, kv_cache, md) -> None:
    """Decode-step bookkeeping for the Quest arena (Stage 2A).

    Three things happen per request, in order:
      (a) one-shot trim on the FIRST decode step for the seq — copies the last
          cap-1 full blocks engine->arena and spills the rest to CPU,
          establishing the bounded resident set (idempotent per seq);
      (b) if this token just completed a full block (sl % block_size == 0),
          register that block's summary + arena slot via on_block_filled;
      (c) refresh the trailing PARTIAL block into the arena. The partial block
          holds the just-generated decode token at position sl-1; it is never
          scored/selected/evicted, but the query must attend it, so it is
          copied into a pinned arena slot keyed (seq_id, full_blocks) and
          re-touched (MRU) every step. On the next boundary it becomes a normal
          full block via (b). When sl % block_size == 0 there is no partial
          block this step, so (c) is skipped.

    The KV was already written into `kv_cache` by reshape_and_cache_flash, so
    full/partial blocks are read back from their engine physical slots.

    Per-forward GC: any seq the manager has state for that is NOT in the
    current batch's request_ids has finished (or been preempted); release its
    LRU/CPU/residency rows via TierManager.free_request. Without this, every
    request leaks its block_filled CPU slots and arena LRU keys, and the CPU
    pool exhausts (RuntimeError) after enough sequential requests.
    """
    tm: TierManager | None = getattr(layer, "tier_manager", None)
    if tm is None:
        return
    block_size = tm.gpu_k.shape[1]
    seq_lens = md.seq_lens.tolist()
    rids = getattr(md, "request_ids", ()) or ()
    active = {rids[i] if i < len(rids) else i for i in range(len(seq_lens))}
    # Compare against the manager's last-seen active set. Anything that
    # disappeared has finished — free it.
    prev_active = getattr(tm, "_active_seqs", None)
    if prev_active is not None:
        for old in prev_active - active:
            tm.free_request(old)
    tm._active_seqs = active
    # kv_cache layout: (num_blocks, 2, block_size, num_kv_heads, head_size).
    k_cache_view = kv_cache[:, 0]
    v_cache_view = kv_cache[:, 1]
    for req_idx, sl in enumerate(seq_lens):
        full_blocks = sl // block_size
        seq_id = _seq_id_for(md, req_idx)
        # (a) one-shot trim on first decode for this seq.
        tm.trim_to_working_set(
            seq_id=seq_id,
            num_full_blocks=full_blocks,
            kv_cache=kv_cache,
            block_table_row=md.block_table[req_idx],
        )
        # (b) a block just completed iff sl % block_size == 0.
        if sl != 0 and sl % block_size == 0:
            block_id = sl // block_size - 1
            phys = int(md.block_table[req_idx, block_id].item())
            tm.on_block_filled(
                seq_id=seq_id,
                logical_block_id=block_id,
                k_block=k_cache_view[phys],
                v_block=v_cache_view[phys],
            )
        # (c) refresh the live partial block into the arena (if any).
        if sl % block_size != 0:
            phys = int(md.block_table[req_idx, full_blocks].item())
            tm.write_live_block(
                seq_id=seq_id,
                live_block_id=full_blocks,
                k_block=k_cache_view[phys],
                v_block=v_cache_view[phys],
            )


def kvshare_write_scratch(impl, layer, key, value, kv_cache, md) -> None:
    """Stage 2C-v2: write a SHARED Quest layer's prefill K/V into the scratch
    tensor (== kv_cache) so FA's paged prefill can read it.

    Under footprint_kvshare the engine skips do_kv_cache_update for shared
    layers (kv_sharing_target guard), leaving the scratch unwritten. md's
    slot_mapping already targets the scratch slots (the shared layer is folded
    into the scratch group), so reusing it makes FA prefill read exactly the
    rows we write. Scatter is byte-identical to the engine's own write
    (reshape_and_cache_flash via FA's do_kv_cache_update).

    Scratch-capacity guard (INVARIANT / temporary-design): the slot_mapping
    indices must fall inside the scratch tensor; at concurrency=1 the scratch
    holds the whole single-sequence prefill, so this holds. Asserted loudly.

    INVARIANT-1 tripwire: all shared layers alias ONE scratch tensor and write
    it serially. We tag the scratch (by data_ptr) with the last writer and
    require its per-layer offload to have completed before the next layer
    overwrites. Serial synchronous forward (write -> FA read -> offload, all
    before the next layer's forward starts) makes this hold structurally; the
    assert catches a future parallel/pipelined executor loudly instead of
    silently corrupting. Cleared by kvshare_prefill_offload.
    """
    n = int(md.num_actual_tokens)
    slot_mapping = md.slot_mapping[:n]
    scratch_slots = kv_cache.shape[0] * kv_cache.shape[2]  # num_blocks * bs
    assert int(slot_mapping.max().item()) < scratch_slots, (
        "kvshare scratch overflow: prefill slot_mapping exceeds the scratch "
        "tensor capacity (concurrency>1 or scratch too small). This is a "
        "temporary-design constraint of footprint_kvshare, not a vLLM invariant."
    )
    ptr = kv_cache.data_ptr()
    prev = _SCRATCH_OWNER.get(ptr)
    assert prev is None or prev[1], (
        f"INVARIANT-1 violated: scratch {ptr:#x} written by layer "
        f"{prev[0] if prev else '?'} whose offload has NOT completed before "
        f"layer {layer.layer_idx} overwrites it. footprint_kvshare assumes "
        f"serial layer execution; a parallel/pipelined executor needs a "
        f"per-layer scratch buffer."
    )
    _SCRATCH_OWNER[ptr] = [layer.layer_idx, False]  # [writer, offload_done]
    impl.do_kv_cache_update(layer, key, value, kv_cache, slot_mapping)
    # Remember which scratch this layer wrote, so its offload can mark it done.
    layer._quest_kvshare_scratch_ptr = ptr


def kvshare_prefill_offload(layer, key, value, md) -> None:
    """Stage 2C-v2: per-layer prefill offload for a SHARED Quest layer, sourced
    from this layer's key/value (NOT the scratch, which the next layer
    overwrites). Establishes the bounded working set + summaries at prefill
    time via TierManager.prefill_ingest_kvshare (keep last cap-1 full blocks,
    spill the rest, stage the trailing partial as the live block).

    Concurrency=1 scope: one request per prefill. Walks query_start_loc to honor
    per-request token ranges defensively, but the design targets a single seq.

    Cross-request GC: ``prefill_ingest_kvshare`` mutates the LRU slot map,
    ``_cpu_slots``/``_host_slots``, and per-(layer, block) residency. The 2A
    path's GC (mirror in ``notify_filled_blocks_after_decode``) is too late
    here — request B's prefill runs BEFORE any decode step, so any state left
    over from request A would either crash ``begin_evict`` (residency still
    ``ON_CPU``) or trip the "must not overflow the arena" guard (LRU still
    full). Diff ``_active_seqs`` and free finished requests up front.
    """
    tm: TierManager | None = getattr(layer, "tier_manager", None)
    if tm is None:
        return
    block_size = tm.gpu_k.shape[1]
    seq_lens = md.seq_lens.tolist()
    qstart = md.query_start_loc.tolist()
    rids = getattr(md, "request_ids", ()) or ()
    active = {rids[i] if i < len(rids) else i for i in range(len(seq_lens))}
    prev_active = getattr(tm, "_active_seqs", None)
    if prev_active is not None:
        for old in prev_active - active:
            tm.free_request(old)
    tm._active_seqs = active
    for req_idx, _sl in enumerate(seq_lens):
        beg = qstart[req_idx]
        end = qstart[req_idx + 1]
        prompt_len = end - beg
        if prompt_len <= 0:
            continue
        num_full = prompt_len // block_size
        seq_id = _seq_id_for(md, req_idx)
        tm.prefill_ingest_kvshare(
            seq_id=seq_id,
            num_full_blocks=num_full,
            key=key[beg:end],
            value=value[beg:end],
            block_size=block_size,
            prompt_len=prompt_len,
        )
    # INVARIANT-1: this layer's offload is done; release the scratch so the next
    # serially-executed shared layer may overwrite it.
    ptr = getattr(layer, "_quest_kvshare_scratch_ptr", None)
    if ptr is not None and ptr in _SCRATCH_OWNER:
        _SCRATCH_OWNER[ptr][1] = True


def kvshare_decode_write(layer, key, value, md) -> None:
    """Stage 2C-v2: write each decode step's token KV into the per-layer arena
    live block for a SHARED Quest layer, sourced from key/value (the engine
    wrote nothing to the scratch under kv-share). Mirrors the GC + active-set
    bookkeeping of notify_filled_blocks_after_decode but without the
    scratch-sourced trim/on_block_filled/write_live_block reads.

    Each request contributes exactly one decode token this step. ``seq_lens``
    here is the length INCLUDING the just-generated token, so the token's
    logical position is ``sl - 1``.
    """
    tm: TierManager | None = getattr(layer, "tier_manager", None)
    if tm is None:
        return
    block_size = tm.gpu_k.shape[1]
    seq_lens = md.seq_lens.tolist()
    rids = getattr(md, "request_ids", ()) or ()
    active = {rids[i] if i < len(rids) else i for i in range(len(seq_lens))}
    prev_active = getattr(tm, "_active_seqs", None)
    if prev_active is not None:
        for old in prev_active - active:
            tm.free_request(old)
    tm._active_seqs = active
    for req_idx, sl in enumerate(seq_lens):
        seq_id = _seq_id_for(md, req_idx)
        # The decode token sits at logical position sl-1; seq_len_before = sl-1.
        tm.append_decode_token_kvshare(
            seq_id=seq_id,
            seq_len_before=sl - 1,
            k_tok=key[req_idx],
            v_tok=value[req_idx],
            block_size=block_size,
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


def _prefetch_window(layer, top_k: int) -> int:
    """Effective prefetch window: min(configured, top_k), with 0 meaning
    'unset' -> backfill to top_k. Returns 0 only under lru (no prefetch)."""
    qc = getattr(layer, "_quest_config_ref", None)
    if qc is None:
        return 0
    if getattr(qc, "block_ordering", "lru") == "lru":
        return 0
    configured = int(getattr(qc, "prefetch_window_blocks", 0)) or top_k
    return min(configured, top_k)


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

    # Increment stats up-front so the counter reflects "sparse path engaged"
    # even when num_blocks=0 makes the inner loop a no-op. selected_total is
    # accumulated below as we loop over requests.
    tm_stats: TierManager | None = getattr(layer, "tier_manager", None)
    if tm_stats is not None:
        tm_stats._stats.select_calls += 1

    selection_fn = getattr(layer, "_quest_selection_callable_ref", None)
    if selection_fn is None:
        # Unit-test fallback: tests that bypass bind_runtime don't stash
        # a ref; default to the torch oracle (the Phase B baseline).
        from vllm.v1.attention.ops.quest_selection_torch import (
            quest_selection_torch,
        )

        selection_fn = quest_selection_torch

    tm: TierManager = layer.tier_manager
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
                seq_id=_seq_id_for(md, req_idx),
                target_layer_idx=layer.layer_idx,
            )
            if prefetch_event is not None:
                torch.cuda.current_stream().wait_event(prefetch_event)

    per_req_top_ids: list[torch.Tensor] = []
    out_chunks = []
    for req_idx in range(num_reqs):
        seq_id = _seq_id_for(md, req_idx)
        sl = int(seq_lens[req_idx].item())
        # Quest scores only *fully filled* blocks: the trailing partial block
        # has no tier_manager entry (on_block_filled registers a block only on
        # a block boundary) and no summary, so it cannot be scored. But that
        # partial block holds the just-generated decode token (at position
        # sl-1), which the query MUST attend to — including itself. So we
        # score/select among the full blocks only, then UNCONDITIONALLY append
        # the partial block to the gather as an always-resident "recent
        # window": never scored, never selected, never evicted. This makes the
        # sparse decode attend the SAME token set as dense FA (all `sl`
        # positions), which is what the seq_too_short gate could not guarantee
        # for steps past the first.
        full_blocks = sl // block_size
        has_partial = (sl % block_size) != 0
        cand = torch.arange(full_blocks, dtype=torch.int32, device=query.device)
        # build [num_kv_heads * G, head_size] view for the last query token
        q_token = query[req_idx]  # [num_heads, head_size]
        # Score using the per-layer summary row. tm.layer_idx is the quest
        # slot (0..num_quest_layers-1), NOT the global layer_idx —
        # summary_store is sized num_quest_layers, so indexing by global
        # layer_idx (2..27) overflows.
        summary_layer = tm.summary_store.summary[tm.layer_idx]
        # Debug gate: NVTX ranges only when overlap/debug capture is on
        # (instrumented pass). Default path / clean pass add zero NVTX calls.
        nvtx_gate = getattr(tm, "enable_overlap_capture", False)
        with _nvtx_range(f"quest.select.L{tm.layer_idx}", nvtx_gate):
            top_ids = selection_fn(
                query=q_token.reshape(layer.num_heads, layer.head_size),
                block_summary=summary_layer,
                candidate_ids=cand,
                num_kv_groups=layer.num_heads // layer.num_kv_heads,
                top_k=min(top_k, full_blocks),
            )
        per_req_top_ids.append(top_ids)
        if tm_stats is not None and getattr(tm, "enable_overlap_capture", False):
            # step = this layer's select_calls-1 (0-based); seq_id = req_idx.
            tm.record_selected(
                step=tm_stats._stats.select_calls - 1,
                seq_id=seq_id,
                block_ids=top_ids.tolist(),
            )
        if tm_stats is not None:
            tm_stats._stats.selected_total += int(top_ids.numel())
            # GPU-residency hit-rate numerator: how many of the just-selected
            # blocks are ALREADY on GPU at selection time. Must be measured
            # here, BEFORE ensure_resident below makes everything resident
            # (after which the metric would always read 100%). Cheap set
            # membership over the LRU map; no GPU sync, no LRU mutation.
            tm_stats._stats.selected_on_gpu += tm.count_resident(
                seq_id=seq_id,
                logical_block_ids=top_ids.tolist(),
            )
        # Wait on H2D completion before kernel reads the slots. Sync mode
        # returns None (no wait); async mode returns an Event we must
        # serialize the compute stream against.
        #
        # keep_resident_ids protects the live partial block (key full_blocks)
        # from being self-evicted while ensure_resident reloads selected misses:
        # it is appended to the gather but is never in top_ids, so without this
        # the two-pass touch wouldn't know to shield it. When there is no
        # partial block (exact boundary) there is nothing extra to keep.
        keep_ids = [full_blocks] if has_partial else None
        with _nvtx_range(f"quest.ensure_resident.L{tm.layer_idx}", nvtx_gate):
            h2d_event = tm.ensure_resident(
                seq_id=seq_id,
                logical_block_ids=top_ids,
                keep_resident_ids=keep_ids,
            )
            if h2d_event is not None:
                torch.cuda.current_stream().wait_event(h2d_event)
        # Stage 2A: gather selected full blocks from the ARENA (tm.gpu_k/gpu_v)
        # via the Quest LRU slot map. ensure_resident (above) has reloaded any
        # CPU-resident selected block into the arena, so every top_id maps to a
        # live arena row here. (Stage 0 read md.block_table / the engine cache
        # because the "arena" then aliased the full engine tensor and its own
        # slots were bogus; Tasks 1-4 made the arena a real, curated, bounded
        # buffer, so we read it directly — the offload round-trip is real.)
        slot_list = [
            tm.logical_to_slot(seq_id=seq_id, logical_block_id=int(b))
            for b in top_ids.tolist()
        ]  # Number of FULL blocks actually gathered (top_k may be < full_blocks).
        num_full_gathered = len(slot_list)
        # Trailing PARTIAL block lifecycle (Stages A/B/C — see Task 5b):
        #  A (management): the partial block holds the live decode token at
        #    position sl-1. It has NO summary and is NOT a selection candidate
        #    (cand = arange(full_blocks) excludes it), so it is never scored,
        #    selected, or evicted.
        #  B (caching): write_live_block copied it from its engine slot into a
        #    PINNED arena slot keyed (req_idx, full_blocks), re-touched to MRU
        #    every step so the LRU never picks it. On the next boundary it
        #    becomes a normal full block via on_block_filled (same key, same
        #    arena slot — no double-occupancy).
        #  C (attention use): it is appended LAST, after the num_full_gathered
        #    selected blocks, so its tokens occupy gather positions
        #    [num_full_gathered*bs, num_full_gathered*bs + residual); cache_seqlens
        #    = sl_effective bounds the read to exactly those live tokens, so the
        #    unwritten tail (residual..bs-1) and anything past it are never read.
        #  Exact boundary (sl % bs == 0): has_partial is False, nothing is
        #    appended, residual = 0 — full-blocks-only gather.
        if has_partial:
            slot_list.append(
                tm.logical_to_slot(seq_id=seq_id, logical_block_id=full_blocks)
            )
        slots = torch.tensor(
            slot_list,
            dtype=torch.int32,
            device=query.device,
        ).unsqueeze(0)
        # TRUE attended length: the full blocks contribute whole blocks; if a
        # partial block is appended it sits immediately after them in the
        # gather, so its live tokens occupy positions
        # [num_full_gathered*block_size, num_full_gathered*block_size + residual).
        # cache_seqlens must point one past the decode token so the kernel
        # attends it (and, with causal=True, attends itself).
        residual = (sl % block_size) if has_partial else 0
        sl_effective = num_full_gathered * block_size + residual
        sub_seq_len = torch.tensor(
            [sl_effective],
            dtype=torch.int32,
            device=query.device,
        )
        # Stage 2A: the gathered `slots` are ARENA slot indices, so the kernel
        # reads the arena (tm.gpu_k/gpu_v), NOT the engine kv_cache.
        k_view = tm.gpu_k
        v_view = tm.gpu_v
        with _nvtx_range(f"quest.sparse_attn.L{tm.layer_idx}", nvtx_gate):
            out_req = flash_attn_with_kvcache(
                query[req_idx : req_idx + 1].unsqueeze(1),
                k_view,
                v_view,
                block_table=slots,
                cache_seqlens=sub_seq_len,
                causal=True,
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
            window = _prefetch_window(layer, top_k)
            if window > 0:
                next_tm = _quest_layer_tier_manager(layer, next_layer_idx)
                if next_tm is not None:
                    for req_idx, top_ids in enumerate(per_req_top_ids):
                        seq_id = _seq_id_for(md, req_idx)
                        ids = top_ids[:window]
                        # mixture: record this layer's selection so the next
                        # layer's eviction protects it. No-op under prefetch
                        # (set_prev_selected only stores under mixture).
                        next_tm.set_prev_selected(
                            seq_id=seq_id,
                            block_ids=ids.tolist(),
                        )
                        # Limit prefetch count to bound LRU-thrash exposure
                        # (see QuestConfig.prefetch_window_blocks docstring).
                        next_tm.prefetch_top_ids(
                            seq_id=seq_id,
                            logical_block_ids=ids,
                        )

    return output

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TierManager: per-layer GPU/CPU coordination for the Quest backend.

Owns:
  - GPU paged cache slice for this layer (gpu_k, gpu_v).
  - LRU policy over GPU slots (via vLLM's existing LRUCachePolicy).
  - Per-seq logical->slot mapping.
  - Residency state machine row.
  - CPU pool slot allocation row.

Phase B is fully synchronous; ensure_resident returns None so callers do
not need to await anything. Phase C will return an Event | None.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING

import torch

from vllm.v1.attention.backends.quest.cache.block_summary import (
    BlockSummaryStore,
)
from vllm.v1.attention.backends.quest.cache.cpu_backing_store import (
    CpuKvBackingStore,
)
from vllm.v1.attention.backends.quest.cache.residency import (
    BlockResidency,
)
from vllm.v1.attention.backends.quest.cache.stats import QuestStats

if TYPE_CHECKING:
    from vllm.v1.attention.backends.quest.async_transfer import QuestStreamPool


class _LRUSlotMap:
    """Small per-layer LRU over (seq_id, logical_block_id) -> gpu_slot.

    capacity = the number of GPU slots in this layer's pool. In Phase B
    that equals `gpu_cache_blocks_per_seq` (one fresh-allocated buffer per
    Quest layer). In Phase E it equals `kv_cache_config.num_blocks` for
    the layer's group (the vLLM block_manager-allocated pool). Either
    way, eviction is invariant: when full, popitem(last=False) removes
    the LRU key and reuses its slot.

    Wraps an OrderedDict so it stays trivial to reason about. We deliberately
    do NOT pull in vLLM's LRUCachePolicy here in Phase B/E — that policy is
    keyed by hashes and carries ref-count machinery we don't need yet. Phase
    F can swap to LRUCachePolicy if we surface ref-count semantics.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._key_to_slot: OrderedDict[tuple[int, int], int] = OrderedDict()
        self._free_slots = list(reversed(range(capacity)))

    def __contains__(self, key) -> bool:
        return key in self._key_to_slot

    def get(self, key) -> int:
        slot = self._key_to_slot[key]
        self._key_to_slot.move_to_end(key)
        return slot

    def add(self, key) -> tuple[int, tuple[int, int] | None]:
        """Add a new key, returning (slot, evicted_key_or_None)."""
        if key in self._key_to_slot:
            return self.get(key), None
        evicted = None
        if self._free_slots:
            slot = self._free_slots.pop()
        else:
            evicted_key, slot = self._key_to_slot.popitem(last=False)
            evicted = evicted_key
        self._key_to_slot[key] = slot
        return slot, evicted

    def free(self, key) -> int:
        slot = self._key_to_slot.pop(key)
        self._free_slots.append(slot)
        return slot


class TierManager:
    def __init__(
        self,
        *,
        layer_idx: int,
        gpu_budget: int,
        gpu_k: torch.Tensor,
        gpu_v: torch.Tensor,
        summary_store: BlockSummaryStore,
        residency: BlockResidency,
        cpu_store: CpuKvBackingStore,
        stream_pool: QuestStreamPool | None = None,
        enable_event_timing: bool = False,
        enable_overlap_capture: bool = False,
        gpu_pool_aliases_kv_cache: bool = False,
        engine_kv_cache: torch.Tensor | None = None,
    ) -> None:
        self.layer_idx = layer_idx
        self.gpu_budget = gpu_budget
        self.gpu_k = gpu_k
        self.gpu_v = gpu_v
        self.summary_store = summary_store
        self.residency = residency
        self.cpu_store = cpu_store
        self.stream_pool = stream_pool
        # Stage 2A: reference to the full engine-allocated kv_cache for this
        # layer (None for unit-test/private-buffer paths). Kept ONLY as the
        # SOURCE for the prefill->decode trim (trim_to_working_set, later task).
        self.engine_kv_cache = engine_kv_cache
        # Stage 2A removed the Stage-0 aliasing mode entirely: gpu_k/gpu_v are
        # ALWAYS a private bounded arena now, never a zero-copy view of the
        # engine kv_cache. The parameter is retained (defaulting False) only so
        # existing call sites/tests keep constructing cleanly; True is no longer
        # a supported mode. Assert it to catch any accidental re-introduction —
        # if this fires, a caller is trying to resurrect the offload-bypass
        # crutch.
        assert not gpu_pool_aliases_kv_cache, (
            "gpu_pool_aliases_kv_cache=True is unsupported after Stage 2A; "
            "the Quest arena is always a private bounded buffer."
        )
        self.gpu_pool_aliases_kv_cache = gpu_pool_aliases_kv_cache
        # Benchmark/debug-only: when True, ensure_resident / _spill_to_cpu
        # bracket their copies with cuda Events and accumulate GPU time into
        # _stats. Zero-cost when False (no Event creation, no sync). Gated by
        # QuestConfig.enable_debug_counters at construction time.
        self.enable_event_timing = enable_event_timing
        # Snapshot of seq_ids that owned slots/spilled blocks at the end of the
        # previous decode step. notify_filled_blocks_after_decode diffs this
        # against the current active set so requests that left the batch
        # release their (seq_id, *) entries via free_request. Populated on
        # demand; mypy needs the declaration to allow attribute access in
        # impl_helpers.
        self._active_seqs: set[int] | None = None
        # Benchmark/debug-only (Stage 1 cross-layer overlap): when True,
        # run_sparse_decode records each step's selected block-id set into
        # _selected_log via record_selected. Drained out-of-band by the
        # apply_model probe after the run. No-op when False (record_selected
        # returns immediately). Gated by QuestConfig.enable_debug_counters at
        # construction time, same gate as enable_event_timing.
        self.enable_overlap_capture = enable_overlap_capture

        self._slot_map = _LRUSlotMap(capacity=gpu_budget)
        # Per-evicted (seq_id, logical_block_id) -> cpu_slot
        self._cpu_slots: dict[tuple[int, int], int] = {}
        self._stats = QuestStats()
        # Debug-only per-(step, seq) selected block-id log (overlap capture).
        self._selected_log: list[dict] = []
        # seq_ids that have already had their one-shot prefill->decode trim
        # (trim_to_working_set). Trim is idempotent per sequence.
        self._trimmed: set[int] = set()
        # Pluggable D2H spill seam. Default = the synchronous _spill_to_cpu
        # defined below. Stage 2B write-through overrides this to mirror blocks
        # to host on a d2h_stream at fill time instead of at eviction time.
        # All eviction call sites (on_block_filled, _ensure_one_sync,
        # _ensure_one_async) route their spill through this attribute.
        self.spill_hook = self._spill_to_cpu

    def stats(self) -> QuestStats:
        return self._stats

    def record_selected(self, step: int, seq_id: int, block_ids) -> None:
        """Debug-only (Stage 1 cross-layer overlap). Append this step's selected
        block ids for this (layer-slot, step, seq). No-op unless capture is on.
        Cheap: a list of small int lists; drained via apply_model after the run.
        """
        if not self.enable_overlap_capture:
            return
        self._selected_log.append(
            {
                "step": int(step),
                "seq_id": int(seq_id),
                "block_ids": [int(b) for b in block_ids],
            }
        )

    def drain_selected(self) -> list[dict]:
        out = self._selected_log
        self._selected_log = []
        return out

    def logical_to_slot(self, seq_id: int, logical_block_id: int) -> int:
        return self._slot_map.get((seq_id, logical_block_id))

    def is_resident(self, seq_id: int, logical_block_id: int) -> bool:
        """True iff (seq_id, logical_block_id) currently holds a GPU slot.

        Read-only: does NOT touch LRU recency (unlike logical_to_slot, which
        calls _slot_map.get and bumps the key to most-recently-used). Used by
        the selection path to measure GPU-residency hit-rate *before*
        ensure_resident makes everything resident.
        """
        return (seq_id, logical_block_id) in self._slot_map

    def count_resident(self, seq_id: int, logical_block_ids) -> int:
        """How many of `logical_block_ids` are already GPU-resident.

        `logical_block_ids` is any iterable of ints (e.g. ``top_ids.tolist()``).
        Cheap set membership over the LRU map; no GPU sync, no LRU mutation.
        """
        return sum(
            1 for bid in logical_block_ids if (seq_id, int(bid)) in self._slot_map
        )

    def register_prefill_summary(
        self,
        seq_id: int,
        logical_block_id: int,
        k_block: torch.Tensor,
    ) -> None:
        """Prefill-time registration: record the per-block K summary ONLY.

        During prefill the engine holds every prompt block in its own paged
        cache; the bounded Quest arena is NOT populated until the one-shot
        trim_to_working_set at the prefill->decode boundary. So prefill must
        record the summary (the only thing selection needs, and the only thing
        that must survive eviction) WITHOUT touching the arena LRU slot map or
        the residency state machine — otherwise a prompt with more than `cap`
        blocks would overflow the arena during prefill and spill blocks to CPU,
        leaving the trim's begin_evict to find them already ON_CPU. The arena
        residency is established exclusively by trim_to_working_set.
        """
        self.summary_store.on_block_filled(
            self.layer_idx,
            logical_block_id,
            k_block,
        )

    def on_block_filled(
        self,
        seq_id: int,
        logical_block_id: int,
        k_block: torch.Tensor,
        v_block: torch.Tensor,
    ) -> int:
        """Called when a block fills up during prefill or chunked prefill.
        Returns the GPU slot index assigned.
        """
        # Update summary first — this is the only thing that survives eviction.
        self.summary_store.on_block_filled(
            self.layer_idx,
            logical_block_id,
            k_block,
        )

        key = (seq_id, logical_block_id)
        slot, evicted = self._slot_map.add(key)
        if evicted is not None:
            # Spill the evicted block's data BEFORE we overwrite the slot.
            self.spill_hook(*evicted, slot=slot)

        # Copy the filled block into its private arena slot. (Stage 2A: the
        # arena is always a real private buffer now — the Stage-0 aliasing mode,
        # where this copy was skipped because gpu_k aliased the engine cache, is
        # gone; see the assert in __init__.)
        self.gpu_k[slot].copy_(k_block, non_blocking=False)
        self.gpu_v[slot].copy_(v_block, non_blocking=False)
        self.residency.mark_on_gpu(self.layer_idx, logical_block_id)
        self._stats.block_filled += 1
        return slot

    def trim_to_working_set(
        self,
        seq_id: int,
        num_full_blocks: int,
        kv_cache: torch.Tensor,
        block_table_row,
    ) -> None:
        """One-shot prefill->decode trim. Keep the last cap-1 full blocks in
        the arena (1 slot reserved for the live decode block), spill the rest
        to CPU. Idempotent per seq.

        kv_cache is the engine tensor (FA layout (nb, 2, bs, h, d));
        block_table_row[b] is the engine physical slot for logical block b of
        this sequence. Kept blocks are copied engine->arena; spilled blocks are
        copied engine->CPU and recorded in _cpu_slots so ensure_resident can
        reload them (H2D) when later selected.
        """
        if seq_id in self._trimmed:
            return
        self._trimmed.add(seq_id)
        keep_n = max(0, min(num_full_blocks, self._slot_map.capacity - 1))
        keep_lo = num_full_blocks - keep_n  # keep [keep_lo, num_full_blocks)
        k_eng, v_eng = kv_cache[:, 0], kv_cache[:, 1]
        for b in range(num_full_blocks):
            phys = int(block_table_row[b])
            if b >= keep_lo:
                slot, evicted = self._slot_map.add((seq_id, b))
                assert evicted is None, "trim must not overflow the arena"
                self.gpu_k[slot].copy_(k_eng[phys])
                self.gpu_v[slot].copy_(v_eng[phys])
                self.residency.mark_on_gpu(self.layer_idx, b)
            else:
                try:
                    cpu_slot = self.cpu_store.alloc(self.layer_idx)
                except RuntimeError as e:
                    raise RuntimeError(
                        f"Quest trim: CPU pool exhausted spilling "
                        f"{num_full_blocks - keep_n} blocks for seq {seq_id} "
                        f"layer {self.layer_idx}; raise cpu_cache_blocks or use "
                        f"a shorter prompt (Stage 2B adds max_model_len-based "
                        f"sizing)"
                    ) from e
                self.cpu_store.store_block(
                    self.layer_idx,
                    cpu_slot,
                    k_eng[phys],
                    v_eng[phys],
                )
                self._cpu_slots[(seq_id, b)] = cpu_slot
                self.residency.begin_evict(self.layer_idx, b)
                self.residency.complete_evict(self.layer_idx, b)
                self._stats.evict_d2h += 1

    def write_live_block(
        self,
        seq_id,
        live_block_id: int,
        k_block: torch.Tensor,
        v_block: torch.Tensor,
    ) -> int:
        """Copy the live partial block into a (re-touched) arena slot. The live
        block holds the just-generated decode token; it is never scored, never
        selected, and must never be evicted while it is the tail — so re-adding
        it every decode step moves it to MRU and the LRU never picks it.
        Returns the arena slot.
        """
        slot, evicted = self._slot_map.add((seq_id, live_block_id))
        if evicted is not None:
            self.spill_hook(*evicted, slot=slot)
        self.gpu_k[slot].copy_(k_block, non_blocking=False)
        self.gpu_v[slot].copy_(v_block, non_blocking=False)
        self.residency.mark_on_gpu(self.layer_idx, live_block_id)
        return slot

    def free_request(self, seq_id) -> None:
        """Release every TierManager state row keyed by ``seq_id``.

        Called when a vLLM request finishes (or its slot is reused for a new
        request). Without this, every Quest layer's ``_slot_map``,
        ``_cpu_slots``, and ``_trimmed`` grow unbounded across requests, and
        a fresh ``seq_id`` cannot reach a clean baseline because residency
        rows still carry stale states from prior seqs.

        Idempotent: a seq_id that was never seen on this layer is a no-op.

        - LRU slot map: every ``(seq_id, *)`` entry is freed back to the slot
          pool. We do NOT D2H-spill these — the request is done, the contents
          are dead.
        - CPU pool: every CPU slot recorded for ``(seq_id, *)`` is returned
          to the per-layer free list (otherwise the pool ceiling is hit
          quickly on long workloads).
        - ``_trimmed``: discard, so a future request reusing this seq_id
          (cannot happen with vLLM's req_ids in practice, but guard anyway)
          re-runs trim_to_working_set.
        - residency: reset every block this layer ever marked for this seq
          back to ON_GPU (the zero state) so the next seq's first transition
          starts from a clean baseline.

        Note: residency is keyed only by ``(layer_idx, logical_block_id)`` —
        not per-seq — so two concurrent seqs sharing a logical_block_id
        index would alias. For Stage 2A's serial / small-batch workloads
        this aliasing is harmless because at most one seq's blocks are
        active at any given physical block_id slot. Multi-seq concurrency
        on overlapping logical_block_ids is a Stage 3 concern (see
        stage2a-delivery-status.md §4); free_request only cleans up after a
        seq is gone, so the aliasing window doesn't matter here.
        """
        # 1. LRU + GPU slot pool. iterating a dict while mutating: copy keys.
        keys = [k for k in self._slot_map._key_to_slot if k[0] == seq_id]
        evicted_blocks: list[int] = []
        for k in keys:
            self._slot_map.free(k)
            evicted_blocks.append(k[1])
        # 2. CPU slots. Each (seq_id, *) -> cpu_slot must be returned to
        # cpu_store free list, otherwise CPU pool exhausts after enough
        # requests and trim raises RuntimeError.
        cpu_keys = [k for k in self._cpu_slots if k[0] == seq_id]
        for k in cpu_keys:
            cpu_slot = self._cpu_slots.pop(k)
            self.cpu_store.free(self.layer_idx, cpu_slot)
            evicted_blocks.append(k[1])
        # 3. trimmed marker.
        self._trimmed.discard(seq_id)
        # 4. Residency rows. Use mark_on_gpu (the legal write to this layer's
        # state machine) rather than poking _states directly. Idempotent.
        for b in evicted_blocks:
            self.residency.mark_on_gpu(self.layer_idx, b)

    def ensure_resident(
        self,
        seq_id: int,
        logical_block_ids: torch.Tensor,
        keep_resident_ids: list[int] | None = None,
    ) -> torch.cuda.Event | None:
        """Make every selected block GPU-resident, returning an H2D Event to
        wait on (async) or None (sync/aliasing).

        Two-pass, to avoid self-eviction within a single decode step. The arena
        holds `cap` blocks; a step needs `len(top_ids)` selected blocks plus the
        blocks in `keep_resident_ids` (the live partial block) simultaneously
        resident. With the invariant top_k <= cap-1, that set always fits — but
        a naive single reload loop could, when the arena is full, evict an
        already-resident block that is ALSO selected this step (or the live
        block) before it is read.

        Pass 1 touches every already-resident selected block AND every
        keep_resident_id to MRU, lifting them into the protected (most-recent)
        end of the LRU. Pass 2 then reloads the CPU-resident misses; because the
        protected set sits at the MRU end and (selected ∪ keep) <= cap, the LRU
        victim for each reload is guaranteed to be a NON-protected block. So no
        block needed this step is evicted before it is gathered.
        """
        ids = logical_block_ids.cpu().tolist()
        keep = keep_resident_ids or []

        # Pass 1: protect the blocks this step needs that are already resident
        # (selected-and-resident + the live/keep blocks) by touching them to
        # MRU. Pure LRU bookkeeping, no data movement.
        self._touch_resident(seq_id, ids, keep)

        # Pass 2: reload the CPU-resident misses. Their evictions now fall on
        # non-protected blocks only.
        if self.stream_pool is None:
            if self.enable_event_timing:
                self._timed_ensure_sync(seq_id, ids)
            else:
                for bid in ids:
                    self._ensure_one_sync(seq_id, bid)
            return None

        with torch.cuda.stream(self.stream_pool.h2d_stream):
            for bid in ids:
                self._ensure_one_async(seq_id, bid)
        return self.stream_pool.record_h2d_done()

    def _touch_resident(
        self,
        seq_id: int,
        ids: list[int],
        keep_ids: list[int],
    ) -> None:
        """Pass 1 of ensure_resident: bump every already-resident block in
        (ids ∪ keep_ids) to MRU so the Pass-2 reload loop never evicts a block
        this step still needs. Skips blocks not on GPU (those are reloaded in
        Pass 2)."""
        for bid in keep_ids:
            if (seq_id, bid) in self._slot_map:
                self._slot_map.get((seq_id, bid))
        for bid in ids:
            if (seq_id, bid) in self._slot_map:
                self._slot_map.get((seq_id, bid))

    def _timed_ensure_sync(self, seq_id: int, ids: list[int]) -> None:
        """Benchmark-only: run the sync ensure loop bracketed by cuda Events
        and accumulate H2D-wait GPU time. Only counts the interval when at
        least one block actually loaded from CPU (resident blocks are free).
        """
        loads_before = self._stats.load_h2d
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for bid in ids:
            self._ensure_one_sync(seq_id, bid)
        end.record()
        end.synchronize()
        if self._stats.load_h2d > loads_before:
            self._stats.h2d_wait_ms += start.elapsed_time(end)
            self._stats.h2d_wait_events += 1

    def _ensure_one_sync(self, seq_id: int, bid: int) -> None:
        key = (seq_id, bid)
        if key in self._slot_map:
            self._slot_map.get(key)
            return
        cpu_slot = self._cpu_slots.pop(key, None)
        if cpu_slot is None:
            raise RuntimeError(f"block {key} is neither on GPU nor in CPU pool")
        slot, evicted = self._slot_map.add(key)
        if evicted is not None:
            self.spill_hook(*evicted, slot=slot)
        self.residency.begin_load(self.layer_idx, bid)
        self.cpu_store.load_block(
            self.layer_idx,
            cpu_slot,
            self.gpu_k[slot],
            self.gpu_v[slot],
        )
        self.cpu_store.free(self.layer_idx, cpu_slot)
        self.residency.complete_load(self.layer_idx, bid)
        self._stats.load_h2d += 1

    def _ensure_one_async(self, seq_id: int, bid: int) -> None:
        """Same as _ensure_one_sync but uses non_blocking copies. Caller
        is inside a `with torch.cuda.stream(h2d_stream):` block."""
        key = (seq_id, bid)
        if key in self._slot_map:
            self._slot_map.get(key)
            return
        cpu_slot = self._cpu_slots.pop(key, None)
        if cpu_slot is None:
            raise RuntimeError(f"block {key} is neither on GPU nor in CPU pool")
        slot, evicted = self._slot_map.add(key)
        if evicted is not None:
            self.spill_hook(*evicted, slot=slot)
        # Residency state machine update fires synchronously, BEFORE the
        # async H2D actually completes. This is intentional: the state
        # tracks INTENT, not completion. The contract is:
        #   - Caller MUST wait on the Event returned by ensure_resident
        #     before reading gpu_k[slot] / gpu_v[slot].
        #   - Callers that need a completion-aware view of residency
        #     (e.g. is_on_gpu_mask before quest_selection candidate
        #     filtering) must guard their reads with the same
        #     wait_event(...) in the async path.
        # Phase D may move this to a deferred-completion model; Phase B/C
        # do not call is_on_gpu_mask between ensure_resident return and
        # the caller's wait_event, so the hazard is dormant.
        self.residency.begin_load(self.layer_idx, bid)
        self.cpu_store.load_block(
            self.layer_idx,
            cpu_slot,
            self.gpu_k[slot],
            self.gpu_v[slot],
            non_blocking=True,
        )
        self.cpu_store.free(self.layer_idx, cpu_slot)
        self.residency.complete_load(self.layer_idx, bid)
        self._stats.load_h2d += 1

    def _spill_to_cpu(
        self,
        seq_id: int,
        logical_block_id: int,
        *,
        slot: int,
    ) -> None:
        """Snapshot gpu_k[slot]/gpu_v[slot] into the CPU pool BEFORE the
        slot is overwritten by the new key.

        Sync (Phase B): blocking D2H, slot is safe to overwrite on return.
        Async (Phase C): non_blocking D2H on d2h_stream. record_stream on
        the source tensor keeps PyTorch's caching allocator from recycling
        the underlying memory until d2h_stream finishes the copy.
        """
        cpu_slot = self.cpu_store.alloc(self.layer_idx)
        self.residency.begin_evict(self.layer_idx, logical_block_id)
        if self.stream_pool is None:
            if self.enable_event_timing:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                self.cpu_store.store_block(
                    self.layer_idx,
                    cpu_slot,
                    self.gpu_k[slot],
                    self.gpu_v[slot],
                )
                end.record()
                end.synchronize()
                self._stats.evict_stall_ms += start.elapsed_time(end)
                self._stats.evict_stall_events += 1
            else:
                self.cpu_store.store_block(
                    self.layer_idx,
                    cpu_slot,
                    self.gpu_k[slot],
                    self.gpu_v[slot],
                )
        else:
            d2h = self.stream_pool.d2h_stream
            # Tell the allocator: don't recycle these GPU tensors until
            # d2h_stream passes this point.
            self.gpu_k[slot].record_stream(d2h)
            self.gpu_v[slot].record_stream(d2h)
            with torch.cuda.stream(d2h):
                self.cpu_store.store_block(
                    self.layer_idx,
                    cpu_slot,
                    self.gpu_k[slot],
                    self.gpu_v[slot],
                    non_blocking=True,
                )
        self.residency.complete_evict(self.layer_idx, logical_block_id)
        self._cpu_slots[(seq_id, logical_block_id)] = cpu_slot
        self._stats.evict_d2h += 1

    def prefetch_top_ids(
        self,
        seq_id: int,
        logical_block_ids: torch.Tensor,
    ) -> None:
        """Mode 2: speculatively H2D the given block ids into this layer's
        pool. Registers an event in the pool keyed by (seq_id, layer_idx).

        No-op when stream_pool is None (sync mode).

        WARNING: this method evicts LRU blocks if the pool is full and the
        speculation is wrong. See QuestConfig.prefetch_window_blocks
        docstring for the LRU-thrash analysis.
        """
        if self.stream_pool is None:
            return
        ids = logical_block_ids.cpu().tolist()
        with torch.cuda.stream(self.stream_pool.h2d_stream):
            for bid in ids:
                self._ensure_one_async(seq_id, bid)
        event = self.stream_pool.record_h2d_done()
        self.stream_pool.register_prefetch_event(
            seq_id=seq_id,
            target_layer_idx=self.layer_idx,
            event=event,
        )

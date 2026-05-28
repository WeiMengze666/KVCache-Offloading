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
        self._key_to_slot: "OrderedDict[tuple[int, int], int]" = OrderedDict()
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
    ) -> None:
        self.layer_idx = layer_idx
        self.gpu_budget = gpu_budget
        self.gpu_k = gpu_k
        self.gpu_v = gpu_v
        self.summary_store = summary_store
        self.residency = residency
        self.cpu_store = cpu_store
        self.stream_pool = stream_pool

        self._slot_map = _LRUSlotMap(capacity=gpu_budget)
        # Per-evicted (seq_id, logical_block_id) -> cpu_slot
        self._cpu_slots: dict[tuple[int, int], int] = {}
        self._stats = QuestStats()

    def stats(self) -> QuestStats:
        return self._stats

    def logical_to_slot(self, seq_id: int, logical_block_id: int) -> int:
        return self._slot_map.get((seq_id, logical_block_id))

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
            self.layer_idx, logical_block_id, k_block,
        )

        key = (seq_id, logical_block_id)
        slot, evicted = self._slot_map.add(key)
        if evicted is not None:
            # Spill the evicted block's data BEFORE we overwrite the slot.
            self._spill_to_cpu(*evicted, slot=slot)

        self.gpu_k[slot].copy_(k_block, non_blocking=False)
        self.gpu_v[slot].copy_(v_block, non_blocking=False)
        self.residency.mark_on_gpu(self.layer_idx, logical_block_id)
        self._stats.block_filled += 1
        return slot

    def ensure_resident(
        self,
        seq_id: int,
        logical_block_ids: torch.Tensor,
    ) -> torch.cuda.Event | None:
        """Sync (Phase B): copies happen inline, returns None.
        Async (Phase C): copies issued on h2d_stream with non_blocking=True;
        returns an Event the caller waits on before reading the slots.
        """
        ids = logical_block_ids.cpu().tolist()
        if self.stream_pool is None:
            for bid in ids:
                self._ensure_one_sync(seq_id, bid)
            return None

        with torch.cuda.stream(self.stream_pool.h2d_stream):
            for bid in ids:
                self._ensure_one_async(seq_id, bid)
        return self.stream_pool.record_h2d_done()

    def _ensure_one_sync(self, seq_id: int, bid: int) -> None:
        key = (seq_id, bid)
        if key in self._slot_map:
            self._slot_map.get(key)
            return
        cpu_slot = self._cpu_slots.pop(key, None)
        if cpu_slot is None:
            raise RuntimeError(
                f"block {key} is neither on GPU nor in CPU pool"
            )
        slot, evicted = self._slot_map.add(key)
        if evicted is not None:
            self._spill_to_cpu(*evicted, slot=slot)
        self.residency.begin_load(self.layer_idx, bid)
        self.cpu_store.load_block(
            self.layer_idx, cpu_slot,
            self.gpu_k[slot], self.gpu_v[slot],
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
            raise RuntimeError(
                f"block {key} is neither on GPU nor in CPU pool"
            )
        slot, evicted = self._slot_map.add(key)
        if evicted is not None:
            self._spill_to_cpu(*evicted, slot=slot)
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
            self.layer_idx, cpu_slot,
            self.gpu_k[slot], self.gpu_v[slot],
            non_blocking=True,
        )
        self.cpu_store.free(self.layer_idx, cpu_slot)
        self.residency.complete_load(self.layer_idx, bid)
        self._stats.load_h2d += 1

    def _spill_to_cpu(
        self, seq_id: int, logical_block_id: int, *, slot: int,
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
            self.cpu_store.store_block(
                self.layer_idx, cpu_slot,
                self.gpu_k[slot], self.gpu_v[slot],
            )
        else:
            d2h = self.stream_pool.d2h_stream
            # Tell the allocator: don't recycle these GPU tensors until
            # d2h_stream passes this point.
            self.gpu_k[slot].record_stream(d2h)
            self.gpu_v[slot].record_stream(d2h)
            with torch.cuda.stream(d2h):
                self.cpu_store.store_block(
                    self.layer_idx, cpu_slot,
                    self.gpu_k[slot], self.gpu_v[slot],
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

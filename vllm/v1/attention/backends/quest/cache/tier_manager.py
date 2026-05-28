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
    ) -> None:
        self.layer_idx = layer_idx
        self.gpu_budget = gpu_budget
        self.gpu_k = gpu_k
        self.gpu_v = gpu_v
        self.summary_store = summary_store
        self.residency = residency
        self.cpu_store = cpu_store

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
    ) -> None:
        """For each id in logical_block_ids that is currently ON_CPU, copy
        it back into a GPU slot (possibly evicting another block). Synchronous
        in Phase B.
        """
        ids = logical_block_ids.cpu().tolist()
        for bid in ids:
            key = (seq_id, bid)
            if key in self._slot_map:
                # Already resident — touch LRU and continue.
                self._slot_map.get(key)
                continue
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

    def _spill_to_cpu(
        self, seq_id: int, logical_block_id: int, *, slot: int,
    ) -> None:
        """Snapshot gpu_k[slot]/gpu_v[slot] into the CPU pool BEFORE the
        slot is overwritten by the new key. `slot` is the index returned
        by `_LRUSlotMap.add` for the new key — its old contents still
        belong to the evicted (seq_id, logical_block_id).
        """
        cpu_slot = self.cpu_store.alloc(self.layer_idx)
        self.residency.begin_evict(self.layer_idx, logical_block_id)
        self.cpu_store.store_block(
            self.layer_idx, cpu_slot,
            self.gpu_k[slot], self.gpu_v[slot],
        )
        self.residency.complete_evict(self.layer_idx, logical_block_id)
        self._cpu_slots[(seq_id, logical_block_id)] = cpu_slot
        self._stats.evict_d2h += 1

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""collective_rpc probe functions, executed inside the EngineCore worker.

Returned dicts must be msgspec-serializable: only ints / floats / strs / Nones
in flat keys. No tensors, no dataclasses.
"""

from __future__ import annotations

import os
from typing import Any


def _torch_metrics() -> dict[str, int]:
    import torch

    stats = torch.accelerator.memory_stats()
    return {
        "torch.allocated_bytes": int(torch.accelerator.memory_allocated()),
        "torch.reserved_bytes": int(torch.accelerator.memory_reserved()),
        "torch.peak_allocated_bytes": int(stats.get("allocated_bytes.all.peak", 0)),
        "torch.active_bytes": int(stats.get("active_bytes.all.current", 0)),
    }


def _nvml_metrics() -> dict[str, int]:
    try:
        import pynvml

        pynvml.nvmlInit()
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        idx = int(visible.split(",")[0]) if visible else 0
        h = pynvml.nvmlDeviceGetHandleByIndex(idx)
        info = pynvml.nvmlDeviceGetMemoryInfo(h)
        return {
            "nvml.gpu_used_bytes": int(info.used),
            "nvml.gpu_total_bytes": int(info.total),
        }
    except Exception:
        return {"nvml.gpu_used_bytes": 0, "nvml.gpu_total_bytes": 0}


def _collect_tier_managers(worker) -> list[Any]:
    """Return [TierManager, ...] for all quest layers, or [] if not Quest.

    Tries `model_runner.quest_tier_managers_for_probe()` first; if that helper
    is not present, scans the runner's `compilation_config.static_forward_context`
    (vLLM v1's authoritative store of attention layers, see
    `vllm/v1/worker/gpu_model_runner.py:946`) and falls back to the legacy
    `attn_layers` / `attention_layers` / `_attn_layers` attributes for older
    vLLM revisions. Tolerates dense engines (returns []).
    """
    runner = getattr(worker, "model_runner", None)
    if runner is None:
        return []
    helper = getattr(runner, "quest_tier_managers_for_probe", None)
    if helper is not None:
        return list(helper())

    out: list[Any] = []

    def _tm_of(layer):
        # Quest stores tier_manager directly on the Attention layer module
        # (see vllm/v1/attention/backends/quest/impl.py:95). Older drafts
        # tucked it under layer.impl; check both.
        tm = getattr(layer, "tier_manager", None)
        if tm is None:
            impl = getattr(layer, "impl", None)
            tm = getattr(impl, "tier_manager", None)
        return tm

    # vLLM v1 path: attention layers live in compilation_config.static_forward_context
    cc = getattr(runner, "compilation_config", None)
    sfc = getattr(cc, "static_forward_context", None) if cc is not None else None
    if sfc:
        for layer in sfc.values():
            tm = _tm_of(layer)
            if tm is not None:
                out.append(tm)
        if out:
            return out

    # Legacy fallbacks for older vLLM revisions.
    for attr in ("attn_layers", "attention_layers", "_attn_layers"):
        layers = getattr(runner, attr, None)
        if layers:
            for layer in layers:
                tm = _tm_of(layer)
                if tm is not None:
                    out.append(tm)
            if out:
                return out
    return []


def _slot_map_size(tm) -> int:
    sm = tm._slot_map
    # _LRUSlotMap doesn't expose len(); read its internal OrderedDict.
    inner = getattr(sm, "_key_to_slot", None)
    if inner is not None:
        return len(inner)
    # Fall back to size() if a future version adds it.
    if hasattr(sm, "size"):
        return int(sm.size())
    return 0


def _aggregate_quest(tms: list[Any]) -> dict[str, Any]:
    if not tms:
        return {
            "quest.gpu_resident_blocks": None,
            "quest.cpu_resident_blocks": None,
            "quest.evict_d2h": None,
            "quest.load_h2d": None,
            "quest.select_calls": None,
            "quest.selected_total": None,
            "quest.selected_on_gpu": None,
            "quest.topk_hit_ratio": None,
        }
    gpu_blocks = sum(_slot_map_size(tm) for tm in tms)
    cpu_blocks = sum(len(tm._cpu_slots) for tm in tms)
    evict_d2h = sum(tm._stats.evict_d2h for tm in tms)
    load_h2d = sum(tm._stats.load_h2d for tm in tms)
    select_calls = sum(tm._stats.select_calls for tm in tms)
    selected_total = sum(tm._stats.selected_total for tm in tms)
    selected_on_gpu = sum(tm._stats.selected_on_gpu for tm in tms)
    hit_ratio = selected_on_gpu / selected_total if selected_total > 0 else None
    return {
        "quest.gpu_resident_blocks": gpu_blocks,
        "quest.cpu_resident_blocks": cpu_blocks,
        "quest.evict_d2h": evict_d2h,
        "quest.load_h2d": load_h2d,
        "quest.select_calls": select_calls,
        "quest.selected_total": selected_total,
        "quest.selected_on_gpu": selected_on_gpu,
        "quest.topk_hit_ratio": hit_ratio,
    }


def _vllm_kv_pool_bytes(worker) -> int | None:
    v = getattr(worker, "available_kv_cache_memory_bytes", None)
    return int(v) if v else None


def _arena_total_bytes(tms: list[Any]) -> int:
    """Total bytes of all Quest TierManager `gpu_k` + `gpu_v` arenas.

    The Quest arena is allocated via `torch.empty` in
    vllm/v1/attention/backends/quest/backend.py (Stage 2A+ private buffer,
    no aliasing). It lives outside `available_kv_cache_memory_bytes` and
    must be subtracted from `torch.allocated_bytes` to get a clean
    weights+workspace `essential` figure. K and V are equally sized, so
    we use `gpu_k.numel() * element_size() * 2`.
    """
    total = 0
    for tm in tms:
        k = tm.gpu_k
        total += int(k.numel()) * int(k.element_size()) * 2
    return total


def _dense_kv_useful_bytes(worker, bytes_per_block: int | None) -> int | None:
    if bytes_per_block is None:
        return None
    try:
        scheduler = worker.scheduler
        mgr = scheduler.kv_cache_manager
        used = mgr.num_used_blocks
        return int(used) * int(bytes_per_block)
    except Exception:
        return None


def probe_snapshot(worker, bytes_per_block: int | None) -> dict[str, Any]:
    """Single sample point. Flat msgspec-friendly dict.

    bytes_per_block: per-layer per-block (K+V) byte count. None disables KV
    useful/slack derivation (Dense without scheduler hook).
    """
    out: dict[str, Any] = {}
    out.update(_torch_metrics())
    out.update(_nvml_metrics())
    tms = _collect_tier_managers(worker)
    out.update(_aggregate_quest(tms))

    if tms and bytes_per_block is not None:
        out["quest.gpu_resident_bytes"] = (
            out["quest.gpu_resident_blocks"] * bytes_per_block
        )
        out["quest.cpu_resident_bytes"] = (
            out["quest.cpu_resident_blocks"] * bytes_per_block
        )
    else:
        out["quest.gpu_resident_bytes"] = None
        out["quest.cpu_resident_bytes"] = None

    weights = int(
        getattr(
            getattr(worker, "model_runner", None),
            "model_memory_usage",
            0,
        )
        or 0
    )
    kv_pool_total = _vllm_kv_pool_bytes(worker)
    arena_total = _arena_total_bytes(tms) if tms else None
    if tms:
        if bytes_per_block is None:
            kv_useful = None
        else:
            kv_useful = out["quest.gpu_resident_blocks"] * bytes_per_block
    else:
        kv_useful = _dense_kv_useful_bytes(worker, bytes_per_block)

    if kv_pool_total is not None and kv_useful is not None:
        slack = max(0, kv_pool_total - kv_useful)
        slack_ratio = slack / kv_pool_total if kv_pool_total > 0 else None
    else:
        slack = None
        slack_ratio = None

    # essential = torch.allocated - kv_pool_total - arena_total
    # In Quest mode the arena is a private torch.empty allocation NOT covered
    # by available_kv_cache_memory_bytes; subtract it explicitly so essential
    # reflects only weights + workspace.
    allocated = out["torch.allocated_bytes"]
    peak_allocated = out["torch.peak_allocated_bytes"]
    pool_for_calc = kv_pool_total or 0
    arena_for_calc = arena_total or 0
    essential = allocated - pool_for_calc - arena_for_calc
    essential_peak = peak_allocated - pool_for_calc - arena_for_calc

    # actual_used = process-resident KV bytes + essential.
    # Quest: arena_total is the entire torch.empty buffer the process holds.
    # Dense: kv_useful (engine pool's used portion) when scheduler bookkeeping
    # is reachable; otherwise we fall back to torch.allocated_bytes — that's
    # the authoritative figure for "bytes vLLM is currently holding" because
    # dense has no private arenas outside the allocator.
    if tms:
        actual_used_kv = arena_total
        actual_used = (
            max(0, essential) + actual_used_kv if actual_used_kv is not None else None
        )
        actual_used_peak = (
            max(0, essential_peak) + actual_used_kv
            if actual_used_kv is not None
            else None
        )
    else:
        if kv_useful is not None:
            actual_used = max(0, essential) + kv_useful
            actual_used_peak = max(0, essential_peak) + kv_useful
        else:
            actual_used = allocated
            actual_used_peak = peak_allocated

    out["vllm.engine_essential_bytes"] = max(0, essential)
    out["vllm.engine_essential_peak_bytes"] = max(0, essential_peak)
    out["vllm.kv_pool_total_bytes"] = kv_pool_total
    out["vllm.gpu_kv_useful_bytes"] = kv_useful
    out["vllm.kv_pool_slack_bytes"] = slack
    out["vllm.kv_pool_slack_ratio"] = slack_ratio
    out["vllm.weights_bytes"] = weights
    out["quest.arena_total_bytes"] = arena_total
    out["vllm.actual_used_bytes"] = actual_used
    out["vllm.actual_used_peak_bytes"] = actual_used_peak
    return out


def reset_peak_stats(worker) -> None:
    import torch

    torch.accelerator.reset_peak_memory_stats()


def probe_bytes_per_block(worker) -> int | None:
    """Per-block bytes from the first quest layer's gpu_k tensor:
    block_size * h_kv * d * dtype.itemsize * 2 (K+V).
    """
    tms = _collect_tier_managers(worker)
    if tms:
        tm = tms[0]
        k = tm.gpu_k
        per_block_k = 1
        for s in k.shape[1:]:
            per_block_k *= int(s)
        return per_block_k * int(k.element_size()) * 2
    try:
        scheduler = worker.scheduler
        cc = scheduler.cache_config
        return int(cc.block_size_bytes) if hasattr(cc, "block_size_bytes") else None
    except Exception:
        return None

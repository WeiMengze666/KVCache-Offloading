# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Backend-private counters."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class QuestStats:
    block_filled: int = 0
    evict_d2h: int = 0
    evict_drop: int = 0    # write-through evictions: GPU slot dropped, no D2H
    load_h2d: int = 0
    select_calls: int = 0
    selected_total: int = 0
    selected_on_gpu: int = 0    # how many selected blocks were already resident

    # --- benchmark/debug-only GPU-event timing (gated by
    # enable_debug_counters; zero-cost when the gate is off) ---
    # Cumulative GPU time (milliseconds) spent in synchronous H2D loads
    # (ensure_resident) and D2H spills (_spill_to_cpu), measured with
    # torch.cuda.Event. These are populated ONLY when the TierManager is
    # built with enable_event_timing=True. Counters of how many timed
    # intervals were recorded let the harness compute a mean per event.
    h2d_wait_ms: float = 0.0
    evict_stall_ms: float = 0.0
    h2d_wait_events: int = 0
    evict_stall_events: int = 0

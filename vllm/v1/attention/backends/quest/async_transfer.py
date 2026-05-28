# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase C async transfer infrastructure for the Quest backend.

`QuestStreamPool` owns a single h2d/d2h stream pair shared across all
Quest layers in the engine, plus a Mode 2 prefetch event registry.

Construction is gated on `quest_config.enable_async_prefetch`; when
disabled the backend doesn't import this module and TierManager runs
the Phase B sync path.
"""
from __future__ import annotations

import torch


class QuestStreamPool:
    """Per-engine streams + Mode 2 prefetch event registry.

    One instance constructed by `init_runtime_state` (when async is
    enabled), passed by reference into every `TierManager`. All Quest
    layers share the same h2d / d2h stream — model layers execute
    serially, so multi-stream parallelism would not buy throughput.

    Mode 2 stashes per-(seq, target_layer) prefetch events so layer
    N+1's forward can wait on the event recorded by layer N's
    schedule_prefetch.
    """

    def __init__(self) -> None:
        self.h2d_stream: torch.cuda.Stream = torch.cuda.Stream()
        self.d2h_stream: torch.cuda.Stream = torch.cuda.Stream()
        # Keyed by (seq_id, target_layer_idx). Drained by pop on use.
        self._pending: dict[tuple[int, int], torch.cuda.Event] = {}

    def record_h2d_done(self) -> torch.cuda.Event:
        """Record an event on the h2d_stream and return it. Caller is
        responsible for waiting on it from compute_stream before reads."""
        event = torch.cuda.Event()
        event.record(self.h2d_stream)
        return event

    def record_d2h_done(self) -> torch.cuda.Event:
        event = torch.cuda.Event()
        event.record(self.d2h_stream)
        return event

    def register_prefetch_event(
        self, *, seq_id: int, target_layer_idx: int,
        event: torch.cuda.Event,
    ) -> None:
        """Mode 2: record a layer-N+1 prefetch event for layer N+1 to wait
        on. If a previous prefetch for the same key is still pending
        (uncommon; would indicate a forward retry), it is overwritten."""
        self._pending[(seq_id, target_layer_idx)] = event

    def pop_prefetch_event(
        self, *, seq_id: int, target_layer_idx: int,
    ) -> torch.cuda.Event | None:
        """Mode 2: retrieve and remove the prefetch event for a layer.
        Returns None when no pending prefetch exists (e.g., layer 0 of a
        fresh seq, or async disabled)."""
        return self._pending.pop((seq_id, target_layer_idx), None)

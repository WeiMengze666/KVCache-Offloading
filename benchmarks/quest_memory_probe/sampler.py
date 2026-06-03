# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Periodic sampling thread.

Calls a user-provided snapshot_fn at fixed interval, writes each result to
a queue with phase='sampling'. Errors are caught and emitted as
phase='probe_error' rows so the run never aborts on a transient probe failure.
"""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable
from typing import Any


def _now_ms() -> int:
    return time.monotonic_ns() // 1_000_000


class Sampler(threading.Thread):
    def __init__(
        self,
        *,
        snapshot_fn: Callable[[], dict[str, Any]],
        interval_s: float,
        queue_: queue.Queue,
    ) -> None:
        super().__init__(name="QuestMemProbeSampler", daemon=True)
        self.snapshot_fn = snapshot_fn
        self.interval_s = float(interval_s)
        self.queue = queue_
        self._stop_event = threading.Event()

    def run(self) -> None:
        while not self._stop_event.is_set():
            t0 = time.monotonic()
            try:
                snap = self.snapshot_fn() or {}
                snap["ts_ms"] = _now_ms()
                snap["phase"] = "sampling"
                self.queue.put(snap)
            except Exception as e:
                self.queue.put(
                    {
                        "ts_ms": _now_ms(),
                        "phase": "probe_error",
                        "error": repr(e),
                    }
                )
            elapsed = time.monotonic() - t0
            self._stop_event.wait(max(0.0, self.interval_s - elapsed))

    def stop(self) -> None:
        self._stop_event.set()

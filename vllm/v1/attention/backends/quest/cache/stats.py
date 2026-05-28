# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Backend-private counters."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class QuestStats:
    block_filled: int = 0
    evict_d2h: int = 0
    load_h2d: int = 0
    select_calls: int = 0
    selected_total: int = 0
    selected_on_gpu: int = 0    # how many selected blocks were already resident

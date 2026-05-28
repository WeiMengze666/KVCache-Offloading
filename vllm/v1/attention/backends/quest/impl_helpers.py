# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Helpers for QuestSparseOffloadImpl.forward.

Phase B Task 13 ships these as stubs so the impl skeleton imports
cleanly. Task 14 fills in the bodies (cache notify, sparse decode).
"""
from __future__ import annotations

from typing import Any


def notify_filled_blocks_after_prefill(
    layer: Any, key: Any, value: Any, attn_metadata: Any,
) -> None:
    """No-op until Task 14 wires the tier_manager."""
    return None


def notify_filled_blocks_after_decode(
    layer: Any, key: Any, value: Any, attn_metadata: Any,
) -> None:
    """No-op until Task 14 wires the tier_manager."""
    return None


def run_sparse_decode(
    impl: Any, layer: Any, query: Any, kv_cache: Any,
    attn_metadata: Any, output: Any,
) -> Any:
    """Sparse decode is not implemented in Task 13. Task 14 fills this in."""
    raise NotImplementedError(
        "QuestSparseOffloadImpl decode-sparse path is wired in Task 14."
    )

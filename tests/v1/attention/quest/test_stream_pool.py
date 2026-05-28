# SPDX-License-Identifier: Apache-2.0
"""QuestStreamPool: per-engine stream pair + Mode 2 event registry."""
from __future__ import annotations

import pytest
import torch


@pytest.fixture
def cuda():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")


def test_stream_pool_creates_one_pair(cuda):
    from vllm.v1.attention.backends.quest.async_transfer import (
        QuestStreamPool,
    )
    pool = QuestStreamPool()
    assert isinstance(pool.h2d_stream, torch.cuda.Stream)
    assert isinstance(pool.d2h_stream, torch.cuda.Stream)
    assert pool.h2d_stream is not pool.d2h_stream


def test_stream_pool_record_h2d_done_returns_event(cuda):
    from vllm.v1.attention.backends.quest.async_transfer import (
        QuestStreamPool,
    )
    pool = QuestStreamPool()
    event = pool.record_h2d_done()
    assert isinstance(event, torch.cuda.Event)


def test_stream_pool_pending_prefetch_registry(cuda):
    from vllm.v1.attention.backends.quest.async_transfer import (
        QuestStreamPool,
    )
    pool = QuestStreamPool()
    event = pool.record_h2d_done()
    pool.register_prefetch_event(seq_id=0, target_layer_idx=3, event=event)
    got = pool.pop_prefetch_event(seq_id=0, target_layer_idx=3)
    assert got is event
    # Second pop returns None (event consumed).
    assert pool.pop_prefetch_event(seq_id=0, target_layer_idx=3) is None


def test_stream_pool_pop_missing_returns_none(cuda):
    from vllm.v1.attention.backends.quest.async_transfer import (
        QuestStreamPool,
    )
    pool = QuestStreamPool()
    assert pool.pop_prefetch_event(seq_id=99, target_layer_idx=99) is None

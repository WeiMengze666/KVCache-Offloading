# SPDX-License-Identifier: Apache-2.0
"""Stage 3: block_ordering / prefetch_touch config + validate rules."""
from __future__ import annotations

import pytest

from vllm.config.quest import QuestConfig


def test_defaults_are_lru_and_no_touch():
    c = QuestConfig(enabled=True)
    assert c.block_ordering == "lru"
    assert c.prefetch_touch is False


def test_illegal_block_ordering_raises():
    c = QuestConfig(enabled=True, block_ordering="bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="block_ordering"):
        c.validate()


def test_prefetch_requires_async():
    c = QuestConfig(
        enabled=True, block_ordering="prefetch",
        enable_async_prefetch=False,
    )
    with pytest.raises(ValueError, match="enable_async_prefetch"):
        c.validate()


def test_mixture_requires_async():
    c = QuestConfig(
        enabled=True, block_ordering="mixture",
        enable_async_prefetch=False,
    )
    with pytest.raises(ValueError, match="enable_async_prefetch"):
        c.validate()


def test_touch_under_lru_raises():
    c = QuestConfig(enabled=True, block_ordering="lru", prefetch_touch=True)
    with pytest.raises(ValueError, match="prefetch_touch"):
        c.validate()


def test_touch_under_prefetch_raises():
    c = QuestConfig(
        enabled=True, block_ordering="prefetch", prefetch_touch=True,
        prefetch_window_blocks=4, enable_async_prefetch=True,
        gpu_cache_blocks_per_seq=8, top_k=4,
    )
    with pytest.raises(ValueError, match="prefetch_touch"):
        c.validate()


def test_mixture_touch_ok():
    c = QuestConfig(
        enabled=True, block_ordering="mixture", prefetch_touch=True,
        prefetch_window_blocks=4, enable_async_prefetch=True,
        gpu_cache_blocks_per_seq=8, top_k=4,
    )
    c.validate()  # no raise


def test_window_ge_arena_raises():
    c = QuestConfig(
        enabled=True, block_ordering="prefetch",
        prefetch_window_blocks=8, enable_async_prefetch=True,
        gpu_cache_blocks_per_seq=8, top_k=4,
    )
    with pytest.raises(ValueError, match="gpu_cache_blocks_per_seq"):
        c.validate()


def test_roundtrip_includes_new_fields():
    c = QuestConfig(enabled=True, block_ordering="mixture", prefetch_touch=True)
    d = c.to_dict()
    assert d["block_ordering"] == "mixture"
    assert d["prefetch_touch"] is True
    restored = QuestConfig.from_dict(d)
    assert restored.block_ordering == "mixture"
    assert restored.prefetch_touch is True

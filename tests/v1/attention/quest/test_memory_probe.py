# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for quest memory probe tooling (no GPU required)."""

from __future__ import annotations

import pytest

from benchmarks.quest_memory_probe.configs import RunConfig


class TestRunConfig:
    def test_pool_size_must_be_multiple_of_top_k(self):
        cfg = RunConfig(
            name="bad",
            quest_enabled=True,
            top_k=16,
            gpu_cache_blocks_per_seq=100,  # 100 % 16 != 0
        )
        with pytest.raises(ValueError, match="must be a multiple of top_k"):
            cfg.validate()

    def test_valid_quest_config_passes(self):
        cfg = RunConfig(
            name="ok",
            quest_enabled=True,
            top_k=16,
            gpu_cache_blocks_per_seq=128,
        )
        cfg.validate()  # no raise

    def test_block_size_must_be_256(self):
        cfg = RunConfig(name="bad", block_size=128)
        with pytest.raises(ValueError, match="block_size must be 256"):
            cfg.validate()

    def test_dense_skips_top_k_check(self):
        cfg = RunConfig(name="dense", quest_enabled=False, top_k=0)
        cfg.validate()

    def test_top_k_positive_when_quest_enabled(self):
        cfg = RunConfig(
            name="bad", quest_enabled=True, top_k=0, gpu_cache_blocks_per_seq=128
        )
        with pytest.raises(ValueError, match="top_k must be > 0"):
            cfg.validate()

    def test_to_dict_roundtrip(self):
        cfg = RunConfig(
            name="x", quest_enabled=True, top_k=16, gpu_cache_blocks_per_seq=128
        )
        d = cfg.to_dict()
        cfg2 = RunConfig.from_dict(d)
        assert cfg == cfg2

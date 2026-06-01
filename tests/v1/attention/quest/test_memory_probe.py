# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for quest memory probe tooling (no GPU required)."""

from __future__ import annotations

import pytest

from benchmarks.quest_memory_probe.configs import (
    RunConfig,
    expand_dense_vs_quest,
    expand_oom_sweep,
    expand_pool_size,
)


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


class TestConfigExpansion:
    def test_dense_vs_quest_yields_two_configs(self):
        cfgs = expand_dense_vs_quest(
            workload_spec="longbench:narrativeqa:lengths=short:n=2",
            top_k=16,
            quest_pool=128,
        )
        assert len(cfgs) == 2
        assert cfgs[0].quest_enabled is False
        assert cfgs[1].quest_enabled is True
        assert cfgs[1].top_k == 16
        assert cfgs[1].gpu_cache_blocks_per_seq == 128
        # All cfgs must validate
        for c in cfgs:
            c.validate()

    def test_pool_size_yields_one_per_pool(self):
        pools = [512, 256, 128, 32, 16]
        cfgs = expand_pool_size(
            workload_spec="longbench:narrativeqa:lengths=short:n=2",
            top_k=16,
            pool_sizes=pools,
        )
        assert len(cfgs) == len(pools)
        assert [c.gpu_cache_blocks_per_seq for c in cfgs] == pools
        assert all(c.quest_enabled for c in cfgs)
        for c in cfgs:
            c.validate()

    def test_pool_size_rejects_non_multiple(self):
        with pytest.raises(ValueError, match="multiple of top_k"):
            expand_pool_size(
                workload_spec="x",
                top_k=16,
                pool_sizes=[100],  # 100 % 16 != 0
            )

    def test_oom_sweep_yields_dense_and_quest(self):
        cfgs = expand_oom_sweep(
            workload_spec="longbench:narrativeqa:lengths=short,medium,long:n=4",
            top_k=16,
            quest_pool=128,
        )
        names = [c.name for c in cfgs]
        assert any("dense" in n for n in names)
        assert any("quest" in n for n in names)

    def test_config_names_are_unique(self):
        cfgs = expand_pool_size(
            workload_spec="x",
            top_k=16,
            pool_sizes=[512, 256, 128],
        )
        assert len({c.name for c in cfgs}) == 3

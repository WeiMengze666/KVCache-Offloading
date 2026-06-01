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
from benchmarks.quest_memory_probe.workload import (
    Sample,
    bucket_for_tokens,
    load_samples_synthetic,  # fallback path; pure Python, no GPU/HF needed
    parse_spec,
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


class TestWorkloadParsing:
    def test_parse_spec_full(self):
        spec = parse_spec("longbench:narrativeqa:lengths=short,medium:n=2")
        assert spec.source == "longbench"
        assert spec.task == "narrativeqa"
        assert spec.buckets == ("short", "medium")
        assert spec.n == 2

    def test_parse_spec_single_bucket(self):
        spec = parse_spec("longbench:lcc:lengths=long:n=1")
        assert spec.buckets == ("long",)
        assert spec.n == 1

    def test_parse_spec_rejects_unknown_source(self):
        with pytest.raises(ValueError, match="unknown workload source"):
            parse_spec("unknownDS:foo:lengths=short:n=1")

    def test_parse_spec_rejects_bad_bucket(self):
        with pytest.raises(ValueError, match="unknown length bucket"):
            parse_spec("longbench:narrativeqa:lengths=tiny:n=1")


class TestBucketing:
    def test_short_bucket(self):
        assert bucket_for_tokens(2000) == "short"

    def test_medium_bucket(self):
        assert bucket_for_tokens(8000) == "medium"

    def test_long_bucket(self):
        assert bucket_for_tokens(20000) == "long"

    def test_xlong_bucket(self):
        assert bucket_for_tokens(60000) == "xlong"

    def test_boundary_4k_is_medium(self):
        assert bucket_for_tokens(4096) == "medium"


class TestSyntheticFallback:
    def test_synthetic_returns_n_samples_per_bucket(self):
        samples = load_samples_synthetic(
            buckets=("short", "medium"),
            n=2,
            seed=42,
        )
        assert len(samples) == 4
        for s in samples:
            assert isinstance(s, Sample)
            assert s.bucket in ("short", "medium")
            assert s.prompt_tokens > 0
            assert s.prompt
        # Each (bucket, idx) pair appears exactly once
        ids = [s.sample_id for s in samples]
        assert len(set(ids)) == 4

    def test_synthetic_short_is_short(self):
        samples = load_samples_synthetic(buckets=("short",), n=1, seed=42)
        assert samples[0].prompt_tokens < 4096

    def test_synthetic_long_is_long(self):
        samples = load_samples_synthetic(buckets=("long",), n=1, seed=42)
        assert samples[0].prompt_tokens >= 16384


class TestLongBenchLoader:
    def test_load_samples_dispatch_synthetic_when_disabled(self, monkeypatch):
        # Force the loader to skip LongBench and use the synthetic fallback.
        from benchmarks.quest_memory_probe import workload

        monkeypatch.setenv("QUEST_MEM_PROBE_FORCE_SYNTHETIC", "1")
        samples = workload.load_samples("longbench:narrativeqa:lengths=short:n=2")
        assert len(samples) == 2
        assert all(s.sample_id.startswith("synthetic/") for s in samples)

    def test_load_samples_falls_back_on_load_dataset_error(self, monkeypatch):
        from benchmarks.quest_memory_probe import workload

        def boom(*args, **kwargs):
            raise RuntimeError("simulated dataset load failure")

        monkeypatch.setattr(workload, "_load_dataset", boom)
        samples = workload.load_samples("longbench:narrativeqa:lengths=short:n=1")
        assert len(samples) == 1
        assert samples[0].sample_id.startswith("synthetic/")


class FakeStats:
    block_filled = 1
    evict_d2h = 2
    load_h2d = 3
    select_calls = 4
    selected_total = 64
    selected_on_gpu = 48
    h2d_wait_ms = 1.5
    evict_stall_ms = 0.5
    h2d_wait_events = 1
    evict_stall_events = 1


class FakeTM:
    def __init__(self, layer_idx, gpu_resident, cpu_resident):
        self.layer_idx = layer_idx
        self._gpu_resident = gpu_resident
        self._cpu_resident = cpu_resident
        self._stats = FakeStats()
        _n = gpu_resident
        self._slot_map = type("M", (), {"size": lambda self, _n=_n: _n})()
        self._cpu_slots = {i: i for i in range(cpu_resident)}

    def stats(self):
        return self._stats


class FakeRunner:
    def __init__(self, tier_managers):
        self._tier_managers = tier_managers
        self.model_memory_usage = 6 * 1024**3

    def quest_tier_managers_for_probe(self):
        return self._tier_managers


class FakeWorker:
    def __init__(self, tier_managers):
        self.model_runner = FakeRunner(tier_managers)
        self.available_kv_cache_memory_bytes = 10 * 1024**3


class TestProbeSnapshot:
    def test_quest_aggregation_sums_layers(self, monkeypatch):
        from benchmarks.quest_memory_probe import probes

        monkeypatch.setattr(
            probes,
            "_torch_metrics",
            lambda: {
                "torch.allocated_bytes": 100,
                "torch.reserved_bytes": 200,
                "torch.peak_allocated_bytes": 150,
                "torch.active_bytes": 90,
            },
        )
        monkeypatch.setattr(
            probes,
            "_nvml_metrics",
            lambda: {
                "nvml.gpu_used_bytes": 300,
                "nvml.gpu_total_bytes": 1000,
            },
        )
        worker = FakeWorker([FakeTM(0, 5, 3), FakeTM(1, 7, 1)])
        snap = probes.probe_snapshot(worker, bytes_per_block=4096)

        assert snap["quest.gpu_resident_blocks"] == 12
        assert snap["quest.cpu_resident_blocks"] == 4
        assert snap["quest.gpu_resident_bytes"] == 12 * 4096
        assert snap["quest.cpu_resident_bytes"] == 4 * 4096
        assert snap["quest.selected_total"] == 128
        assert snap["quest.selected_on_gpu"] == 96
        assert snap["quest.topk_hit_ratio"] == pytest.approx(96 / 128)
        assert snap["vllm.kv_pool_total_bytes"] == 10 * 1024**3
        assert snap["vllm.gpu_kv_useful_bytes"] == 12 * 4096
        assert snap["vllm.kv_pool_slack_bytes"] == 10 * 1024**3 - 12 * 4096

    def test_dense_path_quest_fields_null(self, monkeypatch):
        from benchmarks.quest_memory_probe import probes

        monkeypatch.setattr(
            probes,
            "_torch_metrics",
            lambda: {
                "torch.allocated_bytes": 100,
                "torch.reserved_bytes": 200,
                "torch.peak_allocated_bytes": 150,
                "torch.active_bytes": 90,
            },
        )
        monkeypatch.setattr(
            probes,
            "_nvml_metrics",
            lambda: {
                "nvml.gpu_used_bytes": 300,
                "nvml.gpu_total_bytes": 1000,
            },
        )

        class DenseRunner:
            model_memory_usage = 6 * 1024**3

        class DenseWorker:
            model_runner = DenseRunner()
            available_kv_cache_memory_bytes = 10 * 1024**3

        snap = probes.probe_snapshot(DenseWorker(), bytes_per_block=None)

        assert snap["quest.gpu_resident_blocks"] is None
        assert snap["quest.topk_hit_ratio"] is None
        assert snap["vllm.gpu_kv_useful_bytes"] is None


class TestSampler:
    def test_sampler_collects_at_least_3_samples(self):
        import queue
        import time

        from benchmarks.quest_memory_probe.sampler import Sampler

        counter = {"n": 0}

        def snap():
            counter["n"] += 1
            return {"foo": counter["n"]}

        q: queue.Queue = queue.Queue()
        s = Sampler(snapshot_fn=snap, interval_s=0.05, queue_=q)
        s.start()
        time.sleep(0.25)
        s.stop()
        s.join(timeout=1.0)

        items = []
        while not q.empty():
            items.append(q.get_nowait())
        # at 50ms interval over 250ms, expect >= 3 samples; allow noise up to 12
        assert 3 <= len(items) <= 12
        assert all(it["phase"] == "sampling" for it in items)
        ts = [it["ts_ms"] for it in items]
        assert ts == sorted(ts)

    def test_sampler_records_probe_errors(self):
        import queue
        import time

        from benchmarks.quest_memory_probe.sampler import Sampler

        def boom():
            raise RuntimeError("simulated probe failure")

        q: queue.Queue = queue.Queue()
        s = Sampler(snapshot_fn=boom, interval_s=0.05, queue_=q)
        s.start()
        time.sleep(0.15)
        s.stop()
        s.join(timeout=1.0)

        items = []
        while not q.empty():
            items.append(q.get_nowait())
        assert items
        assert all(it["phase"] == "probe_error" for it in items)
        assert all("simulated probe failure" in it["error"] for it in items)

    def test_sampler_stop_is_idempotent(self):
        import queue

        from benchmarks.quest_memory_probe.sampler import Sampler

        q: queue.Queue = queue.Queue()
        s = Sampler(snapshot_fn=lambda: {}, interval_s=0.05, queue_=q)
        s.start()
        s.stop()
        s.stop()  # second stop must not raise
        s.join(timeout=1.0)
        assert not s.is_alive()

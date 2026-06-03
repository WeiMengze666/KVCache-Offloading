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
        with pytest.warns(UserWarning, match="not a multiple of top_k"):
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
        with pytest.warns(UserWarning, match="not a multiple of top_k"):
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

    def test_longbench_fails_fast_on_unknown_task(self, monkeypatch):
        from benchmarks.quest_memory_probe import workload

        # v2 schema has 'Single-Document QA', etc. — 'narrativeqa' is a v1
        # name and never matches. Old behavior tokenized the whole 503-item
        # corpus before giving up; new behavior should raise immediately.
        items = [
            {
                "domain": "Single-Document QA",
                "sub_domain": "Academic",
                "length": "short",
            },
            {
                "domain": "Multi-Document QA",
                "sub_domain": "Financial",
                "length": "long",
            },
        ]
        monkeypatch.setattr(workload, "_load_dataset", lambda *a, **k: items)
        monkeypatch.setattr(workload, "_read_template", lambda _: "$DOC$")

        with pytest.raises(RuntimeError, match="no items with domain/sub_domain"):
            workload._load_samples_longbench(
                workload.parse_spec("longbench:narrativeqa:lengths=short:n=1"),
                model="ignored",
            )

    def test_longbench_full_ignores_n_cap(self, monkeypatch):
        from benchmarks.quest_memory_probe import workload

        # Fake LongBench dataset: 5 narrativeqa items, all in the 'short'
        # bucket once tokenized. With n=1 we'd normally only keep 1; with
        # longbench_full=True we should get all 5.
        items = [
            {
                "domain": "narrativeqa",
                "length": "short",
                "context": f"ctx-{i}",
                "question": "q",
                "choice_A": "a",
                "choice_B": "b",
                "choice_C": "c",
                "choice_D": "d",
            }
            for i in range(5)
        ]
        monkeypatch.setattr(workload, "_load_dataset", lambda *a, **k: items)
        monkeypatch.setattr(workload, "_read_template", lambda _: "$DOC$")
        # tokenize is no longer used for bucketing, but we still call it for
        # Sample.prompt_tokens — return a constant.
        monkeypatch.setattr(workload, "_tokenize_count", lambda prompt, model: 100)

        capped = workload.load_samples("longbench:narrativeqa:lengths=short:n=1")
        assert len(capped) == 1

        full = workload.load_samples(
            "longbench:narrativeqa:lengths=short:n=1",
            longbench_full=True,
        )
        assert len(full) == 5
        assert all(s.bucket == "short" for s in full)

    def test_longbench_picks_shortest_per_bucket(self, monkeypatch):
        """When subsampling, pick the SHORTEST items per bucket.

        v2's long bucket spans 167k–4M tokens; without sorting we'd hit Code-Repo
        items with millions of tokens. Char count is the proxy used at selection
        time (avoids tokenizing every candidate).
        """
        from benchmarks.quest_memory_probe import workload

        # Three short items with descending context length. Pick first 2 by
        # char count (shortest first).
        items = [
            {
                "domain": "narrativeqa",
                "length": "short",
                "context": "x" * 5000,
                "question": "q",
                "choice_A": "a",
                "choice_B": "b",
                "choice_C": "c",
                "choice_D": "d",
            },
            {
                "domain": "narrativeqa",
                "length": "short",
                "context": "x" * 100,  # shortest
                "question": "q",
                "choice_A": "a",
                "choice_B": "b",
                "choice_C": "c",
                "choice_D": "d",
            },
            {
                "domain": "narrativeqa",
                "length": "short",
                "context": "x" * 1000,
                "question": "q",
                "choice_A": "a",
                "choice_B": "b",
                "choice_C": "c",
                "choice_D": "d",
            },
        ]
        monkeypatch.setattr(workload, "_load_dataset", lambda *a, **k: items)
        monkeypatch.setattr(workload, "_read_template", lambda _: "$DOC$")
        monkeypatch.setattr(
            workload,
            "_tokenize_count",
            lambda prompt, model: len(prompt),
        )

        out = workload.load_samples("longbench:narrativeqa:lengths=short:n=2")
        assert len(out) == 2
        # Picked the 100-char and 1000-char items, in that order; the 5000-char
        # one is skipped.
        token_counts = sorted(s.prompt_tokens for s in out)
        assert token_counts == [100, 1000]


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
    def __init__(
        self,
        layer_idx,
        gpu_resident,
        cpu_resident,
        arena_cap=16,
        block_size=256,
        h_kv=4,
        head_dim=64,
        dtype_bytes=2,
    ):
        self.layer_idx = layer_idx
        self._gpu_resident = gpu_resident
        self._cpu_resident = cpu_resident
        self._stats = FakeStats()
        _n = gpu_resident
        self._slot_map = type("M", (), {"size": lambda self, _n=_n: _n})()
        self._cpu_slots = {i: i for i in range(cpu_resident)}

        class _GpuK:
            def __init__(self, cap, bs, h, d, dbytes):
                self.shape = (cap, bs, h, d)
                self._numel = cap * bs * h * d
                self._dbytes = dbytes

            def numel(self):
                return self._numel

            def element_size(self):
                return self._dbytes

        self.gpu_k = _GpuK(arena_cap, block_size, h_kv, head_dim, dtype_bytes)

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
        assert snap["quest.arena_total_bytes"] is None
        # Dense fallback: when scheduler bookkeeping is unreachable
        # (kv_useful=None), actual_used uses torch.allocated_bytes directly.
        # That's the authoritative figure for vLLM-held memory in dense mode.
        assert snap["vllm.actual_used_bytes"] == 100
        assert snap["vllm.actual_used_peak_bytes"] == 150

    def test_arena_total_sums_across_quest_layers(self):
        from benchmarks.quest_memory_probe import probes

        # Two Quest layers, identical arena geometry.
        # cap=16, block_size=256, h_kv=4, head_dim=64, fp16 (2 bytes).
        # per-layer K bytes = 16*256*4*64*2 = 2_097_152
        # K+V → ×2; two layers → ×2 again.
        tms = [FakeTM(0, 5, 0), FakeTM(1, 5, 0)]
        assert probes._arena_total_bytes(tms) == 2 * 2 * 2_097_152

    def test_arena_total_empty_returns_zero(self):
        from benchmarks.quest_memory_probe import probes

        assert probes._arena_total_bytes([]) == 0

    def test_essential_subtracts_arena_in_quest_mode(self, monkeypatch):
        from benchmarks.quest_memory_probe import probes

        # torch.allocated must exceed kv_pool + arena so essential stays
        # positive (probe clamps to 0 otherwise).
        monkeypatch.setattr(
            probes,
            "_torch_metrics",
            lambda: {
                "torch.allocated_bytes": 15 * 1024**3,
                "torch.reserved_bytes": 16 * 1024**3,
                "torch.peak_allocated_bytes": 17 * 1024**3,
                "torch.active_bytes": 15 * 1024**3,
            },
        )
        monkeypatch.setattr(
            probes,
            "_nvml_metrics",
            lambda: {"nvml.gpu_used_bytes": 0, "nvml.gpu_total_bytes": 0},
        )
        # Two layers × per-layer arena = 2 * 2_097_152 * 2 = 8_388_608 bytes
        # (cap=16, bs=256, h=4, d=64, fp16; ×2 for K+V; ×2 layers)
        tms = [FakeTM(0, 5, 0), FakeTM(1, 5, 0)]
        worker = FakeWorker(tms)  # available_kv_cache_memory_bytes = 10 GiB
        snap = probes.probe_snapshot(worker, bytes_per_block=4096)

        arena_total = 8_388_608
        kv_pool = 10 * 1024**3
        # essential = torch.allocated - kv_pool - arena_total
        assert snap["quest.arena_total_bytes"] == arena_total
        assert snap["vllm.engine_essential_bytes"] == (
            15 * 1024**3 - kv_pool - arena_total
        )
        # actual_used = essential + arena_total
        assert snap["vllm.actual_used_bytes"] == (
            snap["vllm.engine_essential_bytes"] + arena_total
        )
        # peak essential uses torch.peak_allocated_bytes instead
        assert snap["vllm.engine_essential_peak_bytes"] == (
            17 * 1024**3 - kv_pool - arena_total
        )
        assert snap["vllm.actual_used_peak_bytes"] == (
            snap["vllm.engine_essential_peak_bytes"] + arena_total
        )

    def test_actual_used_dense_path_uses_kv_useful(self, monkeypatch):
        from benchmarks.quest_memory_probe import probes

        monkeypatch.setattr(
            probes,
            "_torch_metrics",
            lambda: {
                "torch.allocated_bytes": 8 * 1024**3,
                "torch.reserved_bytes": 9 * 1024**3,
                "torch.peak_allocated_bytes": 9 * 1024**3,
                "torch.active_bytes": 8 * 1024**3,
            },
        )
        monkeypatch.setattr(
            probes,
            "_nvml_metrics",
            lambda: {"nvml.gpu_used_bytes": 0, "nvml.gpu_total_bytes": 0},
        )

        class DenseScheduler:
            class _Mgr:
                num_used_blocks = 100

            kv_cache_manager = _Mgr()

        class DenseRunner:
            model_memory_usage = 6 * 1024**3

        class DenseWorker:
            model_runner = DenseRunner()
            available_kv_cache_memory_bytes = 4 * 1024**3
            scheduler = DenseScheduler()

        snap = probes.probe_snapshot(DenseWorker(), bytes_per_block=4096)

        # Dense: arena_total is None (no Quest), kv_useful = 100 * 4096
        assert snap["quest.arena_total_bytes"] is None
        kv_useful = 100 * 4096
        # essential = torch.allocated - kv_pool - 0
        expected_essential = 8 * 1024**3 - 4 * 1024**3
        assert snap["vllm.engine_essential_bytes"] == expected_essential
        assert snap["vllm.actual_used_bytes"] == expected_essential + kv_useful
        # Peak uses torch.peak_allocated_bytes (9 GiB here)
        expected_essential_peak = 9 * 1024**3 - 4 * 1024**3
        assert snap["vllm.engine_essential_peak_bytes"] == expected_essential_peak
        assert snap["vllm.actual_used_peak_bytes"] == (
            expected_essential_peak + kv_useful
        )


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


class TestCsvWriter:
    def test_writes_union_of_keys(self, tmp_path):
        from benchmarks.quest_memory_probe.csv_writer import write_rows

        rows = [
            {
                "ts_ms": 1,
                "phase": "sampling",
                "torch.allocated_bytes": 100,
                "quest.gpu_resident_blocks": 5,
            },
            {
                "ts_ms": 2,
                "phase": "sample_end",
                "sample_id": "synthetic/short/0",
                "prompt_tokens": 2048,
                "gen_tokens": 64,
                "latency_s": 1.23,
            },
        ]
        out = tmp_path / "samples.csv"
        write_rows(out, rows)

        text = out.read_text()
        header = text.splitlines()[0].split(",")
        for k in (
            "ts_ms",
            "phase",
            "torch.allocated_bytes",
            "quest.gpu_resident_blocks",
            "sample_id",
            "prompt_tokens",
            "gen_tokens",
            "latency_s",
        ):
            assert k in header

    def test_missing_fields_become_empty(self, tmp_path):
        from benchmarks.quest_memory_probe.csv_writer import write_rows

        rows = [
            {"ts_ms": 1, "phase": "sampling", "torch.allocated_bytes": 100},
            {"ts_ms": 2, "phase": "sampling", "torch.reserved_bytes": 200},
        ]
        out = tmp_path / "x.csv"
        write_rows(out, rows)
        lines = out.read_text().splitlines()
        assert len(lines) == 3
        header = lines[0].split(",")
        i_alloc = header.index("torch.allocated_bytes")
        cells = lines[2].split(",")
        assert cells[i_alloc] == ""

    def test_none_serialized_as_empty(self, tmp_path):
        from benchmarks.quest_memory_probe.csv_writer import write_rows

        rows = [{"ts_ms": 1, "phase": "sampling", "quest.topk_hit_ratio": None}]
        out = tmp_path / "x.csv"
        write_rows(out, rows)
        lines = out.read_text().splitlines()
        header = lines[0].split(",")
        i = header.index("quest.topk_hit_ratio")
        assert lines[1].split(",")[i] == ""

    def test_empty_rows_writes_no_file(self, tmp_path):
        from benchmarks.quest_memory_probe.csv_writer import write_rows

        out = tmp_path / "empty.csv"
        write_rows(out, [])
        assert not out.exists()


class TestSummary:
    def test_aggregate_per_sample(self):
        from benchmarks.quest_memory_probe.summary import aggregate_samples

        rows = [
            {
                "ts_ms": 0,
                "phase": "sample_start",
                "sample_id": "s0",
                "prompt_tokens": 1000,
            },
            {
                "ts_ms": 1,
                "phase": "sampling",
                "nvml.gpu_used_bytes": 100,
                "torch.allocated_bytes": 80,
                "vllm.gpu_kv_useful_bytes": 30,
                "vllm.kv_pool_slack_bytes": 70,
                "quest.topk_hit_ratio": 0.8,
            },
            {
                "ts_ms": 2,
                "phase": "sampling",
                "nvml.gpu_used_bytes": 120,
                "torch.allocated_bytes": 90,
                "vllm.gpu_kv_useful_bytes": 40,
                "vllm.kv_pool_slack_bytes": 60,
                "quest.topk_hit_ratio": 0.9,
            },
            {
                "ts_ms": 3,
                "phase": "sampling",
                "nvml.gpu_used_bytes": 110,
                "torch.allocated_bytes": 85,
                "vllm.gpu_kv_useful_bytes": 35,
                "vllm.kv_pool_slack_bytes": 65,
                "quest.topk_hit_ratio": 0.85,
            },
            {
                "ts_ms": 4,
                "phase": "sample_end",
                "sample_id": "s0",
                "gen_tokens": 64,
                "latency_s": 4.0,
            },
            {
                "ts_ms": 5,
                "phase": "sample_start",
                "sample_id": "s1",
                "prompt_tokens": 50000,
            },
            {"ts_ms": 6, "phase": "oom_at_sample", "sample_id": "s1"},
        ]
        per_sample = aggregate_samples(rows)
        assert len(per_sample) == 2
        s0 = per_sample[0]
        assert s0["sample_id"] == "s0"
        assert s0["prompt_tokens"] == 1000
        assert s0["gen_tokens"] == 64
        assert s0["oom"] is False
        assert s0["peak_nvml_used_bytes"] == 120
        assert s0["peak_torch_allocated_bytes"] == 90
        assert s0["peak_kv_useful_bytes"] == 40
        # median of slack [70, 60, 65] = 65
        assert s0["mean_kv_slack_bytes"] == pytest.approx(65)
        assert s0["mean_topk_hit_ratio"] == pytest.approx(0.85)

        s1 = per_sample[1]
        assert s1["sample_id"] == "s1"
        assert s1["oom"] is True
        # OOM samples report whatever sampling rows accumulated (none here)
        assert s1["peak_nvml_used_bytes"] == 0

    def test_aggregate_handles_no_sampling_rows(self):
        from benchmarks.quest_memory_probe.summary import aggregate_samples

        rows = [
            {
                "ts_ms": 0,
                "phase": "sample_start",
                "sample_id": "x",
                "prompt_tokens": 100,
            },
            {
                "ts_ms": 1,
                "phase": "sample_end",
                "sample_id": "x",
                "gen_tokens": 16,
                "latency_s": 0.5,
            },
        ]
        out = aggregate_samples(rows)
        assert len(out) == 1
        assert out[0]["peak_nvml_used_bytes"] == 0
        assert out[0]["mean_topk_hit_ratio"] is None

    def test_aggregate_skips_pre_engine_init_rows(self):
        from benchmarks.quest_memory_probe.summary import aggregate_samples

        rows = [
            {"ts_ms": 0, "phase": "engine_init_done"},
            {
                "ts_ms": 1,
                "phase": "sampling",
                "nvml.gpu_used_bytes": 50,
            },  # before any sample_start; ignored
            {
                "ts_ms": 2,
                "phase": "sample_start",
                "sample_id": "x",
                "prompt_tokens": 100,
            },
            {"ts_ms": 3, "phase": "sampling", "nvml.gpu_used_bytes": 100},
            {
                "ts_ms": 4,
                "phase": "sample_end",
                "sample_id": "x",
                "gen_tokens": 16,
                "latency_s": 0.5,
            },
        ]
        out = aggregate_samples(rows)
        assert len(out) == 1
        assert out[0]["peak_nvml_used_bytes"] == 100

    def test_aggregate_peak_actual_used_and_arena(self):
        from benchmarks.quest_memory_probe.summary import aggregate_samples

        rows = [
            {"phase": "sample_start", "sample_id": "s0", "prompt_tokens": 10},
            {
                "phase": "sampling",
                "vllm.actual_used_bytes": 100,
                "vllm.actual_used_peak_bytes": 150,
                "vllm.engine_essential_peak_bytes": 80,
            },
            {
                "phase": "sampling",
                "vllm.actual_used_bytes": 300,
                "vllm.actual_used_peak_bytes": 350,
                "vllm.engine_essential_peak_bytes": 200,
            },
            {
                "phase": "sample_end",
                "sample_id": "s0",
                "gen_tokens": 5,
                "latency_s": 0.1,
            },
        ]
        out = aggregate_samples(rows)
        assert len(out) == 1
        assert out[0]["peak_actual_used_bytes"] == 300
        assert out[0]["peak_actual_used_peak_bytes"] == 350
        assert out[0]["peak_engine_essential_peak_bytes"] == 200


class TestRunnerHelpers:
    def test_make_quest_engine_kwargs_dense(self):
        from benchmarks.quest_memory_probe.configs import RunConfig
        from benchmarks.quest_memory_probe.runner import (
            _make_engine_kwargs,
        )

        cfg = RunConfig(name="dense", quest_enabled=False)
        kw = _make_engine_kwargs(cfg, quest_json_path=None)
        assert kw["model"] == cfg.model
        assert kw["block_size"] == 256
        assert kw["enforce_eager"] is True
        assert kw["enable_prefix_caching"] is False
        assert kw["enable_chunked_prefill"] is False
        assert "enable_quest_sparse_offload" not in kw

    def test_make_quest_engine_kwargs_quest(self, tmp_path):
        from benchmarks.quest_memory_probe.configs import RunConfig
        from benchmarks.quest_memory_probe.runner import (
            _make_engine_kwargs,
        )

        cfg = RunConfig(
            name="q",
            quest_enabled=True,
            top_k=16,
            gpu_cache_blocks_per_seq=128,
        )
        json_path = tmp_path / "q.json"
        json_path.write_text("{}")
        kw = _make_engine_kwargs(cfg, quest_json_path=str(json_path))
        assert kw["enable_quest_sparse_offload"] is True
        assert kw["quest_config"] == str(json_path)

    def test_make_engine_kwargs_no_hf_overrides_for_native_context(self):
        from benchmarks.quest_memory_probe.configs import RunConfig
        from benchmarks.quest_memory_probe.runner import _make_engine_kwargs

        cfg = RunConfig(name="ok", max_model_len=131072)
        kw = _make_engine_kwargs(cfg, quest_json_path=None)
        assert "hf_overrides" not in kw

    def test_make_engine_kwargs_injects_hf_overrides_when_extending(self):
        # LongBench-v2 long bucket starts at ~167k tokens. Need to extend
        # past Llama-3.2's native 131072. Probe must pass hf_overrides so
        # vLLM doesn't refuse on context overflow.
        from benchmarks.quest_memory_probe.configs import RunConfig
        from benchmarks.quest_memory_probe.runner import _make_engine_kwargs

        cfg = RunConfig(name="big", max_model_len=200000)
        kw = _make_engine_kwargs(cfg, quest_json_path=None)
        assert kw["hf_overrides"] == {"max_position_embeddings": 200000}


class TestRunnerOomDetection:
    def test_is_oom_error_matches_runtime_message(self):
        from benchmarks.quest_memory_probe.runner import _is_oom_error

        err = RuntimeError("CUDA out of memory. Tried to allocate ...")
        assert _is_oom_error(err)

    def test_is_oom_error_rejects_other(self):
        from benchmarks.quest_memory_probe.runner import _is_oom_error

        assert not _is_oom_error(ValueError("nope"))


class TestCli:
    def test_parse_compare_pool_size(self):
        from benchmarks.quest_memory_probe.__main__ import build_parser

        p = build_parser()
        args = p.parse_args(
            [
                "compare-pool-size",
                "--samples",
                "longbench:narrativeqa:lengths=short:n=2",
                "--top-k",
                "16",
                "--pool-sizes",
                "512,128,16",
                "--out-dir",
                "/tmp/x",
            ]
        )
        assert args.subcommand == "compare-pool-size"
        assert args.top_k == 16
        assert args.pool_sizes == [512, 128, 16]
        assert args.out_dir == "/tmp/x"

    def test_parse_dense_vs_quest(self):
        from benchmarks.quest_memory_probe.__main__ import build_parser

        p = build_parser()
        args = p.parse_args(
            [
                "compare-dense-vs-quest",
                "--samples",
                "longbench:narrativeqa:lengths=short:n=2",
                "--top-k",
                "16",
                "--quest-pool",
                "128",
                "--out-dir",
                "/tmp/x",
            ]
        )
        assert args.subcommand == "compare-dense-vs-quest"
        assert args.quest_pool == 128

    def test_parse_oom_sweep(self):
        from benchmarks.quest_memory_probe.__main__ import build_parser

        p = build_parser()
        args = p.parse_args(
            [
                "oom-sweep",
                "--samples",
                "longbench:narrativeqa:lengths=short,medium,long:n=4",
                "--top-k",
                "16",
                "--quest-pool",
                "128",
                "--out-dir",
                "/tmp/x",
            ]
        )
        assert args.subcommand == "oom-sweep"

    def test_args_to_configs_pool_size(self):
        from benchmarks.quest_memory_probe.__main__ import (
            args_to_configs,
            build_parser,
        )

        args = build_parser().parse_args(
            [
                "compare-pool-size",
                "--samples",
                "longbench:narrativeqa:lengths=short:n=1",
                "--top-k",
                "16",
                "--pool-sizes",
                "128,16",
                "--out-dir",
                "/tmp/x",
            ]
        )
        cfgs = args_to_configs(args)
        assert len(cfgs) == 2


class TestReport:
    def _seed_run(self, root):
        """Create a minimal fake out_dir with one dense + one quest cfg."""
        import csv as _csv
        import json as _json

        manifest = {
            "subcommand": "compare-pool-size",
            "timestamp": "2026-06-01T00:00:00",
            "commit": "test",
            "configs": [
                {"name": "dense", "quest_enabled": False},
                {
                    "name": "quest_pool128",
                    "quest_enabled": True,
                    "top_k": 16,
                    "gpu_cache_blocks_per_seq": 128,
                },
            ],
            "argv": ["test"],
        }
        (root / "manifest.json").write_text(_json.dumps(manifest))

        for name, useful, slack in [
            ("dense", 9 * 1024**3, 1 * 1024**3),
            ("quest_pool128", 4 * 1024**3, 6 * 1024**3),
        ]:
            d = root / name
            d.mkdir()
            rows = [
                {"ts_ms": 0, "phase": "engine_init_done"},
                {
                    "ts_ms": 1,
                    "phase": "sample_start",
                    "sample_id": "s0",
                    "prompt_tokens": 2000,
                },
                {
                    "ts_ms": 2,
                    "phase": "sampling",
                    "nvml.gpu_used_bytes": 20 * 1024**3,
                    "torch.allocated_bytes": 18 * 1024**3,
                    "vllm.engine_essential_bytes": 10 * 1024**3,
                    "vllm.gpu_kv_useful_bytes": useful,
                    "vllm.kv_pool_slack_bytes": slack,
                    "vllm.kv_pool_total_bytes": useful + slack,
                    "quest.topk_hit_ratio": 0.7 if "quest" in name else "",
                },
                {
                    "ts_ms": 3,
                    "phase": "sample_end",
                    "sample_id": "s0",
                    "gen_tokens": 64,
                    "latency_s": 1.0,
                },
            ]
            keys: list[str] = []
            seen = set()
            for r in rows:
                for k in r:
                    if k not in seen:
                        seen.add(k)
                        keys.append(k)
            with (d / "samples.csv").open("w", newline="") as f:
                w = _csv.writer(f)
                w.writerow(keys)
                for r in rows:
                    w.writerow(["" if r.get(k) is None else r.get(k, "") for k in keys])
            (d / "summary.json").write_text(
                _json.dumps(
                    {
                        "config": {"name": name, "quest_enabled": "quest" in name},
                        "samples": [
                            {
                                "sample_id": "s0",
                                "prompt_tokens": 2000,
                                "gen_tokens": 64,
                                "latency_s": 1.0,
                                "oom": False,
                                "peak_nvml_used_bytes": 20 * 1024**3,
                                "peak_torch_allocated_bytes": 18 * 1024**3,
                                "peak_kv_useful_bytes": useful,
                                "mean_kv_slack_bytes": slack,
                                "mean_engine_essential_bytes": 10 * 1024**3,
                                "mean_kv_useful_bytes": useful,
                                "mean_topk_hit_ratio": 0.7 if "quest" in name else None,
                            }
                        ],
                    }
                )
            )

    def test_build_report_creates_artifacts(self, tmp_path):
        from benchmarks.quest_memory_probe.report import build_report

        self._seed_run(tmp_path)
        build_report(tmp_path)

        plots = tmp_path / "plots"
        assert (plots / "memory_timeline_dense.png").exists()
        assert (plots / "memory_timeline_quest_pool128.png").exists()
        assert (plots / "memory_peak_bar.png").exists()
        assert (plots / "kv_pool_breakdown.png").exists()
        assert (plots / "topk_hit_ratio_quest_pool128.png").exists()

        md = (tmp_path / "report.md").read_text()
        for header in (
            "# Quest 显存观测报告",
            "## 实验参数",
            "## 配置矩阵",
            "## 关键发现",
            "## 时间序列",
            "## 跨配置峰值对比",
            "## KV 池构成稳态对比",
            "## Top-k 命中率",
        ):
            assert header in md, f"missing section: {header}"

    def test_build_report_takeaway_includes_kv_useful_ratio(self, tmp_path):
        from benchmarks.quest_memory_probe.report import build_report

        self._seed_run(tmp_path)
        build_report(tmp_path)
        md = (tmp_path / "report.md").read_text()
        # quest pool128 useful = 4 GiB, dense useful = 9 GiB → 4/9 ≈ 0.44
        assert "0.44" in md or "44%" in md or "44 %" in md


class TestAttachIntrospection:
    def test_attach_finds_tier_managers_via_attention_layers(self):
        import types

        from benchmarks.quest_memory_probe import probes

        tms = [FakeTM(0, 1, 0), FakeTM(1, 2, 0)]
        layers = [
            types.SimpleNamespace(impl=types.SimpleNamespace(tier_manager=tm))
            for tm in tms
        ]

        class Runner:
            attention_layers = layers

        class Worker:
            model_runner = Runner()

        found = probes._collect_tier_managers(Worker())
        assert len(found) == 2
        assert found[0] is tms[0]

    def test_attach_returns_empty_for_dense_runner(self):
        class Runner:
            pass

        class Worker:
            model_runner = Runner()


@pytest.mark.real_model
class TestE2ESmoke:
    def test_compare_pool_size_smoke(self, tmp_path, monkeypatch):
        """End-to-end: spawns 2 child engines, generates 2 short samples each,
        produces 5 plots + report.md. Total budget: < 10 minutes.

        Forces synthetic prompts so the test does not depend on LongBench
        downloads. Only enabled when pytest is invoked with -m real_model.
        """
        import torch

        if not torch.cuda.is_available():
            pytest.skip("smoke test requires CUDA")

        monkeypatch.setenv("QUEST_MEM_PROBE_FORCE_SYNTHETIC", "1")
        monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

        from benchmarks.quest_memory_probe.__main__ import main

        rc = main(
            [
                "compare-pool-size",
                "--samples",
                "longbench:narrativeqa:lengths=short:n=2",
                "--top-k",
                "16",
                "--pool-sizes",
                "512,16",
                "--probe-interval-ms",
                "200",
                "--max-tokens",
                "16",
                "--out-dir",
                str(tmp_path),
            ]
        )
        assert rc == 0
        for name in (
            "quest_pool512_topk16",
            "quest_pool16_topk16",
        ):
            assert (tmp_path / name / "summary.json").exists()
            assert (tmp_path / name / "samples.csv").exists()

            import json

            summary = json.loads((tmp_path / name / "summary.json").read_text())
            for s in summary["samples"]:
                # actual_used_bytes is essential + arena (Quest mode);
                # peak counter (口径 B) must be ≥ window-max (口径 A).
                assert s["peak_actual_used_bytes"] > 0, s
                assert (
                    s["peak_actual_used_peak_bytes"] >= s["peak_actual_used_bytes"]
                ), s
                assert s["peak_engine_essential_peak_bytes"] > 0, s

        plots = tmp_path / "plots"
        assert (plots / "memory_peak_bar.png").exists()
        assert (plots / "kv_pool_breakdown.png").exists()
        assert (tmp_path / "report.md").read_text()

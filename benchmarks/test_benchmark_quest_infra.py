# SPDX-License-Identifier: Apache-2.0
"""Unit tests for benchmark_quest measurement infra (no real model)."""
from __future__ import annotations

import benchmarks.benchmark_quest as bq


class _FakeEngineCore:
    def __init__(self): self.shut = False
    def shutdown(self, timeout=None): self.shut = True


class _FakeLLM:
    def __init__(self):
        self.llm_engine = type("E", (), {"engine_core": _FakeEngineCore()})()


def test_teardown_engine_calls_engine_core_shutdown():
    llm = _FakeLLM()
    bq._teardown_engine(llm)
    assert llm.llm_engine.engine_core.shut is True


def test_teardown_engine_swallows_errors_on_half_built():
    class Broken: ...
    bq._teardown_engine(Broken())  # must not raise


def test_run_meta_has_required_fields():
    meta = bq._run_meta(argv=["--top-k", "64"])
    for key in ("utc", "argv", "gpu_name", "offload_mode_note"):
        assert key in meta, key
    # start time is the directory key; must be present and ISO-8601 UTC.
    assert meta["utc"].endswith("+00:00") or meta["utc"].endswith("Z")
    assert meta["argv"] == ["--top-k", "64"]


def test_probe_memory_breakdown_fields(monkeypatch):
    import types, torch
    # Fake worker mirroring vllm Worker surface used by the probe.
    runner = types.SimpleNamespace(model_memory_usage=6 * 1024**3)
    worker = types.SimpleNamespace(
        available_kv_cache_memory_bytes=14 * 1024**3,
        model_runner=runner,
    )
    out = bq._probe_memory(worker)
    for k in ("weights_bytes", "kv_reserved_bytes", "torch_reserved_bytes",
              "non_torch_bytes", "cuda_used_bytes", "total_bytes"):
        assert k in out, k
    assert out["weights_bytes"] == 6 * 1024**3
    assert out["kv_reserved_bytes"] == 14 * 1024**3


def test_pass_kind_gates_instrumentation():
    clean = bq.RunConfig(name="x", quest_enabled=True, pass_kind="clean")
    inst = bq.RunConfig(name="x", quest_enabled=True, pass_kind="instrumented")
    # enable_debug_counters must be OFF on the clean pass, ON on instrumented.
    assert bq._debug_counters_for(clean) is False
    assert bq._debug_counters_for(inst) is True


def test_error_stub_record_does_not_break_csv_or_summary(tmp_path):
    """Fault-tolerant sweep: a config that raised is recorded as an error stub
    (name/config/error only). write_versioned + print_summary must handle it
    gracefully alongside a normal record, so one failed point never loses the
    rest of the sweep's data."""
    meta = bq._run_meta(["--demo"])
    good = {
        "name": "dense_clean", "offload_mode": "bypassed",
        "config": {"quest_enabled": False, "pass_kind": "clean",
                   "gpu_cache_blocks_per_seq": 512, "top_k": 64,
                   "offload_mode": "bypassed"},
        "prompt_tokens": 100, "gen_tokens": 8, "latency_s": 0.2,
        "decode_tokens_per_s": 40.0,
    }
    err = {
        "name": "quest_offload_clean",
        "config": {"quest_enabled": True, "pass_kind": "clean",
                   "offload_mode": "bypassed"},
        "error": "RuntimeError('expected offload but evict_d2h=0')",
    }
    out = bq.write_versioned([good, err], meta, tmp_path)
    csv_text = (out / "records.csv").read_text()
    # Both points present; the error stub carries its message.
    assert "dense_clean" in csv_text
    assert "quest_offload_clean" in csv_text
    assert "expected offload" in csv_text
    # records.jsonl has both lines.
    lines = (out / "records.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2
    # print_summary must not raise on the error stub.
    bq.print_summary([good, err])


def test_adjacent_layer_jaccard_simple():
    """slot0={1,2,3}, slot1={2,3,4} for the same (step,seq) -> intersection
    {2,3}=2, union {1,2,3,4}=4 -> Jaccard 0.5, exactly one adjacent pair."""
    per_slot_log = {
        0: [{"step": 0, "seq_id": 0, "block_ids": [1, 2, 3]}],
        1: [{"step": 0, "seq_id": 0, "block_ids": [2, 3, 4]}],
    }
    out = bq.adjacent_layer_jaccard(per_slot_log)
    assert out["mean_jaccard"] == 0.5
    assert out["median_jaccard"] == 0.5
    assert out["n_pairs"] == 1
    assert out["per_pair_mean"] == {"0->1": 0.5}
    assert out["go_no_go"].startswith("GO")


def test_topk_sweep_plan_includes_dense_and_each_topk():
    args = bq.parse_args([
        "--top-k-sweep", "ALL,64,32,16",
        "--num-paragraphs", "40", "--pass", "clean",
    ])
    plan = bq.build_run_plan(args)
    names = [c.name for c in plan]
    assert any(c.name == "dense_clean" for c in plan)
    # one quest point per requested top_k (ALL -> a large top_k >= num blocks)
    assert sum(1 for c in plan if c.quest_enabled) == 4
    topks = sorted(c.top_k for c in plan if c.quest_enabled)
    assert 16 in topks and 32 in topks and 64 in topks


def test_heterogeneous_prompt_blocks_differ():
    """Task 6b (lightweight): the default prompt must NOT be one paragraph
    repeated (that made every KV block identical -> degenerate Quest selection,
    cosine/Jaccard stuck at 1.0). A simple non-homogeneous, reproducible prompt
    is enough — this harness prompt is only for pipeline self-check / smoke. The
    real quality sweep is done by the follow-up team with LongBench (data not on
    this machine). See build_prompt docstring + teamdocs/results/stage1-findings.md."""
    het = bq.build_prompt(40, heterogeneous=True)
    hom = bq.build_prompt(40, heterogeneous=False)
    # Homogeneous = one paragraph repeated many times; heterogeneous must not be.
    assert hom.count(bq._PARAGRAPH.strip()) >= 30
    assert het.count(bq._PARAGRAPH.strip()) == 0
    # Heterogeneous sentences are (almost) all distinct; homogeneous mostly repeat.
    het_sents = [s for s in het.split(". ") if s]
    assert len(set(het_sents)) / max(1, len(het_sents)) > 0.9
    # Deterministic: same input -> same output (reproducible runs).
    assert bq.build_prompt(40) == bq.build_prompt(40)




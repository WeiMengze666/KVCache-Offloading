# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""bind_runtime: model_runner-side single entry point."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def cuda():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")


def _layer(layer_idx, layer_name, num_kv_heads=2, head_size=64, attn_backend=None):
    return SimpleNamespace(
        layer_idx=layer_idx,
        layer_name=layer_name,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        kv_cache_torch_dtype=torch.float16,
        attn_backend=attn_backend,
    )


def _vllm_config(quest_cfg, model_arch="llama"):
    return SimpleNamespace(
        quest_config=quest_cfg,
        model_config=SimpleNamespace(
            architecture=model_arch,
            is_mla=False,
            has_sliding_window=False,
            max_model_len=32768,
        ),
        cache_config=SimpleNamespace(block_size=256),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
        ),
    )


def test_bind_runtime_skips_when_quest_disabled(cuda):
    from vllm.config.quest import QuestConfig
    from vllm.v1.attention.backends.quest.backend import (
        QuestSparseOffloadBackend,
    )
    from vllm.v1.kv_cache_interface import KVCacheConfig

    Q = QuestSparseOffloadBackend
    layer_a = _layer(1, "layer.1", attn_backend=Q)
    layer_b = _layer(2, "layer.2", attn_backend=Q)
    QuestSparseOffloadBackend.bind_runtime(
        vllm_config=_vllm_config(QuestConfig(enabled=False)),
        kv_cache_config=KVCacheConfig(
            num_blocks=12,
            kv_cache_tensors=[],
            kv_cache_groups=[],
        ),
        kv_caches={},
        layers={"layer.1": layer_a, "layer.2": layer_b},
    )
    # No-op when disabled — init_runtime_state must not have run.
    assert getattr(layer_a, "tier_manager", None) is None
    assert getattr(layer_b, "tier_manager", None) is None


def test_bind_runtime_validates_block_size_256(cuda):
    """When validation fails, bind_runtime raises ValueError with reasons."""
    from vllm.config.quest import QuestConfig
    from vllm.v1.attention.backends.quest.backend import (
        QuestSparseOffloadBackend,
    )
    from vllm.v1.kv_cache_interface import KVCacheConfig

    Q = QuestSparseOffloadBackend
    layer_a = _layer(1, "layer.1", attn_backend=Q)
    cfg = _vllm_config(QuestConfig(enabled=True))
    cfg.cache_config = SimpleNamespace(block_size=128)  # not multiple of 256
    with pytest.raises(ValueError, match="256"):
        QuestSparseOffloadBackend.bind_runtime(
            vllm_config=cfg,
            kv_cache_config=KVCacheConfig(
                num_blocks=12,
                kv_cache_tensors=[],
                kv_cache_groups=[],
            ),
            kv_caches={},
            layers={"layer.1": layer_a},
        )
    # Validation must fire before init_runtime_state — no tier_manager attached.
    assert getattr(layer_a, "tier_manager", None) is None


def test_bind_runtime_attaches_tier_manager_using_kv_cache_view(cuda):
    """End-to-end: pass real fake-shaped tensors, get tier_manager wired."""
    from vllm.config.quest import QuestConfig
    from vllm.v1.attention.backends.quest.backend import (
        QuestSparseOffloadBackend,
    )
    from vllm.v1.kv_cache_interface import KVCacheConfig

    quest_cfg = QuestConfig(
        enabled=True,
        full_kv_layers=[0],
        # top_k=3 <= cap-1: default top_k=64 would fail validation against gpu_cache_blocks_per_seq=4
        gpu_cache_blocks_per_seq=4,
        top_k=3,
        cpu_cache_blocks=4,
    )
    Q = QuestSparseOffloadBackend
    layers_dict = {
        "layer.0": _layer(0, "layer.0"),  # full_kv
        "layer.1": _layer(1, "layer.1", attn_backend=Q),  # quest
        "layer.2": _layer(2, "layer.2", attn_backend=Q),  # quest
    }
    fake_kv = {
        "layer.1": torch.empty(
            (12, 2, 256, 2, 64),
            dtype=torch.float16,
            device="cuda",
        ),
        "layer.2": torch.empty(
            (12, 2, 256, 2, 64),
            dtype=torch.float16,
            device="cuda",
        ),
    }
    QuestSparseOffloadBackend.bind_runtime(
        vllm_config=_vllm_config(quest_cfg),
        kv_cache_config=KVCacheConfig(
            num_blocks=12,
            kv_cache_tensors=[],
            kv_cache_groups=[],
        ),
        kv_caches=fake_kv,
        layers=layers_dict,
    )
    assert layers_dict["layer.1"].tier_manager is not None
    assert layers_dict["layer.2"].tier_manager is not None
    # Stage 2A: each Quest layer gets a PRIVATE bounded arena, not a view of
    # the full engine cache. gpu_k must NOT alias the engine tensor, and the
    # arena size is gpu_cache_blocks_per_seq.
    assert (
        layers_dict["layer.1"].tier_manager.gpu_k.data_ptr()
        != fake_kv["layer.1"][:, 0].data_ptr()
    )
    assert (
        layers_dict["layer.1"].tier_manager.gpu_budget
        == quest_cfg.gpu_cache_blocks_per_seq
    )
    # full_kv layer 0: no tier_manager attached.
    assert getattr(layers_dict["layer.0"], "tier_manager", None) is None


def test_bind_runtime_passes_layers_dict_directly(cuda):
    """layers can be a dict[name -> layer] (matches get_layers_from_vllm_config)."""
    from vllm.config.quest import QuestConfig
    from vllm.v1.attention.backends.quest.backend import (
        QuestSparseOffloadBackend,
    )
    from vllm.v1.kv_cache_interface import KVCacheConfig

    Q = QuestSparseOffloadBackend
    layers_dict = {
        "layer.0": _layer(0, "layer.0"),
        **{f"layer.{i}": _layer(i, f"layer.{i}", attn_backend=Q) for i in (1, 2)},
    }
    fake_kv = {
        "layer.1": torch.empty((8, 2, 256, 2, 64), dtype=torch.float16, device="cuda"),
        "layer.2": torch.empty((8, 2, 256, 2, 64), dtype=torch.float16, device="cuda"),
    }
    QuestSparseOffloadBackend.bind_runtime(
        vllm_config=_vllm_config(
            QuestConfig(
                enabled=True,
                full_kv_layers=[0],
                # top_k=3 <= cap-1: default top_k=64 would fail validation against gpu_cache_blocks_per_seq=4
                gpu_cache_blocks_per_seq=4,
                top_k=3,
                cpu_cache_blocks=4,
            )
        ),
        kv_cache_config=KVCacheConfig(
            num_blocks=8,
            kv_cache_tensors=[],
            kv_cache_groups=[],
        ),
        kv_caches=fake_kv,
        layers=layers_dict,
    )
    assert layers_dict["layer.1"].tier_manager is not None


def test_bind_runtime_constructs_stream_pool_when_async_enabled(cuda):
    """Mode 1 enabled: tier_managers all share the same QuestStreamPool."""
    from vllm.config.quest import QuestConfig
    from vllm.v1.attention.backends.quest.async_transfer import (
        QuestStreamPool,
    )
    from vllm.v1.attention.backends.quest.backend import (
        QuestSparseOffloadBackend,
    )
    from vllm.v1.kv_cache_interface import KVCacheConfig

    quest_cfg = QuestConfig(
        enabled=True,
        full_kv_layers=[0],
        gpu_cache_blocks_per_seq=4,
        top_k=3,
        cpu_cache_blocks=4,
        enable_async_prefetch=True,
    )
    Q = QuestSparseOffloadBackend
    layers_dict = {
        "layer.0": _layer(0, "layer.0"),
        "layer.1": _layer(1, "layer.1", attn_backend=Q),
        "layer.2": _layer(2, "layer.2", attn_backend=Q),
    }
    fake_kv = {
        f"layer.{i}": torch.empty(
            (8, 2, 256, 2, 64), dtype=torch.float16, device="cuda"
        )
        for i in (1, 2)
    }
    QuestSparseOffloadBackend.bind_runtime(
        vllm_config=_vllm_config(quest_cfg),
        kv_cache_config=KVCacheConfig(
            num_blocks=8,
            kv_cache_tensors=[],
            kv_cache_groups=[],
        ),
        kv_caches=fake_kv,
        layers=layers_dict,
    )
    pool1 = layers_dict["layer.1"].tier_manager.stream_pool
    pool2 = layers_dict["layer.2"].tier_manager.stream_pool
    assert isinstance(pool1, QuestStreamPool)
    assert pool1 is pool2  # shared singleton


def test_bind_runtime_no_stream_pool_when_async_disabled(cuda):
    """Default path: stream_pool stays None — Phase B sync behavior intact."""
    from vllm.config.quest import QuestConfig
    from vllm.v1.attention.backends.quest.backend import (
        QuestSparseOffloadBackend,
    )
    from vllm.v1.kv_cache_interface import KVCacheConfig

    quest_cfg = QuestConfig(
        enabled=True,
        full_kv_layers=[0],
        gpu_cache_blocks_per_seq=4,
        top_k=3,
        cpu_cache_blocks=4,
        # enable_async_prefetch defaults to False
    )
    Q = QuestSparseOffloadBackend
    layers_dict = {
        "layer.0": _layer(0, "layer.0"),
        "layer.1": _layer(1, "layer.1", attn_backend=Q),
    }
    fake_kv = {
        "layer.1": torch.empty((8, 2, 256, 2, 64), dtype=torch.float16, device="cuda")
    }
    QuestSparseOffloadBackend.bind_runtime(
        vllm_config=_vllm_config(quest_cfg),
        kv_cache_config=KVCacheConfig(
            num_blocks=8,
            kv_cache_tensors=[],
            kv_cache_groups=[],
        ),
        kv_caches=fake_kv,
        layers=layers_dict,
    )
    assert layers_dict["layer.1"].tier_manager.stream_pool is None


def test_bind_runtime_attaches_quest_refs_to_layers(cuda):
    """For Mode 2: each Quest layer must have _quest_layer_tm_registry,
    _quest_layer_indices_view, and _quest_config_ref attributes after
    bind_runtime (when async + window are configured)."""
    from vllm.config.quest import QuestConfig
    from vllm.v1.attention.backends.quest.backend import (
        QuestSparseOffloadBackend,
    )
    from vllm.v1.kv_cache_interface import KVCacheConfig

    quest_cfg = QuestConfig(
        enabled=True,
        full_kv_layers=[0],
        gpu_cache_blocks_per_seq=4,
        top_k=3,
        cpu_cache_blocks=4,
        enable_async_prefetch=True,
        prefetch_window_blocks=2,
    )
    Q = QuestSparseOffloadBackend
    layers_dict = {
        "layer.0": _layer(0, "layer.0"),
        "layer.1": _layer(1, "layer.1", attn_backend=Q),
        "layer.2": _layer(2, "layer.2", attn_backend=Q),
    }
    fake_kv = {
        f"layer.{i}": torch.empty(
            (8, 2, 256, 2, 64), dtype=torch.float16, device="cuda"
        )
        for i in (1, 2)
    }
    QuestSparseOffloadBackend.bind_runtime(
        vllm_config=_vllm_config(quest_cfg),
        kv_cache_config=KVCacheConfig(
            num_blocks=8,
            kv_cache_tensors=[],
            kv_cache_groups=[],
        ),
        kv_caches=fake_kv,
        layers=layers_dict,
    )
    for idx in (1, 2):
        layer = layers_dict[f"layer.{idx}"]
        assert layer._quest_config_ref is quest_cfg
        # Registry maps quest layer_idx -> tier_manager.
        registry = layer._quest_layer_tm_registry
        assert registry[1] is layers_dict["layer.1"].tier_manager
        assert registry[2] is layers_dict["layer.2"].tier_manager
        # Indices view: list of quest layer global indices.
        view = layer._quest_layer_indices_view
        assert sorted(view) == [1, 2]


def test_bind_runtime_does_not_attach_refs_when_async_disabled(cuda):
    """When stream_pool is None (sync mode), no Mode-2 prefetch refs are
    attached. _quest_config_ref IS attached on every Quest layer regardless of
    async mode — the Stage 2C-v2 footprint_kvshare forward path reads it to
    decide whether to take the Quest-owned-write path (it must work in the
    default sync mode too). The Mode-2-only registry stays absent."""
    from vllm.config.quest import QuestConfig
    from vllm.v1.attention.backends.quest.backend import (
        QuestSparseOffloadBackend,
    )
    from vllm.v1.kv_cache_interface import KVCacheConfig

    quest_cfg = QuestConfig(
        enabled=True,
        full_kv_layers=[0],
        gpu_cache_blocks_per_seq=4,
        top_k=3,
        cpu_cache_blocks=4,
        # async off → sync path, no Mode 2 plumbing
    )
    Q = QuestSparseOffloadBackend
    layers_dict = {
        "layer.0": _layer(0, "layer.0"),
        "layer.1": _layer(1, "layer.1", attn_backend=Q),
    }
    fake_kv = {
        "layer.1": torch.empty((8, 2, 256, 2, 64), dtype=torch.float16, device="cuda")
    }
    QuestSparseOffloadBackend.bind_runtime(
        vllm_config=_vllm_config(quest_cfg),
        kv_cache_config=KVCacheConfig(
            num_blocks=8,
            kv_cache_tensors=[],
            kv_cache_groups=[],
        ),
        kv_caches=fake_kv,
        layers=layers_dict,
    )
    # _quest_config_ref IS attached now (footprint_kvshare needs it in sync
    # mode); only the Mode-2 prefetch registry stays absent.
    assert layers_dict["layer.1"]._quest_config_ref is quest_cfg
    assert not hasattr(layers_dict["layer.1"], "_quest_layer_tm_registry")


def test_bind_runtime_stashes_selection_callable_torch():
    """init_runtime_state pre-resolves selection_impl=='torch' and stashes
    the callable on each Quest layer as `_quest_selection_callable_ref`."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    from vllm.config.quest import QuestConfig
    from vllm.v1.attention.backends.quest.backend import (
        QuestSparseOffloadBackend,
    )
    from vllm.v1.attention.ops.quest_selection_torch import (
        quest_selection_torch,
    )

    quest_cfg = QuestConfig(
        enabled=True,
        block_size=256,
        top_k=4,
        gpu_cache_blocks_per_seq=8,
        full_kv_layers=[],
        selection_impl="torch",
    )
    layer = SimpleNamespace(
        layer_idx=2,
        num_kv_heads=4,
        head_size=128,
        num_heads=8,
        layer_name="quest.0",
    )
    QuestSparseOffloadBackend.init_runtime_state(
        layers=[layer],
        block_size=256,
        num_kv_heads=4,
        head_size=128,
        max_blocks_total=16,
        dtype=torch.float16,
        quest_config=quest_cfg,
        kv_caches=None,
    )
    assert getattr(layer, "_quest_selection_callable_ref", None) is (
        quest_selection_torch
    )


def test_bind_runtime_stashes_selection_callable_triton():
    """Same as above but for selection_impl=='triton'."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    from vllm.config.quest import QuestConfig
    from vllm.v1.attention.backends.quest.backend import (
        QuestSparseOffloadBackend,
    )
    from vllm.v1.attention.ops.quest_selection_triton import (
        quest_selection_triton,
    )

    quest_cfg = QuestConfig(
        enabled=True,
        block_size=256,
        top_k=4,
        gpu_cache_blocks_per_seq=8,
        full_kv_layers=[],
        selection_impl="triton",
    )
    layer = SimpleNamespace(
        layer_idx=2,
        num_kv_heads=4,
        head_size=128,
        num_heads=8,
        layer_name="quest.0",
    )
    QuestSparseOffloadBackend.init_runtime_state(
        layers=[layer],
        block_size=256,
        num_kv_heads=4,
        head_size=128,
        max_blocks_total=16,
        dtype=torch.float16,
        quest_config=quest_cfg,
        kv_caches=None,
    )
    assert getattr(layer, "_quest_selection_callable_ref", None) is (
        quest_selection_triton
    )


def test_real_engine_layer_gets_bounded_arena():
    """init_runtime_state must give each Quest layer a private arena of
    gpu_cache_blocks_per_seq blocks, NOT a view of the full engine cache."""
    import torch
    import pytest
    from types import SimpleNamespace
    from vllm.config.quest import QuestConfig
    from vllm.v1.attention.backends.quest.backend import QuestSparseOffloadBackend

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    cap = 8
    block_size, h_kv, hd = 256, 2, 64
    full_blocks = 64  # engine cache is much bigger than the arena
    qcfg = QuestConfig(enabled=True, top_k=8, gpu_cache_blocks_per_seq=cap,
                       full_kv_layers=[0, 1], block_size=block_size)
    layer = SimpleNamespace(
        layer_idx=2, layer_name="model.layers.2.self_attn.attn",
        num_kv_heads=h_kv, head_size=hd, kv_cache_torch_dtype=torch.float16,
    )
    engine_kv = torch.zeros(full_blocks, 2, block_size, h_kv, hd,
                            dtype=torch.float16, device="cuda")
    QuestSparseOffloadBackend.init_runtime_state(
        layers=[layer], block_size=block_size, num_kv_heads=h_kv, head_size=hd,
        max_blocks_total=full_blocks, dtype=torch.float16, quest_config=qcfg,
        kv_caches={"model.layers.2.self_attn.attn": engine_kv},
    )
    tm = layer.tier_manager
    assert tm.gpu_budget == cap, f"arena must be cap={cap}, got {tm.gpu_budget}"
    assert tm.gpu_k.shape[0] == cap
    assert tm.gpu_k.data_ptr() != engine_kv[:, 0].data_ptr()  # not aliased
    assert tm.gpu_pool_aliases_kv_cache is False
    assert tm.engine_kv_cache is engine_kv  # kept as trim source

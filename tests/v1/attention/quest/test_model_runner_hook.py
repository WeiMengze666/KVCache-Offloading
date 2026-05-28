# SPDX-License-Identifier: Apache-2.0
"""Confirm GPUModelRunner.initialize_kv_cache invokes Quest bind_runtime
when quest_config is enabled, and skips it otherwise."""
from __future__ import annotations

from unittest.mock import patch

import pytest
import torch


@pytest.fixture
def cuda():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")


def test_bind_runtime_called_when_quest_enabled(cuda, monkeypatch):
    """Patch bind_runtime; build a minimal GPUModelRunner shim and call
    initialize_kv_cache. Verify bind_runtime was invoked exactly once."""
    pytest.importorskip("vllm")

    from unittest.mock import MagicMock
    from vllm.v1.attention.backends.quest.backend import (
        QuestSparseOffloadBackend,
    )

    # Track invocation.
    call_log = []

    def _fake_bind(*, vllm_config, kv_cache_config, kv_caches, layers):
        call_log.append({
            "vllm_config": vllm_config,
            "kv_cache_config": kv_cache_config,
            "kv_caches": kv_caches,
            "layers": layers,
        })

    # We test the call site at the model_runner level by importing the
    # function and asserting it is gated correctly. Building a real
    # GPUModelRunner requires full vLLM init machinery; instead, isolate
    # the new code into a helper `_invoke_quest_bind_runtime` in the model
    # runner and unit-test that.
    from vllm.v1.worker.gpu.model_runner import _invoke_quest_bind_runtime

    monkeypatch.setattr(
        QuestSparseOffloadBackend, "bind_runtime", _fake_bind,
    )

    # Build minimum-viable args.
    fake_vllm_config = MagicMock()
    fake_vllm_config.quest_config = MagicMock()
    fake_vllm_config.quest_config.enabled = True
    fake_kv_cache_config = MagicMock()
    fake_kv_caches = {}
    fake_static_forward_context = {}
    _invoke_quest_bind_runtime(
        vllm_config=fake_vllm_config,
        kv_cache_config=fake_kv_cache_config,
        kv_caches=fake_kv_caches,
        static_forward_context=fake_static_forward_context,
    )
    assert len(call_log) == 1


def test_bind_runtime_skipped_when_quest_config_none(monkeypatch):
    """When vllm_config.quest_config is None, the helper short-circuits
    without importing the quest backend module."""
    from unittest.mock import MagicMock
    import sys

    # Snapshot whether the quest backend module is loaded.
    pre = "vllm.v1.attention.backends.quest.backend" in sys.modules

    from vllm.v1.worker.gpu.model_runner import _invoke_quest_bind_runtime

    fake_vllm_config = MagicMock()
    fake_vllm_config.quest_config = None
    _invoke_quest_bind_runtime(
        vllm_config=fake_vllm_config,
        kv_cache_config=MagicMock(),
        kv_caches={},
        static_forward_context={},
    )
    # If quest wasn't loaded before, it MUST NOT be loaded after a None config.
    if not pre:
        assert (
            "vllm.v1.attention.backends.quest.backend" not in sys.modules
        )


def test_bind_runtime_skipped_when_quest_config_disabled(monkeypatch):
    """quest_config.enabled=False: same short-circuit."""
    from unittest.mock import MagicMock
    from vllm.v1.attention.backends.quest.backend import (
        QuestSparseOffloadBackend,
    )

    call_log = []

    def _fake_bind(**kwargs):
        call_log.append(kwargs)

    monkeypatch.setattr(
        QuestSparseOffloadBackend, "bind_runtime", _fake_bind,
    )

    from vllm.v1.worker.gpu.model_runner import _invoke_quest_bind_runtime

    fake_vllm_config = MagicMock()
    fake_vllm_config.quest_config = MagicMock(enabled=False)
    _invoke_quest_bind_runtime(
        vllm_config=fake_vllm_config,
        kv_cache_config=MagicMock(),
        kv_caches={},
        static_forward_context={},
    )
    assert call_log == []

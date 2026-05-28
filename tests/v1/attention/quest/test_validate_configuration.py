# SPDX-License-Identifier: Apache-2.0
"""validate_quest_configuration enforces R1 + Quest invariants."""
from __future__ import annotations

from types import SimpleNamespace

import pytest


def _model_config(architecture="llama"):
    return SimpleNamespace(
        architecture=architecture,
        is_mla=False,
        has_sliding_window=False,
    )


def _quest_cfg(**overrides):
    from vllm.config.quest import QuestConfig
    cfg = QuestConfig(enabled=True)
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def test_block_size_must_be_multiple_of_256():
    """Discovered empirically by the R1 spike on flash_attn 2.8.3 SM89."""
    from vllm.v1.attention.backends.quest.backend import (
        QuestSparseOffloadBackend,
    )

    errors = QuestSparseOffloadBackend.validate_quest_configuration(
        model_config=_model_config(),
        cache_config=SimpleNamespace(block_size=128),
        quest_config=_quest_cfg(),
    )
    assert any("256" in e for e in errors), errors


def test_block_size_256_is_accepted():
    from vllm.v1.attention.backends.quest.backend import (
        QuestSparseOffloadBackend,
    )

    errors = QuestSparseOffloadBackend.validate_quest_configuration(
        model_config=_model_config(),
        cache_config=SimpleNamespace(block_size=256),
        quest_config=_quest_cfg(),
    )
    assert errors == []


def test_top_k_exceeds_gpu_budget_rejected():
    from vllm.v1.attention.backends.quest.backend import (
        QuestSparseOffloadBackend,
    )

    errors = QuestSparseOffloadBackend.validate_quest_configuration(
        model_config=_model_config(),
        cache_config=SimpleNamespace(block_size=256),
        quest_config=_quest_cfg(top_k=300, gpu_cache_blocks_per_seq=256),
    )
    assert any("gpu_cache_blocks_per_seq" in e for e in errors)


def test_unknown_architecture_in_error_mode():
    from vllm.v1.attention.backends.quest.backend import (
        QuestSparseOffloadBackend,
    )

    errors = QuestSparseOffloadBackend.validate_quest_configuration(
        model_config=_model_config(architecture="deepseek_v2"),
        cache_config=SimpleNamespace(block_size=256),
        quest_config=_quest_cfg(),
    )
    assert any("deepseek" in e.lower() for e in errors)


def test_unknown_architecture_in_fallback_mode_returns_no_errors():
    """unsupported_model_policy=fallback lets selector pick the default."""
    from vllm.v1.attention.backends.quest.backend import (
        QuestSparseOffloadBackend,
    )

    errors = QuestSparseOffloadBackend.validate_quest_configuration(
        model_config=_model_config(architecture="deepseek_v2"),
        cache_config=SimpleNamespace(block_size=256),
        quest_config=_quest_cfg(unsupported_model_policy="fallback"),
    )
    assert errors == []


def test_quest_disabled_returns_no_errors():
    """If the gate is off, validation always passes."""
    from vllm.config.quest import QuestConfig
    from vllm.v1.attention.backends.quest.backend import (
        QuestSparseOffloadBackend,
    )

    errors = QuestSparseOffloadBackend.validate_quest_configuration(
        model_config=_model_config("deepseek_v2"),
        cache_config=SimpleNamespace(block_size=128),
        quest_config=QuestConfig(enabled=False),
    )
    assert errors == []

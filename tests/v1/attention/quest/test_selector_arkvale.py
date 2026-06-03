# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""selector.py routes ArkVale to QuestSparseOffloadBackend."""

import sys
import types

import torch


def _patch_current_vllm_config(monkeypatch, *, quest=None, arkvale=None):
    fake_vc = types.SimpleNamespace(
        quest_config=quest,
        arkvale_config=arkvale,
        cache_config=types.SimpleNamespace(
            user_specified_block_size=False,
            block_size=256,
        ),
        attention_config=types.SimpleNamespace(
            backend=None,
            use_non_causal=False,
        ),
    )

    def fake_current():
        return fake_vc

    monkeypatch.setattr(
        "vllm.config.get_current_vllm_config",
        fake_current,
    )


def test_selector_registers_quest_when_arkvale_enabled(monkeypatch):
    from vllm.config.arkvale import ArkValeConfig

    _patch_current_vllm_config(
        monkeypatch,
        arkvale=ArkValeConfig(enabled=True),
    )
    for m in list(sys.modules):
        if "vllm.v1.attention.backends.quest.registration" in m:
            sys.modules.pop(m, None)

    import contextlib

    from vllm.v1.attention.selector import get_attn_backend

    with contextlib.suppress(Exception):
        # Backend lookup may fail downstream (no real layers); we only
        # care quest.registration was lazily imported.
        get_attn_backend(
            head_size=64,
            dtype=torch.float16,
            kv_cache_dtype=None,
        )

    assert any(
        "vllm.v1.attention.backends.quest.registration" in m for m in sys.modules
    ), "quest registration was not lazily imported on ArkVale enable"

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Helper that picks the active sparse cfg (Quest or ArkVale)."""

from types import SimpleNamespace

from vllm.config import ArkValeConfig, QuestConfig, get_active_sparse_cfg


def _vc(quest=None, arkvale=None):
    return SimpleNamespace(quest_config=quest, arkvale_config=arkvale)


def test_returns_none_when_neither_set():
    assert get_active_sparse_cfg(_vc()) is None


def test_returns_none_when_both_disabled():
    q = QuestConfig(enabled=False)
    a = ArkValeConfig(enabled=False)
    assert get_active_sparse_cfg(_vc(q, a)) is None


def test_returns_quest_when_only_quest_enabled():
    q = QuestConfig(enabled=True)
    out = get_active_sparse_cfg(_vc(q, None))
    assert out is q


def test_returns_arkvale_when_only_arkvale_enabled():
    a = ArkValeConfig(enabled=True)
    out = get_active_sparse_cfg(_vc(None, a))
    assert out is a


def test_arkvale_wins_when_both_enabled():
    """If both somehow slipped past the EngineArgs mutex, prefer ArkVale.
    Unreachable in practice; helper picks deterministically rather than
    crashing — EngineArgs mutex is the real guard."""
    q = QuestConfig(enabled=True)
    a = ArkValeConfig(enabled=True)
    out = get_active_sparse_cfg(_vc(q, a))
    assert out is a


def test_attribute_absent_works():
    """vllm_config from older callers may lack arkvale_config attr."""
    q = QuestConfig(enabled=True)
    vc = SimpleNamespace(quest_config=q)  # no arkvale_config attr
    assert get_active_sparse_cfg(vc) is q

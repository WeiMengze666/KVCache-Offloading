# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage 2C-v2 Task A1/A2: kv-share scratch-target selection.

`_quest_kvshare_target` is the pure decision the construction-time wiring makes:
given this layer's index, the quest config, and the layer names already
registered (in construction order), return the scratch layer name this Quest
layer should kv-share to — or None if it must keep its own KV (full-KV layers
and the scratch layer itself).

Keeping this a pure helper (no engine, no quest-module import) lets us test the
designation logic directly and keeps the wiring site in attention.py free of
quest imports (zero-impact-when-disabled).
"""
from __future__ import annotations


def _names(n):
    """Layer names in construction order, matching vLLM's prefix scheme."""
    return [f"model.layers.{i}.self_attn.attn" for i in range(n)]


def test_full_kv_layers_get_no_target():
    from vllm.model_executor.layers.attention.attention import (
        _quest_kvshare_target,
    )

    names = _names(6)
    full_kv = [0, 1]
    # layers 0,1 are full-KV -> keep own KV (no share target)
    for idx in (0, 1):
        assert _quest_kvshare_target(idx, full_kv, names[idx], names) is None


def test_first_quest_layer_is_the_scratch_and_shares_nothing():
    from vllm.model_executor.layers.attention.attention import (
        _quest_kvshare_target,
    )

    names = _names(6)
    full_kv = [0, 1]
    # layer 2 is the first quest layer = scratch -> keeps its own (no target)
    assert _quest_kvshare_target(2, full_kv, names[2], names) is None


def test_later_quest_layers_share_to_scratch():
    from vllm.model_executor.layers.attention.attention import (
        _quest_kvshare_target,
    )

    names = _names(6)
    full_kv = [0, 1]
    scratch = names[2]
    for idx in (3, 4, 5):
        assert (
            _quest_kvshare_target(idx, full_kv, names[idx], names) == scratch
        )


def test_scratch_is_lowest_quest_index_even_with_gappy_full_kv():
    from vllm.model_executor.layers.attention.attention import (
        _quest_kvshare_target,
    )

    names = _names(8)
    # full-KV = {0, 3}; lowest quest index is 1 -> scratch = layer 1
    full_kv = [0, 3]
    assert _quest_kvshare_target(1, full_kv, names[1], names) is None  # scratch
    assert _quest_kvshare_target(3, full_kv, names[3], names) is None  # full-KV
    assert _quest_kvshare_target(2, full_kv, names[2], names) == names[1]
    assert _quest_kvshare_target(4, full_kv, names[4], names) == names[1]

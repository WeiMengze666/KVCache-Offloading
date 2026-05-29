# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase D: selection_impl dispatcher contract."""

from __future__ import annotations

import pytest


def test_resolve_torch_returns_torch_callable():
    from vllm.v1.attention.ops.quest_selection_dispatch import (
        _resolve_selection_callable,
    )
    from vllm.v1.attention.ops.quest_selection_torch import (
        quest_selection_torch,
    )

    fn = _resolve_selection_callable("torch")
    assert fn is quest_selection_torch


def test_resolve_triton_returns_triton_callable():
    from vllm.v1.attention.ops.quest_selection_dispatch import (
        _resolve_selection_callable,
    )
    from vllm.v1.attention.ops.quest_selection_triton import (
        quest_selection_triton,
    )

    fn = _resolve_selection_callable("triton")
    assert fn is quest_selection_triton


def test_resolve_cuda_returns_cuda_callable_when_built():
    """If vllm._C is importable, dispatcher returns the cuda wrapper."""
    pytest.importorskip("vllm._C")
    from vllm.v1.attention.ops.quest_selection_cuda import (
        quest_selection_cuda,
    )
    from vllm.v1.attention.ops.quest_selection_dispatch import (
        _resolve_selection_callable,
    )

    fn = _resolve_selection_callable("cuda")
    assert fn is quest_selection_cuda


def test_resolve_unknown_impl_raises_value_error():
    from vllm.v1.attention.ops.quest_selection_dispatch import (
        _resolve_selection_callable,
    )

    with pytest.raises(ValueError, match="selection_impl"):
        _resolve_selection_callable("cublas")  # type: ignore[arg-type]

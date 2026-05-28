# SPDX-License-Identifier: Apache-2.0
"""BlockResidency state machine + invariants."""
from __future__ import annotations

import pytest


def test_state_enum_values():
    from vllm.v1.attention.backends.quest.cache.residency import (
        BlockResidency,
        ResidencyState,
    )

    assert ResidencyState.ON_GPU != ResidencyState.ON_CPU
    assert ResidencyState.LOADING != ResidencyState.EVICTING
    assert {
        ResidencyState.ON_GPU,
        ResidencyState.ON_CPU,
        ResidencyState.LOADING,
        ResidencyState.EVICTING,
    } == set(ResidencyState)


def test_initial_state_is_on_gpu():
    """A newly observed block is by definition just written -> ON_GPU."""
    from vllm.v1.attention.backends.quest.cache.residency import (
        BlockResidency,
        ResidencyState,
    )

    r = BlockResidency(num_layers=2, max_blocks=8)
    r.mark_on_gpu(layer_idx=1, block_id=3)
    assert r.state(1, 3) == ResidencyState.ON_GPU


def test_evict_then_complete_d2h():
    from vllm.v1.attention.backends.quest.cache.residency import (
        BlockResidency,
        ResidencyState,
    )

    r = BlockResidency(num_layers=1, max_blocks=8)
    r.mark_on_gpu(0, 5)
    r.begin_evict(0, 5)
    assert r.state(0, 5) == ResidencyState.EVICTING
    r.complete_evict(0, 5)
    assert r.state(0, 5) == ResidencyState.ON_CPU


def test_load_then_complete_h2d():
    from vllm.v1.attention.backends.quest.cache.residency import (
        BlockResidency,
        ResidencyState,
    )

    r = BlockResidency(num_layers=1, max_blocks=8)
    r.mark_on_gpu(0, 5)
    r.begin_evict(0, 5)
    r.complete_evict(0, 5)
    r.begin_load(0, 5)
    assert r.state(0, 5) == ResidencyState.LOADING
    r.complete_load(0, 5)
    assert r.state(0, 5) == ResidencyState.ON_GPU


def test_invalid_transition_raises():
    from vllm.v1.attention.backends.quest.cache.residency import (
        BlockResidency,
    )

    r = BlockResidency(num_layers=1, max_blocks=8)
    r.mark_on_gpu(0, 5)
    with pytest.raises(ValueError, match="cannot complete_load"):
        r.complete_load(0, 5)
    with pytest.raises(ValueError, match="cannot complete_evict"):
        r.complete_evict(0, 5)


def test_is_on_gpu_mask():
    """Vectorized GPU residency check used by selection to filter."""
    import torch
    from vllm.v1.attention.backends.quest.cache.residency import (
        BlockResidency,
    )

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    r = BlockResidency(num_layers=1, max_blocks=8)
    for i in [0, 1, 4, 7]:
        r.mark_on_gpu(0, i)
    r.begin_evict(0, 4)
    r.complete_evict(0, 4)

    ids = torch.tensor([0, 1, 4, 7], dtype=torch.int32, device="cuda")
    mask = r.is_on_gpu_mask(0, ids)
    assert mask.tolist() == [True, True, False, True]


def test_double_evict_is_idempotent():
    """Calling begin_evict twice is a no-op (eviction already in flight)."""
    from vllm.v1.attention.backends.quest.cache.residency import (
        BlockResidency,
        ResidencyState,
    )

    r = BlockResidency(num_layers=1, max_blocks=4)
    r.mark_on_gpu(0, 0)
    r.begin_evict(0, 0)
    r.begin_evict(0, 0)
    assert r.state(0, 0) == ResidencyState.EVICTING

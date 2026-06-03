# SPDX-License-Identifier: Apache-2.0


def test_nvtx_range_disabled_is_noop():
    # Import inside the test body (not module top-level) so pytest collection
    # does NOT load any quest module into sys.modules for the whole session.
    # A module-level import would trip the e2e default-path zero-impact guard
    # (test_default_path_does_not_import_quest_modules), which asserts no
    # `vllm.v1.attention.backends.quest.*` module is present. Every other quest
    # test file follows this same deferred-import convention.
    from vllm.v1.attention.backends.quest.impl_helpers import _nvtx_range

    with _nvtx_range("x", enabled=False):
        pass  # must not import torch / call cuda; just yields


def test_nvtx_range_enabled_yields():
    from vllm.v1.attention.backends.quest.impl_helpers import _nvtx_range

    # enabled path imports torch.cuda.nvtx; on a CUDA box this is safe.
    import torch
    if not torch.cuda.is_available():
        return
    with _nvtx_range("x", enabled=True):
        pass

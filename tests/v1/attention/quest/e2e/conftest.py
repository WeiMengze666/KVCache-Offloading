# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E2E fixtures for Phase E.1 — load real models, run through vLLM engine.

All fixtures here are gated behind `@pytest.mark.real_model`. The marker is
registered in pyproject.toml so default `pytest` runs skip these tests.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

QUEST_E2E_MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
MIN_GPU_MEMORY_GIB = 40


@pytest.fixture(scope="session")
def quest_e2e_model_id() -> str:
    return QUEST_E2E_MODEL_ID


def _hf_cache_has(model_id: str) -> bool:
    """Return True if the HF cache already contains snapshots for `model_id`."""
    repo_dir_name = "models--" + model_id.replace("/", "--")
    cache_root = (
        Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
        / "hub"
        / repo_dir_name
    )
    if not cache_root.is_dir():
        return False
    snapshots = cache_root / "snapshots"
    return snapshots.is_dir() and any(snapshots.iterdir())


@pytest.fixture(scope="session", autouse=True)
def _real_model_e2e_gates(request, quest_e2e_model_id):
    """Skip the entire e2e/ subtree unless GPU + HF cache prerequisites pass.

    Autouse so it runs once per session before any other fixture, but only
    when at least one collected test has the `real_model` mark. When pytest
    collected nothing under `-m real_model`, this fixture is still computed
    but its checks are cheap and harmless.
    """
    if not torch.cuda.is_available():
        pytest.skip("real_model e2e requires CUDA")
    total_bytes = torch.cuda.get_device_properties(0).total_memory
    total_gib = total_bytes / (1024**3)
    if total_gib < MIN_GPU_MEMORY_GIB:
        pytest.skip(
            f"real_model e2e needs >={MIN_GPU_MEMORY_GIB} GiB GPU, "
            f"have {total_gib:.1f} GiB"
        )
    if not _hf_cache_has(quest_e2e_model_id):
        pytest.skip(
            f"HF cache missing {quest_e2e_model_id}. "
            f"Run: huggingface-cli download {quest_e2e_model_id}"
        )

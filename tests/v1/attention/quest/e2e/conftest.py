# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E2E fixtures for Phase E.1 — load real models, run through vLLM engine.

All fixtures here are gated behind `@pytest.mark.real_model`. The marker is
registered in pyproject.toml so default `pytest` runs skip these tests.
"""

from __future__ import annotations

import pytest

QUEST_E2E_MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
MIN_GPU_MEMORY_GIB = 40


@pytest.fixture(scope="session")
def quest_e2e_model_id() -> str:
    return QUEST_E2E_MODEL_ID

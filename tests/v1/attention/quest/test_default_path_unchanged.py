# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Default vLLM path is not affected by Quest backend code being on disk."""

from __future__ import annotations

import sys


def test_quest_packages_not_imported_by_vllm_attention_module():
    # Importing the vLLM attention machinery must not eagerly drag in any
    # quest module. Other tests in this package may have already imported
    # quest submodules, so snapshot+restore sys.modules to avoid leaving
    # later tests with dangling module references.
    saved = {
        name: mod
        for name, mod in sys.modules.items()
        if name.startswith("vllm.v1.attention.backends.quest")
    }
    for name in saved:
        del sys.modules[name]
    try:
        import vllm.v1.attention.selector  # noqa: F401
        import vllm.v1.attention.backends.flash_attn  # noqa: F401
        import vllm.v1.attention.backends.registry  # noqa: F401

        bad = [
            m for m in sys.modules if m.startswith("vllm.v1.attention.backends.quest")
        ]
        assert bad == [], (
            f"Quest packages were eagerly imported by vLLM core: {bad}. "
            "The Quest backend must remain opt-in."
        )
    finally:
        # Restore quest modules so subsequent tests see the same module
        # objects they already imported.
        for name, mod in saved.items():
            sys.modules[name] = mod


def test_vllm_config_can_be_built_without_quest_config():
    from vllm.config import VllmConfig

    cfg = VllmConfig.__new__(VllmConfig)
    # Just make sure the field has a None default and is not required.
    import dataclasses

    field = next(f for f in dataclasses.fields(VllmConfig) if f.name == "quest_config")
    # default OR default_factory must produce None.
    if field.default is not dataclasses.MISSING:
        assert field.default is None
    elif field.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
        assert field.default_factory() is None
    else:
        raise AssertionError("quest_config has neither default nor default_factory")


def test_model_runner_does_not_import_quest_packages_when_disabled():
    """Touching vllm.v1.worker.gpu.model_runner should not pull in
    vllm.v1.attention.backends.quest. The bind_runtime call site uses a
    lazy import gated on quest_config.enabled."""
    import importlib

    # Make sure quest backend is NOT already loaded by a previous test.
    for mod in list(sys.modules):
        if mod.startswith("vllm.v1.attention.backends.quest"):
            del sys.modules[mod]

    importlib.import_module("vllm.v1.worker.gpu.model_runner")

    leaked = [
        m for m in sys.modules if m.startswith("vllm.v1.attention.backends.quest")
    ]
    assert leaked == [], f"quest packages leaked into model_runner import: {leaked}"


def test_default_path_does_not_import_phase_d_modules():
    """Phase D adds quest_selection_dispatch + quest_selection_cuda. They
    must NOT be imported on the default path (quest_config disabled)."""
    import subprocess
    import sys

    code = (
        "import sys\n"
        "import vllm.v1.attention\n"
        "import vllm.v1.worker.gpu.model_runner\n"
        "leaked = [m for m in sys.modules "
        "if 'quest_selection_dispatch' in m "
        "or 'quest_selection_cuda' in m]\n"
        "assert not leaked, 'Phase D modules leaked on default path: ' "
        "+ str(leaked)\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, f"stdout={result.stdout!r}\nstderr={result.stderr!r}"
    assert "OK" in result.stdout

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
        import vllm.v1.attention.backends.flash_attn  # noqa: F401
        import vllm.v1.attention.backends.registry  # noqa: F401
        import vllm.v1.attention.selector  # noqa: F401

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
    # Just make sure the field has a None default and is not required.
    import dataclasses

    from vllm.config import VllmConfig

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


def test_vllm_config_can_be_built_without_arkvale_config():
    """Mirror of test_vllm_config_can_be_built_without_quest_config for ArkVale.

    ArkValeConfig must be optional on VllmConfig with a None default,
    matching the pattern established for QuestConfig.
    """
    import dataclasses

    from vllm.config import VllmConfig

    field = next(
        f for f in dataclasses.fields(VllmConfig) if f.name == "arkvale_config"
    )
    if field.default is not dataclasses.MISSING:
        assert field.default is None, "arkvale_config default must be None"
    elif field.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
        assert field.default_factory() is None, (
            "arkvale_config factory must produce None"
        )
    else:
        raise AssertionError("arkvale_config has neither default nor default_factory")


def test_quest_packages_not_imported_when_arkvale_explicitly_disabled():
    """Smoke test: explicitly setting enable_arkvale_sparse_offload=False
    (the default) must not cause the Quest backend to be loaded.

    This locks in that the Task-9 selector + gpu_model_runner ArkVale
    code paths don't introduce a side-effect import.

    Note: vllm.config.arkvale IS expected to be in sys.modules after
    'import vllm.config' — it's a config dataclass, not a backend.
    The invariant here is on backend modules under
    vllm.v1.attention.backends.quest.
    """
    saved = {
        name: mod
        for name, mod in sys.modules.items()
        if name.startswith("vllm.v1.attention.backends.quest")
    }
    for name in saved:
        del sys.modules[name]
    try:
        # Import the config dataclass (arkvale_config field default is None
        # / disabled) — this must not pull in any Quest backend module.
        from vllm.config import ArkValeConfig, VllmConfig  # noqa: F401

        assert not ArkValeConfig().enabled, "ArkValeConfig() must default to disabled"

        import vllm.v1.attention.backends.flash_attn  # noqa: F401
        import vllm.v1.attention.backends.registry  # noqa: F401
        import vllm.v1.attention.selector  # noqa: F401

        bad = [
            m for m in sys.modules if m.startswith("vllm.v1.attention.backends.quest")
        ]
        assert bad == [], (
            f"Quest backend packages leaked when ArkVale is explicitly "
            f"disabled (arkvale_config=None): {bad}. "
            "The Quest backend must remain opt-in."
        )
    finally:
        for name, mod in saved.items():
            sys.modules[name] = mod


def test_model_runner_does_not_import_quest_when_arkvale_disabled():
    """Mirror of test_model_runner_does_not_import_quest_packages_when_disabled
    with ArkVale explicitly disabled.

    Touching vllm.v1.worker.gpu.model_runner with arkvale_config=None must
    not pull in vllm.v1.attention.backends.quest.* as a side effect.
    """
    import importlib

    for mod in list(sys.modules):
        if mod.startswith("vllm.v1.attention.backends.quest"):
            del sys.modules[mod]

    # Import the ArkVale config to confirm it's disabled by default.
    from vllm.config import ArkValeConfig  # noqa: F401

    assert not ArkValeConfig().enabled

    importlib.import_module("vllm.v1.worker.gpu.model_runner")

    leaked = [
        m for m in sys.modules if m.startswith("vllm.v1.attention.backends.quest")
    ]
    assert leaked == [], (
        f"Quest packages leaked into model_runner import when ArkVale "
        f"is disabled: {leaked}"
    )

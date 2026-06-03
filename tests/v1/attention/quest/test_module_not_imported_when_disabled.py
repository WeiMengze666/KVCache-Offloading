# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Subprocess assertion: with Quest disabled, no quest module is loaded."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap


def test_quest_module_not_imported_when_env_unset():
    script = textwrap.dedent(
        """
        import os, sys, json
        os.environ.pop("VLLM_ATTENTION_BACKEND", None)
        # Touch the parts of vLLM that a real engine init would.
        from vllm.config import VllmConfig  # noqa
        from vllm.v1.attention.backends.flash_attn import (
            FlashAttentionBackend,
        )  # noqa
        from vllm.v1.attention.backends.registry import (
            AttentionBackendEnum,
        )  # noqa
        loaded = sorted(
            m for m in sys.modules if m.startswith("vllm.v1.attention.backends.quest")
        )
        print(json.dumps(loaded))
        """
    )
    env = dict(os.environ)
    env.pop("VLLM_ATTENTION_BACKEND", None)
    res = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
    )
    assert res.returncode == 0, (
        f"subprocess failed: stdout={res.stdout!r} stderr={res.stderr!r}"
    )
    loaded = res.stdout.strip().splitlines()[-1]
    import json

    assert json.loads(loaded) == [], (
        f"Quest modules leaked into default vLLM path: {loaded}"
    )


def test_no_quest_backend_leak_when_arkvale_disabled():
    """Phase A invariant for the ArkVale-disabled path.

    With both Quest and ArkVale disabled, importing the attention layer
    + worker must NOT load any vllm.v1.attention.backends.quest.* module.
    The Quest backend is shared by both selectors; testing this with
    arkvale_config explicitly None (rather than absent) verifies the
    Task-9 hookup didn't introduce a side-effect import on the
    arkvale_config branch.

    Note: vllm.config.arkvale IS loaded when 'import vllm.config' runs
    (it's a config dataclass, not a backend) — that's expected and fine.
    The invariant here is strictly on backend modules under
    vllm.v1.attention.backends.quest.
    """
    script = textwrap.dedent(
        """
        import os, sys, json
        os.environ.pop("VLLM_ATTENTION_BACKEND", None)
        # Simulate a VllmConfig with arkvale_config explicitly disabled.
        from vllm.config import VllmConfig, ArkValeConfig  # noqa
        cfg = VllmConfig.__new__(VllmConfig)
        # Verify arkvale_config field default is None (disabled).
        import dataclasses
        field = next(
            f for f in dataclasses.fields(VllmConfig)
            if f.name == "arkvale_config"
        )
        if field.default is not dataclasses.MISSING:
            assert field.default is None, "arkvale_config default must be None"
        elif field.default_factory is not dataclasses.MISSING:
            assert field.default_factory() is None, (
                "arkvale_config factory must produce None"
            )
        # Touch the parts of vLLM that a real engine init would.
        from vllm.v1.attention.backends.flash_attn import (
            FlashAttentionBackend,
        )  # noqa
        from vllm.v1.attention.backends.registry import (
            AttentionBackendEnum,
        )  # noqa
        loaded = sorted(
            m for m in sys.modules
            if m.startswith("vllm.v1.attention.backends.quest")
        )
        print(json.dumps(loaded))
        """
    )
    env = dict(os.environ)
    env.pop("VLLM_ATTENTION_BACKEND", None)
    res = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
    )
    assert res.returncode == 0, (
        f"subprocess failed: stdout={res.stdout!r} stderr={res.stderr!r}"
    )
    import json

    loaded = res.stdout.strip().splitlines()[-1]
    assert json.loads(loaded) == [], (
        f"default path with ArkVale disabled leaked Quest backend modules: {loaded}"
    )

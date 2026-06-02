# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EngineArgs hookup for ArkVale + mutex with Quest."""

from __future__ import annotations

import argparse
import json
import os
import tempfile


def _parse(argv):
    """Helper: feed argv to EngineArgs.from_cli_args and return EngineArgs."""
    from vllm.engine.arg_utils import EngineArgs

    parser = EngineArgs.add_cli_args(argparse.ArgumentParser())
    ns = parser.parse_args(argv)
    return EngineArgs.from_cli_args(ns)


def _write_cfg_json(d: dict) -> str:
    """Write a config dict to a temp JSON file and return the path."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(d, f)
        return f.name


def test_arkvale_config_loaded_from_json(tmp_path):
    from vllm.config.arkvale import ArkValeConfig
    from vllm.engine.arg_utils import _arkvale_config_from_args

    cfg = ArkValeConfig(
        enabled=True, top_k=8, block_size=4, gpu_cache_blocks_per_seq=16
    )
    cfg.validate()
    p = tmp_path / "arkvale.json"
    p.write_text(json.dumps(cfg.to_dict()))
    args = _parse(
        [
            "--model",
            "facebook/opt-125m",
            "--enable-arkvale-sparse-offload",
            "--arkvale-config",
            str(p),
        ]
    )
    result = _arkvale_config_from_args(args)
    assert result is not None
    assert result.enabled is True
    assert result.top_k == 8
    assert result.digest_mode == "arkvale_cuboid_mean"


def test_arkvale_cli_overrides_top_k(tmp_path):
    from vllm.config.arkvale import ArkValeConfig
    from vllm.engine.arg_utils import _arkvale_config_from_args

    cfg = ArkValeConfig(
        enabled=True, top_k=8, block_size=4, gpu_cache_blocks_per_seq=64
    )
    p = tmp_path / "arkvale.json"
    p.write_text(json.dumps(cfg.to_dict()))
    args = _parse(
        [
            "--model",
            "facebook/opt-125m",
            "--enable-arkvale-sparse-offload",
            "--arkvale-config",
            str(p),
            "--arkvale-top-k",
            "32",
        ]
    )
    result = _arkvale_config_from_args(args)
    assert result is not None
    assert result.top_k == 32


def test_arkvale_disabled_means_no_config_attached():
    from vllm.engine.arg_utils import _arkvale_config_from_args

    args = _parse(["--model", "facebook/opt-125m"])
    assert _arkvale_config_from_args(args) is None


def test_quest_and_arkvale_mutually_exclusive():
    """Mutex check fires inside _resolve_sparse_offload_configs (used by
    create_engine_config). Calling the helper directly avoids the heavy
    model-load path while still exercising the production mutex."""
    import pytest

    from vllm.config.arkvale import ArkValeConfig
    from vllm.config.quest import QuestConfig
    from vllm.engine.arg_utils import EngineArgs

    qcfg = QuestConfig(enabled=True, top_k=8, gpu_cache_blocks_per_seq=16, block_size=4)
    acfg = ArkValeConfig(
        enabled=True, top_k=8, gpu_cache_blocks_per_seq=16, block_size=4
    )
    qpath = _write_cfg_json(qcfg.to_dict())
    apath = _write_cfg_json(acfg.to_dict())
    try:
        args = EngineArgs(
            model="facebook/opt-125m",
            enable_quest_sparse_offload=True,
            quest_config=qpath,
            enable_arkvale_sparse_offload=True,
            arkvale_config=apath,
        )
        with pytest.raises(ValueError, match="mutually exclusive"):
            args._resolve_sparse_offload_configs()
    finally:
        os.unlink(qpath)
        os.unlink(apath)


def test_resolve_sparse_offload_configs_quest_only():
    from vllm.config.quest import QuestConfig
    from vllm.engine.arg_utils import EngineArgs

    qcfg = QuestConfig(enabled=True, top_k=8, gpu_cache_blocks_per_seq=16, block_size=4)
    qpath = _write_cfg_json(qcfg.to_dict())
    try:
        args = EngineArgs(
            model="facebook/opt-125m",
            enable_quest_sparse_offload=True,
            quest_config=qpath,
        )
        quest_out, ark_out = args._resolve_sparse_offload_configs()
        assert quest_out is not None and quest_out.enabled
        assert ark_out is None or not ark_out.enabled
    finally:
        os.unlink(qpath)


def test_resolve_sparse_offload_configs_arkvale_only():
    from vllm.config.arkvale import ArkValeConfig
    from vllm.engine.arg_utils import EngineArgs

    acfg = ArkValeConfig(
        enabled=True, top_k=8, gpu_cache_blocks_per_seq=16, block_size=4
    )
    apath = _write_cfg_json(acfg.to_dict())
    try:
        args = EngineArgs(
            model="facebook/opt-125m",
            enable_arkvale_sparse_offload=True,
            arkvale_config=apath,
        )
        quest_out, ark_out = args._resolve_sparse_offload_configs()
        assert ark_out is not None and ark_out.enabled
        assert quest_out is None or not quest_out.enabled
    finally:
        os.unlink(apath)


def test_resolve_sparse_offload_configs_neither():
    from vllm.engine.arg_utils import EngineArgs

    args = EngineArgs(model="facebook/opt-125m")
    quest_out, ark_out = args._resolve_sparse_offload_configs()
    assert quest_out is None and ark_out is None

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EngineArgs hookup for ArkVale + mutex with Quest."""

from __future__ import annotations

import argparse
import json


def _parse(argv):
    """Helper: feed argv to EngineArgs.from_cli_args and return EngineArgs."""
    from vllm.engine.arg_utils import EngineArgs

    parser = EngineArgs.add_cli_args(argparse.ArgumentParser())
    ns = parser.parse_args(argv)
    return EngineArgs.from_cli_args(ns)


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


def test_quest_and_arkvale_mutually_exclusive(tmp_path):
    import pytest

    from vllm.config.arkvale import ArkValeConfig
    from vllm.config.quest import QuestConfig
    from vllm.engine.arg_utils import (
        _arkvale_config_from_args,
        _quest_config_from_args,
    )

    qcfg = QuestConfig(enabled=True, top_k=8, gpu_cache_blocks_per_seq=16, block_size=4)
    acfg = ArkValeConfig(
        enabled=True, top_k=8, gpu_cache_blocks_per_seq=16, block_size=4
    )
    qp = tmp_path / "quest.json"
    ap = tmp_path / "arkvale.json"
    qp.write_text(json.dumps(qcfg.to_dict()))
    ap.write_text(json.dumps(acfg.to_dict()))

    args = _parse(
        [
            "--model",
            "facebook/opt-125m",
            "--enable-quest-sparse-offload",
            "--quest-config",
            str(qp),
            "--enable-arkvale-sparse-offload",
            "--arkvale-config",
            str(ap),
        ]
    )

    quest_cfg = _quest_config_from_args(args)
    arkvale_cfg = _arkvale_config_from_args(args)

    quest_on = quest_cfg is not None and quest_cfg.enabled
    arkvale_on = arkvale_cfg is not None and arkvale_cfg.enabled

    with pytest.raises(ValueError, match="mutually exclusive"):
        if quest_on and arkvale_on:
            raise ValueError(
                "enable_quest_sparse_offload and "
                "enable_arkvale_sparse_offload are mutually exclusive; "
                "pick exactly one."
            )

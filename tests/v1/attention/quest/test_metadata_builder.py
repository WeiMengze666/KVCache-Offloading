# SPDX-License-Identifier: Apache-2.0
"""QuestAttentionMetadata + builder."""
from __future__ import annotations

import pytest
import torch


def test_quest_metadata_subclasses_fa_metadata():
    from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata
    from vllm.v1.attention.backends.quest.metadata import (
        QuestAttentionMetadata,
    )

    # In Phase B QuestAttentionMetadata is its own dataclass that inherits
    # all fields from FlashAttentionMetadata and adds Quest-specific ones.
    assert issubclass(QuestAttentionMetadata, FlashAttentionMetadata)


def test_quest_metadata_extra_fields():
    from vllm.v1.attention.backends.quest.metadata import (
        QuestAttentionMetadata,
    )

    fields = {f for f in QuestAttentionMetadata.__dataclass_fields__}
    assert "sparse_block_table" in fields
    assert "quest_layer_indices" in fields
    assert "is_full_kv_layer" in fields


def test_quest_metadata_builder_passthrough(monkeypatch):
    """Builder emits a metadata whose FA fields equal what the FA builder
    would have produced (delegation), plus default-empty Quest fields when
    selection has not yet been computed.
    """
    from vllm.v1.attention.backends.quest.metadata import (
        QuestMetadataBuilder, QuestAttentionMetadata,
    )

    # We just check the delegation contract on a fake FA builder.
    class _FakeFA:
        def build(self, *a, **k):
            from vllm.v1.attention.backends.flash_attn import (
                FlashAttentionMetadata,
            )
            return FlashAttentionMetadata(
                num_actual_tokens=4,
                max_query_len=1,
                query_start_loc=torch.tensor([0, 1, 2, 3, 4],
                                              dtype=torch.int32),
                max_seq_len=4,
                seq_lens=torch.tensor([4], dtype=torch.int32),
                block_table=torch.tensor([[0, 1]], dtype=torch.int32),
                slot_mapping=torch.tensor([0, 1, 2, 3], dtype=torch.int64),
                use_cascade=False,
                common_prefix_len=0,
                cu_prefix_query_lens=None,
                prefix_kv_lens=None,
                suffix_kv_lens=None,
            )

    builder = QuestMetadataBuilder.__new__(QuestMetadataBuilder)
    builder._fa_builder = _FakeFA()
    builder._quest_layer_indices = torch.tensor([-1, -1, 0, 1, 2],
                                                 dtype=torch.int32)
    md = builder.build_for_test()
    assert isinstance(md, QuestAttentionMetadata)
    assert md.num_actual_tokens == 4
    assert md.sparse_block_table is None  # populated by impl pre-attn
    assert md.is_full_kv_layer.tolist() == [True, True, False, False, False]
    assert md.quest_layer_indices.tolist() == [-1, -1, 0, 1, 2]

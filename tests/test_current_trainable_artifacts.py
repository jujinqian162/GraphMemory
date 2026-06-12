from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from graph_memory.models.dense_finetune.metadata import (
    DENSE_FT_METADATA_FILENAME,
    DenseFinetuneModelMetadata,
    DenseFinetuneSelectionMetadata,
    load_dense_ft_model_metadata,
    write_dense_ft_model_metadata,
)
from graph_memory.models.graph_retriever.checkpoint import (
    load_rgcn_checkpoint,
    save_rgcn_checkpoint,
)
from graph_memory.models.graph_retriever.config.records import (
    NodeFeatureConfig,
    RgcnModelConfig,
    RgcnTrainingConfig,
)
from graph_memory.validation import ContractValidationError


def _rgcn_model_config() -> RgcnModelConfig:
    return RgcnModelConfig(
        method_name="dense_rgcn_graph_retriever",
        encoder_model="fake-encoder",
        encoder_dim=4,
        query_prefix="query: ",
        passage_prefix="passage: ",
        encoder_batch_size=11,
        hidden_dim=8,
        num_layers=1,
        dropout=0.0,
        feature_config=NodeFeatureConfig(),
        relation_vocab=("query_overlap_forward",),
        graph_encoder_type="rgcn",
        message_transform_type="typed",
        edge_weight_policy="artifact",
        enabled_edge_types=("query_overlap",),
        ablation_name="full_rgcn",
    )


def test_rgcn_checkpoint_has_no_version_and_rejects_versioned_payload(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "best.pt"
    model = torch.nn.Linear(4, 1)
    save_rgcn_checkpoint(
        checkpoint_path,
        method_name="dense_rgcn_graph_retriever",
        model=model,
        optimizer_state_dict={},
        scheduler_state_dict={},
        epoch=1,
        global_step=2,
        best_dev_metric=0.5,
        model_config=_rgcn_model_config(),
        training_config=RgcnTrainingConfig(),
    )

    checkpoint = load_rgcn_checkpoint(
        checkpoint_path,
        expected_method="dense_rgcn_graph_retriever",
    )
    assert "checkpoint_version" not in checkpoint.payload
    assert checkpoint.model_config == _rgcn_model_config()

    versioned_path = tmp_path / "versioned.pt"
    versioned_payload = dict(checkpoint.payload)
    versioned_payload["checkpoint_version"] = 1
    torch.save(versioned_payload, versioned_path)

    with pytest.raises(ContractValidationError, match="unknown fields.*checkpoint_version"):
        load_rgcn_checkpoint(
            versioned_path,
            expected_method="dense_rgcn_graph_retriever",
        )


def test_dense_ft_metadata_round_trip_has_no_version_and_records_device(tmp_path: Path) -> None:
    model_dir = tmp_path / "best_model"
    metadata = DenseFinetuneModelMetadata(
        base_model="fake-base",
        query_prefix="Q: ",
        passage_prefix="P: ",
        batch_size=7,
        device="cuda",
        selection=DenseFinetuneSelectionMetadata(
            selected_metric="eval_dev_cos_sim_map@100",
            higher_is_better=True,
        ),
    )

    metadata_path = write_dense_ft_model_metadata(model_dir=model_dir, metadata=metadata)
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert metadata_path == model_dir / DENSE_FT_METADATA_FILENAME
    assert "schema_version" not in payload
    assert load_dense_ft_model_metadata(model_dir) == metadata


def test_dense_ft_metadata_rejects_versioned_payload(tmp_path: Path) -> None:
    model_dir = tmp_path / "best_model"
    model_dir.mkdir()
    (model_dir / DENSE_FT_METADATA_FILENAME).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "method": "dense_ft",
                "base_model": "fake-base",
                "query_prefix": "Q: ",
                "passage_prefix": "P: ",
                "batch_size": 7,
                "device": "cpu",
                "selection": {
                    "selected_metric": "eval_dev_cos_sim_map@100",
                    "higher_is_better": True,
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsupported fields.*schema_version"):
        load_dense_ft_model_metadata(model_dir)

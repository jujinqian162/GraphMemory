from __future__ import annotations

from pathlib import Path
from typing import Literal

from graph_memory.contracts.model import DomainModel, NonEmptyStr, PositiveInt
from graph_memory.infrastructure.io import read_json, write_json
from graph_memory.retrieval.methods.ids import DenseCandidateView, RetrievalMethodId

CROSS_ENCODER_METADATA_FILENAME = "cross_encoder_model_config.json"


class CrossEncoderModelMetadata(DomainModel):
    base_model: NonEmptyStr
    max_length: PositiveInt
    train_batch_size: PositiveInt
    eval_batch_size: PositiveInt
    device: NonEmptyStr
    selected_metric: Literal["dev_recall_at_5"] = "dev_recall_at_5"
    variant: DenseCandidateView = "flat"
    method: Literal[RetrievalMethodId.CROSS_ENCODER] = RetrievalMethodId.CROSS_ENCODER


def write_cross_encoder_model_metadata(
    *, model_dir: Path, metadata: CrossEncoderModelMetadata
) -> Path:
    path = model_dir / CROSS_ENCODER_METADATA_FILENAME
    write_json(path, metadata.model_dump(mode="json"))
    return path


def load_cross_encoder_model_metadata(model_dir: Path) -> CrossEncoderModelMetadata:
    path = model_dir / CROSS_ENCODER_METADATA_FILENAME
    if not path.is_file():
        raise ValueError(
            f"Missing {CROSS_ENCODER_METADATA_FILENAME} for cross_encoder model: {model_dir}"
        )
    return CrossEncoderModelMetadata.model_validate(read_json(path))


__all__ = [
    "CROSS_ENCODER_METADATA_FILENAME",
    "CrossEncoderModelMetadata",
    "load_cross_encoder_model_metadata",
    "write_cross_encoder_model_metadata",
]

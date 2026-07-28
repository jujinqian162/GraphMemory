from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import StrictBool

from graph_memory.contracts.model import DomainModel, NonEmptyStr, PositiveInt
from graph_memory.infrastructure.io import read_json, write_json
from graph_memory.retrieval.methods.ids import RetrievalMethodId

DENSE_FT_METADATA_FILENAME = "dense_ft_model_config.json"


class DenseFinetuneSelectionMetadata(DomainModel):
    selected_metric: NonEmptyStr
    higher_is_better: StrictBool


class DenseFinetuneModelMetadata(DomainModel):
    base_model: NonEmptyStr
    query_prefix: str
    passage_prefix: str
    batch_size: PositiveInt
    device: NonEmptyStr
    selection: DenseFinetuneSelectionMetadata
    method: Literal[RetrievalMethodId.DENSE_FT] = RetrievalMethodId.DENSE_FT


def write_dense_ft_model_metadata(
    *,
    model_dir: Path,
    metadata: DenseFinetuneModelMetadata,
) -> Path:
    metadata_path = model_dir / DENSE_FT_METADATA_FILENAME
    write_json(metadata_path, metadata.model_dump(mode="json"))
    return metadata_path


def load_dense_ft_model_metadata(model_dir: Path) -> DenseFinetuneModelMetadata:
    metadata_path = model_dir / DENSE_FT_METADATA_FILENAME
    if not metadata_path.is_file():
        raise ValueError(
            f"Missing {DENSE_FT_METADATA_FILENAME} for dense_ft model: {model_dir}"
        )
    return DenseFinetuneModelMetadata.model_validate(read_json(metadata_path))


__all__ = [
    "DENSE_FT_METADATA_FILENAME",
    "DenseFinetuneModelMetadata",
    "DenseFinetuneSelectionMetadata",
    "load_dense_ft_model_metadata",
    "write_dense_ft_model_metadata",
]

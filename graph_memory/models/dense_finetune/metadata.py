from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from graph_memory.config.converter import ConfigConverter
from graph_memory.contracts.common import JsonObject
from graph_memory.infrastructure.io import read_json, write_json
from graph_memory.registry.retrieval import RetrievalMethodId

DENSE_FT_METADATA_FILENAME = "dense_ft_model_config.json"


@dataclass(frozen=True)
class DenseFinetuneSelectionMetadata:
    selected_metric: str
    higher_is_better: bool


@dataclass(frozen=True)
class DenseFinetuneModelMetadata:
    base_model: str
    query_prefix: str
    passage_prefix: str
    batch_size: int
    device: str
    selection: DenseFinetuneSelectionMetadata
    method: Literal[RetrievalMethodId.DENSE_FT] = RetrievalMethodId.DENSE_FT


def write_dense_ft_model_metadata(
    *,
    model_dir: Path,
    metadata: DenseFinetuneModelMetadata,
) -> Path:
    metadata_path = model_dir / DENSE_FT_METADATA_FILENAME
    payload = ConfigConverter().unstructure(metadata)
    write_json(metadata_path, cast(JsonObject, payload))
    return metadata_path


def load_dense_ft_model_metadata(model_dir: Path) -> DenseFinetuneModelMetadata:
    metadata_path = model_dir / DENSE_FT_METADATA_FILENAME
    if not metadata_path.is_file():
        raise ValueError(f"Missing {DENSE_FT_METADATA_FILENAME} for dense_ft model: {model_dir}")
    payload = read_json(metadata_path)
    try:
        return cast(
            DenseFinetuneModelMetadata,
            ConfigConverter().structure(payload, DenseFinetuneModelMetadata),
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid {DENSE_FT_METADATA_FILENAME} for dense_ft model {model_dir}: {error}") from error


__all__ = [
    "DENSE_FT_METADATA_FILENAME",
    "DenseFinetuneModelMetadata",
    "DenseFinetuneSelectionMetadata",
    "load_dense_ft_model_metadata",
    "write_dense_ft_model_metadata",
]

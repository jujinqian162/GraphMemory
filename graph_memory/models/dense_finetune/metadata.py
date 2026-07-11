from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from pydantic import TypeAdapter, ValidationError
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
    payload = TypeAdapter(DenseFinetuneModelMetadata).dump_python(metadata, mode="json")
    write_json(metadata_path, cast(JsonObject, payload))
    return metadata_path


def load_dense_ft_model_metadata(model_dir: Path) -> DenseFinetuneModelMetadata:
    metadata_path = model_dir / DENSE_FT_METADATA_FILENAME
    if not metadata_path.is_file():
        raise ValueError(
            f"Missing {DENSE_FT_METADATA_FILENAME} for dense_ft model: {model_dir}"
        )
    payload = read_json(metadata_path)
    try:
        _reject_unsupported_fields(payload)
        return TypeAdapter(DenseFinetuneModelMetadata).validate_python(payload)
    except (TypeError, ValueError, ValidationError) as error:
        raise ValueError(
            f"Invalid {DENSE_FT_METADATA_FILENAME} for dense_ft model {model_dir}: {error}"
        ) from error


def _reject_unsupported_fields(payload: object) -> None:
    if not isinstance(payload, dict):
        return
    supported = {
        "method",
        "base_model",
        "query_prefix",
        "passage_prefix",
        "batch_size",
        "device",
        "selection",
    }
    unsupported = sorted(set(payload) - supported)
    if unsupported:
        raise ValueError(f"unsupported fields: {', '.join(unsupported)}")
    selection = payload.get("selection")
    if isinstance(selection, dict):
        unsupported_selection = sorted(
            set(selection) - {"selected_metric", "higher_is_better"}
        )
        if unsupported_selection:
            raise ValueError(
                "unsupported selection fields: " + ", ".join(unsupported_selection)
            )


__all__ = [
    "DENSE_FT_METADATA_FILENAME",
    "DenseFinetuneModelMetadata",
    "DenseFinetuneSelectionMetadata",
    "load_dense_ft_model_metadata",
    "write_dense_ft_model_metadata",
]

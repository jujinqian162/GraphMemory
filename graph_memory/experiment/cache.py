from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from prefect.cache_policies import CachePolicy
from prefect.utilities.hashing import hash_objects
from pydantic import BaseModel

from graph_memory.experiment.artifacts import (
    DirectorySourceRef,
    FileSourceRef,
    ProcessedArtifactRef,
    RevisionSourceRef,
)


# These fields choose where/how work runs, not what scientific result it computes.
# They are removed at every nesting level because device currently appears both as
# a task argument and inside several Pydantic stage configs.
_RUNTIME_ONLY_FIELDS = frozenset(
    {"device", "enable_gpupool", "workers", "chunk_size"}
)


class ScientificInputs(CachePolicy):
    """Hash scientific identities instead of runtime placement and artifact paths."""

    def compute_key(
        self,
        task_ctx: Any,
        inputs: dict[str, Any],
        flow_parameters: dict[str, Any],
        **kwargs: Any,
    ) -> str | None:
        del task_ctx, flow_parameters, kwargs
        if not inputs:
            return None
        return hash_objects(_scientific_value(inputs), raise_on_failure=True)


def _scientific_value(value: Any) -> Any:
    if isinstance(value, ProcessedArtifactRef):
        return {
            "__scientific_type__": "processed_artifact",
            "kind": value.kind.value,
            "digest": value.digest,
        }
    if isinstance(value, (FileSourceRef, DirectorySourceRef)):
        return {
            "__scientific_type__": "content_source",
            "kind": value.kind,
            "digest": value.digest,
        }
    if isinstance(value, RevisionSourceRef):
        return {
            "__scientific_type__": "immutable_revision",
            "uri": value.uri,
            "revision": value.revision,
        }
    if isinstance(value, BaseModel):
        return _scientific_value(value.model_dump(mode="python"))
    if isinstance(value, Mapping):
        return {
            key: _scientific_value(item)
            for key, item in value.items()
            if key not in _RUNTIME_ONLY_FIELDS
        }
    if isinstance(value, tuple):
        return tuple(_scientific_value(item) for item in value)
    if isinstance(value, list):
        return [_scientific_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_scientific_value(item) for item in value), key=repr)
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_scientific_value(item) for item in value]
    return value


__all__ = ["ScientificInputs"]

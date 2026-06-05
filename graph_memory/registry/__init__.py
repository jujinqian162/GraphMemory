from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from graph_memory.registry.app import AppRegistry, Registry
    from graph_memory.registry.ids import StageId, StrEnum
    from graph_memory.registry.retrieval import RetrievalMethodMetadata
    from graph_memory.registry.specs import StageConfigSpec


def __getattr__(name: str) -> Any:
    if name in {"AppRegistry", "Registry"}:
        from graph_memory.registry.app import AppRegistry, Registry

        return {"AppRegistry": AppRegistry, "Registry": Registry}[name]
    if name in {"StageId", "StrEnum", "parse_closed_value"}:
        from graph_memory.registry.ids import StageId, StrEnum, parse_closed_value

        return {"StageId": StageId, "StrEnum": StrEnum, "parse_closed_value": parse_closed_value}[name]
    if name in {"RETRIEVAL_METHOD_METADATA", "RetrievalMethodMetadata"}:
        from graph_memory.registry.retrieval import RETRIEVAL_METHOD_METADATA, RetrievalMethodMetadata

        return {
            "RETRIEVAL_METHOD_METADATA": RETRIEVAL_METHOD_METADATA,
            "RetrievalMethodMetadata": RetrievalMethodMetadata,
        }[name]
    if name == "StageConfigSpec":
        from graph_memory.registry.specs import StageConfigSpec

        return StageConfigSpec
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "AppRegistry",
    "RETRIEVAL_METHOD_METADATA",
    "Registry",
    "RetrievalMethodMetadata",
    "StageConfigSpec",
    "StageId",
    "StrEnum",
    "parse_closed_value",
]

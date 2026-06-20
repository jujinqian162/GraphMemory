from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

from graph_memory.contracts.common import NodeId
from graph_memory.retrieval.requests import TemporalMemoryRankingRequest, TextCandidate


def importance_content_digest(request: TemporalMemoryRankingRequest) -> str:
    payload = {
        "items": [_semantic_item(candidate, request.metadata) for candidate in request.candidates],
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def task_node_ids(request: TemporalMemoryRankingRequest) -> list[NodeId]:
    return [candidate.item_id for candidate in request.candidates]


def _semantic_item(candidate: TextCandidate, request_metadata: Mapping[str, object]) -> dict[str, object]:
    metadata = dict(candidate.metadata)
    return {
        "id": candidate.item_id,
        "source": _optional_string(metadata, "source_ref") or _optional_string(metadata, "title"),
        "text": candidate.text,
        "position": _position_for(candidate.item_id, request_metadata),
    }


def _optional_string(metadata: Mapping[str, object], field_name: str) -> str | None:
    value = metadata.get(field_name)
    return value if isinstance(value, str) else None


def _position_for(item_id: str, request_metadata: Mapping[str, object]) -> int | None:
    raw_positions = request_metadata.get("position_by_item_id")
    if not isinstance(raw_positions, Mapping):
        return None
    value = raw_positions.get(item_id)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


__all__ = ["importance_content_digest", "task_node_ids"]

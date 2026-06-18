from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from abstraction.domain.common.identifiers import ArtifactId


@dataclass(frozen=True)
class ArtifactRef:
    artifact_id: ArtifactId
    artifact_kind: str
    path: Path
    metadata: Mapping[str, str]


@dataclass(frozen=True)
class ArtifactContract:
    artifact_kind: str
    producer_context: str
    consumer_context: str
    visibility: str


@dataclass(frozen=True)
class ArtifactManifestRef:
    manifest_path: Path
    manifest_version: str

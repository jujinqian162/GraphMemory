from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from enum import Enum
from pathlib import Path
from typing import Annotated, Generic, Literal, TypeAlias, TypeVar, Union, cast

from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter, validate_call


_CHUNK_SIZE = 1024 * 1024
_MANIFEST_NAME = "manifest.json"
_REFRESH_RECOVERY = "Rerun the experiment with cache.refresh=true."


class _ClosedModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)


class ArtifactKind(str, Enum):
    DATASET = "dataset"
    EVIDENCE_GRAPH = "evidence_graph"
    TRAINING_PAIRS = "training_pairs"
    MODEL = "model"
    PREDICTIONS = "predictions"
    EVALUATION = "evaluation"


class FileSourceRef(_ClosedModel):
    kind: Literal["file"] = "file"
    uri: str
    digest: str
    size_bytes: int = Field(ge=0)
    file_count: Literal[1] = 1


class DirectorySourceRef(_ClosedModel):
    kind: Literal["directory"] = "directory"
    uri: str
    digest: str
    size_bytes: int = Field(ge=0)
    file_count: int = Field(ge=0)


class RevisionSourceRef(_ClosedModel):
    kind: Literal["revision"] = "revision"
    uri: str
    revision: str = Field(min_length=1)
    digest: None = None
    size_bytes: int | None = Field(default=None, ge=0)
    file_count: int | None = Field(default=None, ge=0)


class ArtifactPayload(_ClosedModel):
    role: str = Field(min_length=1)
    relative_path: str = Field(min_length=1)
    kind: Literal["file", "directory"]
    digest: str = Field(min_length=1)
    size_bytes: int = Field(ge=0)
    file_count: int = Field(ge=1)


class ArtifactManifest(_ClosedModel):
    version: Literal[1] = 1
    artifact_id: str = Field(min_length=1)
    kind: ArtifactKind
    digest: str = Field(min_length=1)
    origin: dict[str, JsonValue]
    payloads: tuple[ArtifactPayload, ...] = Field(min_length=1)
    size_bytes: int = Field(ge=0)
    file_count: int = Field(ge=1)
    shape: dict[str, JsonValue] = Field(default_factory=dict)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


ArtifactKindT = TypeVar("ArtifactKindT", bound=ArtifactKind)


class ProcessedArtifactRef(_ClosedModel, Generic[ArtifactKindT]):
    uri: str
    kind: ArtifactKindT
    digest: str = Field(min_length=1)
    manifest_uri: str
    payloads: tuple[ArtifactPayload, ...] = Field(min_length=1)
    origin: dict[str, JsonValue]
    size_bytes: int = Field(ge=0)
    file_count: int = Field(ge=1)
    shape: dict[str, JsonValue] = Field(default_factory=dict)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class DatasetArtifactRef(ProcessedArtifactRef[Literal[ArtifactKind.DATASET]]):
    pass


class EvidenceGraphArtifactRef(
    ProcessedArtifactRef[Literal[ArtifactKind.EVIDENCE_GRAPH]]
):
    pass


class TrainingPairsArtifactRef(
    ProcessedArtifactRef[Literal[ArtifactKind.TRAINING_PAIRS]]
):
    pass


class ModelArtifactRef(ProcessedArtifactRef[Literal[ArtifactKind.MODEL]]):
    pass


class PredictionsArtifactRef(ProcessedArtifactRef[Literal[ArtifactKind.PREDICTIONS]]):
    pass


class EvaluationArtifactRef(ProcessedArtifactRef[Literal[ArtifactKind.EVALUATION]]):
    pass


ArtifactRef: TypeAlias = Annotated[
    Union[
        DatasetArtifactRef,
        EvidenceGraphArtifactRef,
        TrainingPairsArtifactRef,
        ModelArtifactRef,
        PredictionsArtifactRef,
        EvaluationArtifactRef,
    ],
    Field(discriminator="kind"),
]
_ARTIFACT_REF_ADAPTER = TypeAdapter(ArtifactRef)


class ProcessedAssetError(RuntimeError):
    pass


class ProcessedAssetStore:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    @property
    def datasets_root(self) -> Path:
        return self.root / "datasets"

    @property
    def evidence_graphs_root(self) -> Path:
        return self.root / "evidence_graphs"

    @property
    def training_pairs_root(self) -> Path:
        return self.root / "training_pairs"

    @property
    def models_root(self) -> Path:
        return self.root / "models"

    @property
    def predictions_root(self) -> Path:
        return self.root / "predictions"

    @property
    def evaluations_root(self) -> Path:
        return self.root / "evaluations"

    @property
    def prefect_results_root(self) -> Path:
        return self.root / "prefect" / "results"

    @property
    def staging_root(self) -> Path:
        return self.root / ".staging"

    def root_for(self, kind: ArtifactKind, *, namespace: str | None = None) -> Path:
        root = {
            ArtifactKind.DATASET: self.datasets_root,
            ArtifactKind.EVIDENCE_GRAPH: self.evidence_graphs_root,
            ArtifactKind.TRAINING_PAIRS: self.training_pairs_root,
            ArtifactKind.MODEL: self.models_root,
            ArtifactKind.PREDICTIONS: self.predictions_root,
            ArtifactKind.EVALUATION: self.evaluations_root,
        }[kind]
        root.mkdir(parents=True, exist_ok=True)
        if namespace is not None:
            if (
                not namespace
                or namespace in {".", ".."}
                or any(token in namespace for token in ("/", "\\"))
            ):
                raise ValueError(f"invalid processed asset namespace: {namespace!r}")
            root = root / namespace
            root.mkdir(parents=True, exist_ok=True)
        return root


class ArtifactPublisher:
    def __init__(
        self,
        store: ProcessedAssetStore,
        *,
        kind: ArtifactKind,
        origin: dict[str, JsonValue],
        namespace: str | None = None,
        task_identity: str | None = None,
    ) -> None:
        self.store = store
        self.kind = kind
        self.origin = dict(origin)
        self.namespace = namespace
        self.artifact_id = uuid.uuid4().hex
        prefix = _safe_component(task_identity or kind.value)
        self.workspace = store.staging_root / f"{prefix}-{self.artifact_id}"
        self._published = False

    def __enter__(self) -> ArtifactPublisher:
        self.workspace.parent.mkdir(parents=True, exist_ok=True)
        self.workspace.mkdir(parents=True, exist_ok=False)
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        if not self._published:
            shutil.rmtree(self.workspace, ignore_errors=True)

    def publish(
        self,
        declared_payloads: dict[str, str],
        *,
        shape: dict[str, JsonValue] | None = None,
        metadata: dict[str, JsonValue] | None = None,
    ) -> ArtifactRef:
        if self._published:
            raise ProcessedAssetError("artifact publisher has already published")
        if not declared_payloads:
            raise ProcessedAssetError(
                "artifact publication requires a declared payload"
            )

        payloads = tuple(
            self._payload(role, relative_path)
            for role, relative_path in sorted(declared_payloads.items())
        )
        digest = _artifact_digest(self.kind, payloads)
        manifest = ArtifactManifest(
            artifact_id=self.artifact_id,
            kind=self.kind,
            digest=digest,
            origin=self.origin,
            payloads=payloads,
            size_bytes=sum(payload.size_bytes for payload in payloads),
            file_count=sum(payload.file_count for payload in payloads),
            shape={} if shape is None else shape,
            metadata={} if metadata is None else metadata,
        )
        _write_json_durable(
            self.workspace / _MANIFEST_NAME,
            manifest.model_dump(mode="json"),
        )

        destination_root = self.store.root_for(self.kind, namespace=self.namespace)
        destination = destination_root / self.artifact_id
        if destination.exists():
            raise ProcessedAssetError(
                f"processed artifact destination already exists: {destination}"
            )
        os.replace(self.workspace, destination)
        self._published = True
        return _ARTIFACT_REF_ADAPTER.validate_python(
            {
                "uri": destination.as_posix(),
                "kind": self.kind,
                "digest": manifest.digest,
                "manifest_uri": (destination / _MANIFEST_NAME).as_posix(),
                "payloads": manifest.payloads,
                "origin": manifest.origin,
                "size_bytes": manifest.size_bytes,
                "file_count": manifest.file_count,
                "shape": manifest.shape,
                "metadata": manifest.metadata,
            }
        )

    def _payload(self, role: str, relative_path: str) -> ArtifactPayload:
        if not role:
            raise ProcessedAssetError("declared payload role must be non-empty")
        relative = Path(relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ProcessedAssetError(
                f"declared payload must stay within staging: {relative_path}"
            )
        path = self.workspace / relative
        if not path.exists():
            raise ProcessedAssetError(f"declared payload is missing: {path}")
        if path.is_file():
            digest, size = _file_digest(path)
            return ArtifactPayload(
                role=role,
                relative_path=relative.as_posix(),
                kind="file",
                digest=digest,
                size_bytes=size,
                file_count=1,
            )
        if path.is_dir():
            digest, size, file_count = _directory_digest(path)
            if file_count == 0:
                raise ProcessedAssetError(
                    f"declared payload directory is empty: {path}"
                )
            return ArtifactPayload(
                role=role,
                relative_path=relative.as_posix(),
                kind="directory",
                digest=digest,
                size_bytes=size,
                file_count=file_count,
            )
        raise ProcessedAssetError(f"declared payload has unsupported type: {path}")

@validate_call
def identify_external_source(
    path: str | Path,
    *,
    repository_root: Path | None = None,
) -> FileSourceRef | DirectorySourceRef:
    source = Path(path).resolve()
    if repository_root is not None:
        _reject_run_input(source, repository_root.resolve())
    if source.is_file():
        digest, size = _file_digest(source)
        return FileSourceRef(
            uri=source.as_posix(),
            digest=digest,
            size_bytes=size,
        )
    if source.is_dir():
        digest, size, file_count = _directory_digest(source)
        return DirectorySourceRef(
            uri=source.as_posix(),
            digest=digest,
            size_bytes=size,
            file_count=file_count,
        )
    raise FileNotFoundError(f"external source does not exist: {source}")


@validate_call
def identify_immutable_revision(
    uri: str,
    revision: str,
    *,
    size_bytes: int | None = None,
) -> RevisionSourceRef:
    return RevisionSourceRef(uri=uri, revision=revision, size_bytes=size_bytes)


@validate_call
def artifact_payload_path(
    reference: ArtifactRef,
    role: str,
) -> Path:
    for payload in reference.payloads:
        if payload.role == role:
            path = Path(reference.uri) / payload.relative_path
            exists = path.is_file() if payload.kind == "file" else path.is_dir()
            if not exists:
                raise ProcessedAssetError(
                    f"processed artifact payload is missing: {path}. {_REFRESH_RECOVERY}"
                )
            return path
    raise ProcessedAssetError(
        f"processed artifact has no payload role={role!r}: {reference.uri}. "
        f"{_REFRESH_RECOVERY}"
    )


def _reject_run_input(path: Path, repository_root: Path) -> None:
    runs_root = (repository_root / "runs").resolve()
    if path == runs_root or path.is_relative_to(runs_root):
        raise ValueError(
            f"runs/ is output-only and cannot be a scientific input: {path}"
        )


def _file_digest(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _directory_digest(path: Path) -> tuple[str, int, int]:
    digest = hashlib.sha256()
    total_size = 0
    file_count = 0
    files = sorted(item for item in path.rglob("*") if item.is_file())
    for item in files:
        if item.is_symlink():
            raise ValueError(f"symlinks are not supported in content identity: {item}")
        relative = item.relative_to(path).as_posix().encode("utf-8")
        file_digest, size = _file_digest(item)
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(size.to_bytes(16, "big"))
        digest.update(bytes.fromhex(file_digest))
        total_size += size
        file_count += 1
    return digest.hexdigest(), total_size, file_count


def _artifact_digest(
    kind: ArtifactKind,
    payloads: tuple[ArtifactPayload, ...],
) -> str:
    value = {
        "kind": kind.value,
        "payloads": [payload.model_dump(mode="json") for payload in payloads],
    }
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json_durable(path: Path, value: object) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _safe_component(value: str) -> str:
    rendered = "".join(character if character.isalnum() else "-" for character in value)
    return cast(str, rendered.strip("-") or "task")


__all__ = [
    "ArtifactKind",
    "ArtifactManifest",
    "ArtifactPayload",
    "ArtifactPublisher",
    "ArtifactRef",
    "DatasetArtifactRef",
    "DirectorySourceRef",
    "EvaluationArtifactRef",
    "EvidenceGraphArtifactRef",
    "FileSourceRef",
    "ModelArtifactRef",
    "PredictionsArtifactRef",
    "ProcessedArtifactRef",
    "ProcessedAssetError",
    "ProcessedAssetStore",
    "RevisionSourceRef",
    "TrainingPairsArtifactRef",
    "artifact_payload_path",
    "identify_external_source",
    "identify_immutable_revision",
]

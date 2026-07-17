from __future__ import annotations

from pathlib import Path

import pytest

from graph_memory.experiment.artifacts import (
    ArtifactKind,
    ArtifactPublisher,
    ProcessedAssetError,
    ProcessedAssetStore,
    artifact_payload_path,
    identify_external_source,
)


def test_external_file_and_directory_identity_tracks_content(tmp_path: Path) -> None:
    source_file = tmp_path / "source.json"
    source_file.write_text('{"value": 1}\n', encoding="utf-8")
    first_file = identify_external_source(source_file)
    source_file.write_text('{"value": 2}\n', encoding="utf-8")
    second_file = identify_external_source(source_file)

    source_dir = tmp_path / "model"
    source_dir.mkdir()
    (source_dir / "config.json").write_text("{}\n", encoding="utf-8")
    first_dir = identify_external_source(source_dir)
    (source_dir / "weights.bin").write_bytes(b"weights")
    second_dir = identify_external_source(source_dir)

    assert first_file.digest != second_file.digest
    assert first_dir.digest != second_dir.digest
    assert second_dir.file_count == 2


def test_directory_identity_is_independent_of_creation_order(tmp_path: Path) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()
    (left / "b.txt").write_text("b", encoding="utf-8")
    (left / "a.txt").write_text("a", encoding="utf-8")
    (right / "a.txt").write_text("a", encoding="utf-8")
    (right / "b.txt").write_text("b", encoding="utf-8")

    assert (
        identify_external_source(left).digest == identify_external_source(right).digest
    )


def test_atomic_publication_returns_typed_payload_metadata(tmp_path: Path) -> None:
    store = ProcessedAssetStore(tmp_path / "data" / "processed")
    with ArtifactPublisher(
        store,
        kind=ArtifactKind.DATASET,
        origin={"stage": "prepare", "version": "prepare-v1"},
    ) as publisher:
        (publisher.workspace / "tasks.json").write_text("[]\n", encoding="utf-8")
        reference = publisher.publish({"tasks": "tasks.json"})

    assert Path(reference.uri).parent == store.datasets_root
    assert Path(reference.manifest_uri).is_file()
    assert reference.payloads[0].role == "tasks"
    assert artifact_payload_path(reference, "tasks").is_file()
    assert not any(store.staging_root.iterdir())


def test_failed_publication_is_not_visible(tmp_path: Path) -> None:
    store = ProcessedAssetStore(tmp_path / "data" / "processed")
    with pytest.raises(ProcessedAssetError, match="declared payload"):
        with ArtifactPublisher(
            store,
            kind=ArtifactKind.MODEL,
            origin={"stage": "train", "version": "train-v1"},
        ) as publisher:
            _ = publisher.publish({"checkpoint": "best.pt"})

    assert list(store.models_root.rglob("manifest.json")) == []
    assert not any(store.staging_root.iterdir())


def test_missing_declared_payload_reports_refresh_command(tmp_path: Path) -> None:
    store = ProcessedAssetStore(tmp_path / "data" / "processed")
    with ArtifactPublisher(
        store,
        kind=ArtifactKind.PREDICTIONS,
        origin={"stage": "rank", "version": "rank-v1"},
    ) as publisher:
        (publisher.workspace / "predictions.json").write_text("[]\n", encoding="utf-8")
        reference = publisher.publish({"predictions": "predictions.json"})

    artifact_payload_path(reference, "predictions").unlink()
    with pytest.raises(ProcessedAssetError, match=r"cache\.refresh=true"):
        artifact_payload_path(reference, "predictions")


def test_scientific_source_below_runs_is_rejected(tmp_path: Path) -> None:
    repository_root = tmp_path / "repo"
    source = repository_root / "runs" / "old" / "checkpoint.pt"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"checkpoint")

    with pytest.raises(ValueError, match="output-only"):
        identify_external_source(source, repository_root=repository_root)

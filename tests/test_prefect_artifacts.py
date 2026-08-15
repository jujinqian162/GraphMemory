from __future__ import annotations

from pathlib import Path

import pytest

from graph_memory.experiment.artifacts import (
    ArtifactKind,
    ArtifactPublisher,
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


def test_scientific_source_below_runs_is_rejected(tmp_path: Path) -> None:
    repository_root = tmp_path / "repo"
    source = repository_root / "runs" / "old" / "checkpoint.pt"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"checkpoint")

    with pytest.raises(ValueError, match="output-only"):
        identify_external_source(source, repository_root=repository_root)

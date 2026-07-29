from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import huggingface_hub
import pytest

from graph_memory.experiment import inputs
from graph_memory.experiment.config import ResolvedExperimentConfig
from scripts import prepare_dataset


def test_isetrace_registry_is_revision_pinned_and_complete() -> None:
    spec = prepare_dataset.DATASET_REGISTRY["isetrace"]

    assert len(spec.files) == 9
    assert {item.split for item in spec.files} == {
        "intents",
        *(f"trajectories-{index:05d}" for index in range(8)),
    }
    assert all(
        f"/resolve/{prepare_dataset.ISETRACE_REVISION}/" in item.url
        for item in spec.files
    )
    assert all(item.num_bytes is not None for item in spec.files)


def test_prepare_dataset_retries_huggingface_download_once_via_mirror(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_url = "https://huggingface.co/datasets/example/data/resolve/rev/train.jsonl"
    spec = prepare_dataset.DatasetSpec(
        dataset="example",
        display_name="Example",
        files=(
            prepare_dataset.DatasetFile(
                split="train",
                filename="nested/train.jsonl",
                url=source_url,
            ),
        ),
    )
    observed_urls: list[str] = []

    def download(url: str, destination: Path, _size: int | None) -> None:
        observed_urls.append(url)
        if len(observed_urls) == 1:
            raise OSError("primary endpoint unavailable")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(
        prepare_dataset,
        "SUMMARY_DIR",
        tmp_path / "summaries",
    )

    result = prepare_dataset.main(
        [
            "--dataset",
            "example",
            "--name",
            "example",
            "--data_dir",
            str(tmp_path / "data"),
            "--no_verify",
        ],
        registry={"example": spec},
        downloader=download,
        show_progress=False,
    )

    assert result == 0
    assert observed_urls == [
        source_url,
        "https://hf-mirror.com/datasets/example/data/resolve/rev/train.jsonl",
    ]
    assert (tmp_path / "data/example/raw/nested/train.jsonl").exists()


def test_prepare_dataset_does_not_mirror_non_huggingface_downloads(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    spec = prepare_dataset.DatasetSpec(
        dataset="example",
        display_name="Example",
        files=(
            prepare_dataset.DatasetFile(
                split="train",
                filename="train.json",
                url="https://example.com/train.json",
            ),
        ),
    )
    calls = 0

    def fail(_url: str, _destination: Path, _size: int | None) -> None:
        nonlocal calls
        calls += 1
        raise OSError("unavailable")

    monkeypatch.setattr(
        prepare_dataset,
        "SUMMARY_DIR",
        tmp_path / "summaries",
    )

    with pytest.raises(OSError, match="unavailable"):
        prepare_dataset.main(
            [
                "--dataset",
                "example",
                "--name",
                "example",
                "--data_dir",
                str(tmp_path / "data"),
                "--no_verify",
            ],
            registry={"example": spec},
            downloader=fail,
            show_progress=False,
        )

    assert calls == 1


def test_ensure_dataset_dispatches_isetrace_download(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed: list[list[str]] = []
    prepare_module = SimpleNamespace(
        DATASET_REGISTRY=prepare_dataset.DATASET_REGISTRY,
        main=lambda argv: observed.append(list(argv)),
    )
    monkeypatch.setattr(inputs, "_load_prepare_dataset", lambda _root: prepare_module)
    missing = Path("data/isetrace/raw/trajectories/trajectories-00000.jsonl")
    config = cast(
        ResolvedExperimentConfig,
        cast(
            object,
            SimpleNamespace(
                dataset=SimpleNamespace(
                    name="isetrace",
                    splits={
                        split: SimpleNamespace(source=missing)
                        for split in ("train", "dev", "test")
                    },
                )
            ),
        ),
    )

    inputs.ensure_dataset(config, repository_root=tmp_path)

    assert observed == [
        [
            "--dataset",
            "isetrace",
            "--name",
            "isetrace",
            "--data_dir",
            str(tmp_path / "data"),
            "--no_verify",
        ]
    ]


def test_ensure_dataset_skips_complete_isetrace_raw_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    spec = prepare_dataset.DATASET_REGISTRY["isetrace"]
    raw_dir = tmp_path / "data/isetrace/raw"
    for item in spec.files:
        path = raw_dir / item.filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    prepare_module = SimpleNamespace(
        DATASET_REGISTRY=prepare_dataset.DATASET_REGISTRY,
        main=lambda _argv: pytest.fail("complete ISETrace data must not download"),
    )
    monkeypatch.setattr(inputs, "_load_prepare_dataset", lambda _root: prepare_module)
    missing_split = Path("data/isetrace/derived/not-built-yet.jsonl")
    config = cast(
        ResolvedExperimentConfig,
        cast(
            object,
            SimpleNamespace(
                dataset=SimpleNamespace(
                    name="isetrace",
                    splits={
                        split: SimpleNamespace(source=missing_split)
                        for split in ("train", "dev", "test")
                    },
                )
            ),
        ),
    )

    inputs.ensure_dataset(config, repository_root=tmp_path)


def test_encoder_download_retries_once_via_huggingface_mirror(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[dict[str, object]] = []

    def snapshot_download(**kwargs: object) -> str:
        calls.append(kwargs)
        if len(calls) == 1:
            raise OSError("primary endpoint unavailable")
        return str(kwargs["local_dir"])

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)

    inputs._ensure_model(
        "models/intfloat-e5-base-v2",
        repository_root=tmp_path,
    )

    assert len(calls) == 2
    assert "endpoint" not in calls[0]
    assert calls[1]["endpoint"] == inputs.HUGGINGFACE_MIRROR_BASE_URL

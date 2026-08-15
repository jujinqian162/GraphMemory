from __future__ import annotations

from pathlib import Path

import huggingface_hub
import pytest

from graph_memory.experiment import inputs
from scripts import prepare_dataset


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


def test_prepare_dataset_can_use_huggingface_mirror_directly(
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
                filename="train.jsonl",
                url=source_url,
            ),
        ),
    )
    observed_urls: list[str] = []

    def download(url: str, destination: Path, _size: int | None) -> None:
        observed_urls.append(url)
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
            "--mirror",
        ],
        registry={"example": spec},
        downloader=download,
        show_progress=False,
    )

    assert result == 0
    assert observed_urls == [
        "https://hf-mirror.com/datasets/example/data/resolve/rev/train.jsonl"
    ]


def test_encoder_download_uses_huggingface_mirror_and_revision(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[dict[str, object]] = []

    def snapshot_download(**kwargs: object) -> str:
        calls.append(kwargs)
        return str(kwargs["local_dir"])

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)

    inputs._ensure_model(
        "models/BAAI-bge-reranker-base",
        repository_root=tmp_path,
    )

    assert calls == [
        {
            "repo_id": "BAAI/bge-reranker-base",
            "revision": "2cfc18c9415c912f9d8155881c133215df768a70",
            "local_dir": str(tmp_path / "models/BAAI-bge-reranker-base"),
            "endpoint": inputs.HUGGINGFACE_MIRROR_BASE_URL,
        }
    ]


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

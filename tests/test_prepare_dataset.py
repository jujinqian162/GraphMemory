from __future__ import annotations

import hashlib

import pytest

from graph_memory.io import read_json
import scripts.prepare_dataset as prepare_dataset


def test_prepare_dataset_downloads_registered_files_with_status_summary(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare_dataset, "SUMMARY_DIR", tmp_path / "summaries")
    payload = b'{"ok": true}\n'
    checksum = hashlib.sha256(payload).hexdigest()
    calls: list[str] = []

    registry = {
        "tiny": prepare_dataset.DatasetSpec(
            dataset="tiny",
            display_name="Tiny Dataset",
            files=(
                prepare_dataset.DatasetFile(
                    split="train",
                    filename="train.json",
                    url="https://example.test/train.json",
                    sha256=checksum,
                    num_bytes=len(payload),
                ),
            ),
        )
    }

    def fake_downloader(url, destination, expected_size):
        calls.append(url)
        destination.write_bytes(payload)

    exit_code = prepare_dataset.main(
        ["--dataset", "tiny", "--name", "demo", "--data_dir", str(tmp_path)],
        registry=registry,
        downloader=fake_downloader,
        show_progress=False,
    )

    raw_path = tmp_path / "demo" / "raw" / "train.json"
    summary_path = prepare_dataset.SUMMARY_DIR / "prepare_dataset.run_summary.json"

    assert exit_code == 0
    assert raw_path.read_bytes() == payload
    assert calls == ["https://example.test/train.json"]
    summary = read_json(summary_path)
    assert summary["status"] == "success"
    assert summary["counts"]["downloaded_files"] == 1
    assert summary["outputs"]["raw_dir"] == str(tmp_path / "demo" / "raw")


def test_prepare_dataset_skips_existing_file_when_checksum_matches(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare_dataset, "SUMMARY_DIR", tmp_path / "summaries")
    payload = b"already downloaded"
    checksum = hashlib.sha256(payload).hexdigest()
    raw_path = tmp_path / "demo" / "raw" / "train.json"
    raw_path.parent.mkdir(parents=True)
    raw_path.write_bytes(payload)

    registry = {
        "tiny": prepare_dataset.DatasetSpec(
            dataset="tiny",
            display_name="Tiny Dataset",
            files=(
                prepare_dataset.DatasetFile(
                    split="train",
                    filename="train.json",
                    url="https://example.test/train.json",
                    sha256=checksum,
                    num_bytes=len(payload),
                ),
            ),
        )
    }

    def failing_downloader(url, destination, expected_size):
        raise AssertionError("download should be skipped")

    prepare_dataset.main(
        ["--dataset", "tiny", "--name", "demo", "--data_dir", str(tmp_path)],
        registry=registry,
        downloader=failing_downloader,
        show_progress=False,
    )

    summary = read_json(prepare_dataset.SUMMARY_DIR / "prepare_dataset.run_summary.json")
    assert summary["counts"]["skipped_files"] == 1
    assert summary["counts"]["downloaded_files"] == 0


def test_prepare_dataset_records_failed_summary_when_checksum_mismatches(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare_dataset, "SUMMARY_DIR", tmp_path / "summaries")
    expected_payload = b"expected"
    downloaded_payload = b"corrupt"
    checksum = hashlib.sha256(expected_payload).hexdigest()
    registry = {
        "tiny": prepare_dataset.DatasetSpec(
            dataset="tiny",
            display_name="Tiny Dataset",
            files=(
                prepare_dataset.DatasetFile(
                    split="train",
                    filename="train.json",
                    url="https://example.test/train.json",
                    sha256=checksum,
                    num_bytes=len(downloaded_payload),
                ),
            ),
        )
    }

    def corrupt_downloader(url, destination, expected_size):
        destination.write_bytes(downloaded_payload)

    with pytest.raises(ValueError, match="checksum mismatch"):
        prepare_dataset.main(
            ["--dataset", "tiny", "--name", "demo", "--data_dir", str(tmp_path)],
            registry=registry,
            downloader=corrupt_downloader,
            show_progress=False,
        )

    summary = read_json(prepare_dataset.SUMMARY_DIR / "prepare_dataset.run_summary.json")
    assert summary["status"] == "failed"
    assert "checksum mismatch" in summary["error"]

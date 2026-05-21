from __future__ import annotations

import hashlib

import scripts.prepare_dataset as prepare_dataset


def test_prepare_dataset_downloads_registered_files_with_status_summary(tmp_path):
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
    summary = prepare_dataset.read_json(summary_path)
    assert summary["status"] == "success"
    assert summary["counts"]["downloaded_files"] == 1
    assert summary["outputs"]["raw_dir"] == str(tmp_path / "demo" / "raw")


def test_prepare_dataset_skips_existing_file_when_checksum_matches(tmp_path):
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

    summary = prepare_dataset.read_json(prepare_dataset.SUMMARY_DIR / "prepare_dataset.run_summary.json")
    assert summary["counts"]["skipped_files"] == 1
    assert summary["counts"]["downloaded_files"] == 0

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graph_memory.observability import build_run_summary, collect_environment, now_iso, write_run_summary

LOGGER = logging.getLogger("prepare_dataset")


@dataclass(frozen=True)
class DatasetFile:
    split: str
    filename: str
    url: str
    sha256: str | None = None
    num_bytes: int | None = None


@dataclass(frozen=True)
class DatasetSpec:
    dataset: str
    display_name: str
    files: tuple[DatasetFile, ...]


@dataclass(frozen=True)
class PrepareDatasetArgs:
    dataset: str
    name: str
    data_dir: str
    force: bool
    no_verify: bool


Downloader = Callable[[str, Path, int | None], None]

SUMMARY_DIR = Path("results") / "debug" / "datasets-prepare"
DATASET_REGISTRY: dict[str, DatasetSpec] = {
    "hotpotqa-v1": DatasetSpec(
        dataset="hotpotqa-v1",
        display_name="HotpotQA v1 distractor",
        files=(
            DatasetFile(
                split="train",
                filename="train.json",
                url="http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json",
                sha256="26650cf50234ef5fb2e664ed70bbecdfd87815e6bffc257e068efea5cf7cd316",
                num_bytes=566426227,
            ),
            DatasetFile(
                split="dev",
                filename="dev.json",
                url="http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json",
                sha256="4e9ecb5c8d3b719f624d66b60f8d56bf227f03914f5f0753d6fa1b359d7104ea",
                num_bytes=46320117,
            ),
        ),
    )
}


def main(
    argv: Sequence[str] | None = None,
    *,
    registry: Mapping[str, DatasetSpec] | None = None,
    downloader: Downloader | None = None,
    show_progress: bool = True,
) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")

    started_at = now_iso()
    start_time = time.perf_counter()
    active_registry = registry or DATASET_REGISTRY
    dataset_spec = get_dataset_spec(active_registry, args.dataset)
    dataset_dir = Path(args.data_dir) / args.name
    raw_dir = dataset_dir / "raw"
    summary_path = SUMMARY_DIR / "prepare_dataset.run_summary.json"
    effective_config = {
        "dataset": args.dataset,
        "name": args.name,
        "data_dir": args.data_dir,
        "raw_dir": str(raw_dir),
        "force": args.force,
        "verify_checksum": not args.no_verify,
    }
    inputs = {
        "dataset": args.dataset,
        "sources": {dataset_file.split: dataset_file.url for dataset_file in dataset_spec.files},
    }
    outputs = {
        "dataset_dir": str(dataset_dir),
        "raw_dir": str(raw_dir),
        "run_summary": str(summary_path),
    }
    counts = {"dataset_files": len(dataset_spec.files), "downloaded_files": 0, "skipped_files": 0}
    notes: list[str] = []

    try:
        raw_dir.mkdir(parents=True, exist_ok=True)
        status(f"Dataset: {dataset_spec.display_name} ({dataset_spec.dataset})")
        status(f"Target: {raw_dir}")

        download_file = downloader or download_url
        file_iterator = tqdm(
            dataset_spec.files,
            desc="files",
            unit="file",
            disable=not show_progress,
            dynamic_ncols=True,
        )
        for dataset_file in file_iterator:
            destination = raw_dir / dataset_file.filename
            file_iterator.set_postfix_str(dataset_file.split)
            if should_skip(destination, dataset_file, verify=not args.no_verify, force=args.force):
                counts["skipped_files"] += 1
                status(f"OK skip {dataset_file.split}: {destination}")
                continue

            status(f"GET {dataset_file.split}: {dataset_file.url}")
            download_file(dataset_file.url, destination, dataset_file.num_bytes)
            if not args.no_verify:
                verify_file(destination, dataset_file)
            counts["downloaded_files"] += 1
            status(f"OK saved {dataset_file.split}: {destination}")

        summary = build_run_summary(
            script="prepare_dataset.py",
            started_at=started_at,
            finished_at=now_iso(),
            status="success",
            effective_config=effective_config,
            inputs=inputs,
            outputs=outputs,
            counts=counts,
            timings={"total_seconds": time.perf_counter() - start_time},
            environment=collect_environment(),
            notes=notes,
        )
        write_run_summary(summary_path, summary)
        status(f"Summary: {summary_path}")
        return 0
    except Exception as error:
        LOGGER.error("%s", error)
        summary = build_run_summary(
            script="prepare_dataset.py",
            started_at=started_at,
            finished_at=now_iso(),
            status="failed",
            effective_config=effective_config,
            inputs=inputs,
            outputs=outputs,
            counts=counts,
            timings={"total_seconds": time.perf_counter() - start_time},
            environment=collect_environment(),
            notes=notes,
            error=str(error),
        )
        write_run_summary(summary_path, summary)
        raise


def status(message: str) -> None:
    tqdm.write(f"[prepare_dataset] {message}")


def get_dataset_spec(registry: Mapping[str, DatasetSpec], dataset: str) -> DatasetSpec:
    try:
        return registry[dataset]
    except KeyError as error:
        available = ", ".join(sorted(registry))
        raise ValueError(f"Unknown dataset '{dataset}'. Available datasets: {available}") from error


def should_skip(destination: Path, dataset_file: DatasetFile, *, verify: bool, force: bool) -> bool:
    if force or not destination.exists():
        return False
    if verify:
        verify_file(destination, dataset_file)
    return True


def verify_file(path: Path, dataset_file: DatasetFile) -> None:
    if dataset_file.num_bytes is not None and path.stat().st_size != dataset_file.num_bytes:
        raise ValueError(
            f"Downloaded file has wrong size: {path} "
            f"expected={dataset_file.num_bytes} actual={path.stat().st_size}"
        )
    if dataset_file.sha256 is not None:
        actual = sha256_file(path)
        if actual != dataset_file.sha256:
            raise ValueError(f"Downloaded file checksum mismatch: {path} expected={dataset_file.sha256} actual={actual}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_url(url: str, destination: Path, expected_size: int | None) -> None:
    temporary_path = destination.with_suffix(f"{destination.suffix}.part")
    temporary_path.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url) as response:
        total = expected_size or int(response.headers.get("Content-Length", "0") or 0) or None
        with temporary_path.open("wb") as output:
            with tqdm(total=total, unit="B", unit_scale=True, desc=destination.name, dynamic_ncols=True) as progress:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
                    progress.update(len(chunk))
    temporary_path.replace(destination)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download registered raw datasets into data/<name>/raw.")
    parser.add_argument("--dataset", required=True, help="Dataset registry key to download, e.g. hotpotqa-v1.")
    parser.add_argument("--name", required=True, help="Dataset directory name under --data_dir, e.g. hotpotqa.")
    parser.add_argument("--data_dir", default="data", help="Root data directory. Defaults to data.")
    parser.add_argument("--force", action="store_true", help="Re-download files even when they already exist.")
    parser.add_argument("--no_verify", action="store_true", help="Skip size and SHA-256 verification.")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> PrepareDatasetArgs:
    namespace = build_parser().parse_args(argv)
    return PrepareDatasetArgs(
        dataset=namespace.dataset,
        name=namespace.name,
        data_dir=namespace.data_dir,
        force=namespace.force,
        no_verify=namespace.no_verify,
    )


if __name__ == "__main__":
    raise SystemExit(main())

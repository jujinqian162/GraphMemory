from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

DEFAULT_MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def collect_run_artifacts(
    run_dir: str | Path,
    *,
    output_root: str | Path = "results",
    max_file_size_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES,
    include_report: bool = False,
    report_dir: str | Path = "report",
    dry_run: bool = False,
) -> dict[str, Any]:
    source = Path(run_dir)
    if not source.exists() or not source.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {source}")

    output_dir = Path(output_root) / source.name
    _reject_overlapping_paths(source, output_dir)
    run_mode, jobs = _detect_run_structure(source)
    copied: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    total_copied_bytes = 0
    staging_dir: Path | None = None

    if not dry_run:
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        staging_dir = output_dir.with_name(f".{output_dir.name}.delivery-{uuid4().hex}")
        staging_dir.mkdir(parents=True)
        for path in sorted(source.rglob("*")):
            if path.is_dir():
                (staging_dir / path.relative_to(source)).mkdir(
                    parents=True, exist_ok=True
                )

    try:
        for path in sorted(source.rglob("*")):
            if not path.is_file():
                continue
            relative_path = path.relative_to(source).as_posix()
            size_bytes = path.stat().st_size
            decision = _classify_run_file(
                relative_path, size_bytes, max_file_size_bytes
            )
            entry = {"relative_path": relative_path, "size_bytes": size_bytes}
            if decision == "copy":
                copied.append(entry)
                total_copied_bytes += size_bytes
                if staging_dir is not None:
                    destination = staging_dir / relative_path
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, destination)
            else:
                skipped.append({**entry, "reason": decision})

        if include_report:
            report_source = Path(report_dir)
            for path in (
                sorted(report_source.rglob("*")) if report_source.exists() else []
            ):
                if not path.is_file():
                    continue
                relative_path = f"report/{path.relative_to(report_source).as_posix()}"
                size_bytes = path.stat().st_size
                if size_bytes > max_file_size_bytes:
                    skipped.append(
                        {
                            "relative_path": relative_path,
                            "size_bytes": size_bytes,
                            "reason": "too_large",
                        }
                    )
                    continue
                copied.append(
                    {"relative_path": relative_path, "size_bytes": size_bytes}
                )
                total_copied_bytes += size_bytes
                if staging_dir is not None:
                    destination = staging_dir / relative_path
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, destination)

        manifest = {
            "source_run_dir": str(source.resolve()),
            "run_name": source.name,
            "run_mode": run_mode,
            "jobs": jobs,
            "output_dir": str(output_dir.resolve()),
            "max_file_size_bytes": max_file_size_bytes,
            "dry_run": dry_run,
            "include_report": include_report,
            "copied": copied,
            "skipped": skipped,
            "copied_count": len(copied),
            "skipped_count": len(skipped),
            "total_copied_bytes": total_copied_bytes,
        }
        if staging_dir is not None:
            _write_json(staging_dir / "delivery_manifest.json", manifest)
            _replace_directory(staging_dir, output_dir)
            staging_dir = None
        return manifest
    finally:
        if staging_dir is not None and staging_dir.exists():
            shutil.rmtree(staging_dir)


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    run_dir = REPOSITORY_ROOT / "runs" / args.name
    manifest = collect_run_artifacts(
        run_dir,
        output_root=args.output_root,
        max_file_size_bytes=_megabytes_to_bytes(args.max_file_size_mb),
        include_report=args.include_report,
        report_dir=args.report_dir,
        dry_run=args.dry_run,
    )
    print(
        f"output={manifest['output_dir']} "
        f"copied={manifest['copied_count']} "
        f"skipped={manifest['skipped_count']} "
        f"bytes={manifest['total_copied_bytes']}"
    )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Mirror one complete named run while excluding known huge artifacts. "
            "Contract: --name <train_id> reads the full runs/<name> tree and writes "
            "results/<name> by default, including every multirun job."
        )
    )
    parser.add_argument(
        "--name", required=True, help="Run name under runs/, e.g. rgcn_full_train."
    )
    parser.add_argument(
        "--output-root",
        default="results",
        help="Destination root; run name is appended.",
    )
    parser.add_argument(
        "--max-file-size-mb",
        type=float,
        default=DEFAULT_MAX_FILE_SIZE_BYTES / (1024 * 1024),
        help="Maximum size for files that match include rules.",
    )
    parser.add_argument(
        "--include-report",
        action="store_true",
        help="Also copy files from --report-dir under report/.",
    )
    parser.add_argument(
        "--report-dir",
        default="report",
        help="Report directory used with --include-report.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print manifest summary without copying files.",
    )
    return parser


def _classify_run_file(
    relative_path: str, size_bytes: int, max_file_size_bytes: int
) -> str:
    path = Path(relative_path)
    parts = path.parts
    name = path.name

    exclusion = _known_exclusion(parts, name)
    if exclusion is not None:
        return exclusion
    if size_bytes > max_file_size_bytes:
        return "too_large"
    return "copy"


def _known_exclusion(parts: tuple[str, ...], name: str) -> str | None:
    if not parts:
        return "not_selected"
    directories = parts[:-1]
    # New runs are output-only. These guards keep the collector safe when it is
    # pointed at a historical run that still contains computation artifacts.
    if "inputs" in directories:
        return "historical_input"
    if "graphs" in directories and name.endswith(".graphs.json"):
        return "historical_graph"
    if "predictions" in directories and name.endswith(".ranked.json"):
        return "historical_prediction"
    if "checkpoints" in directories or Path(name).suffix in {".pt", ".ckpt"}:
        return "historical_checkpoint"
    if name == "train.pairs.json":
        return "historical_train_pairs"
    if name.endswith(".dev_candidates.json"):
        return "historical_tuning_candidates"
    if Path(name).suffix in {".bin", ".safetensors", ".npy", ".npz"}:
        return "excluded_model_or_embedding"
    return None


def _detect_run_structure(source: Path) -> tuple[str, list[str]]:
    jobs = sorted(
        path.parent.parent.name
        for path in source.glob("*/workflow/summary.yaml")
        if path.is_file()
    )
    if jobs or (source / "multirun.yaml").is_file():
        return "multirun", jobs
    return "single", []


def _reject_overlapping_paths(source: Path, output_dir: Path) -> None:
    resolved_source = source.resolve()
    resolved_output = output_dir.resolve()
    if (
        resolved_source == resolved_output
        or resolved_source.is_relative_to(resolved_output)
        or resolved_output.is_relative_to(resolved_source)
    ):
        raise ValueError(
            "Delivery source and output directories cannot overlap: "
            f"source={resolved_source} output={resolved_output}"
        )


def _replace_directory(staging_dir: Path, output_dir: Path) -> None:
    backup_dir: Path | None = None
    if output_dir.exists():
        backup_dir = output_dir.with_name(f".{output_dir.name}.backup-{uuid4().hex}")
        output_dir.replace(backup_dir)
    try:
        staging_dir.replace(output_dir)
    except BaseException:
        if backup_dir is not None and not output_dir.exists():
            backup_dir.replace(output_dir)
        raise
    if backup_dir is not None:
        if backup_dir.is_dir():
            shutil.rmtree(backup_dir)
        else:
            backup_dir.unlink()


def _megabytes_to_bytes(value: float) -> int:
    if value <= 0:
        raise ValueError("--max-file-size-mb must be positive.")
    return int(value * 1024 * 1024)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(data, file, ensure_ascii=False, indent=2, sort_keys=True)
        file.write("\n")


if __name__ == "__main__":
    raise SystemExit(main())

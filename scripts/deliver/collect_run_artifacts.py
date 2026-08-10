from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

DEFAULT_MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
REQUIRED_JOB_FILES = (
    "config/resolved.yaml",
    "config/overrides.yaml",
    "workflow/summary.yaml",
    "workflow/ranking_origin.yaml",
    "assets/manifest.yaml",
    "metrics/final.metrics.csv",
    "metrics/per_task.jsonl",
    "debug/failure_cases.jsonl",
)
REQUIRED_SCHEMA_V2_JOB_FILES = ("predictions/ranked_prefix.jsonl.gz",)


def collect_run_artifacts(
    run_dir: str | Path,
    *,
    output_root: str | Path = "results",
    max_file_size_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES,
    include_report: bool = False,
    report_dir: str | Path = "report",
    dry_run: bool = False,
    validate_complete: bool = True,
    expected_per_task_count: int | None = None,
) -> dict[str, Any]:
    source = Path(run_dir)
    if not source.exists() or not source.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {source}")

    output_dir = Path(output_root) / source.name
    _reject_overlapping_paths(source, output_dir)
    run_mode, jobs = _detect_run_structure(source)
    scientific_jobs = (
        _validate_complete_jobs(
            source,
            run_mode=run_mode,
            jobs=jobs,
            expected_per_task_count=expected_per_task_count,
        )
        if validate_complete
        else []
    )
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
            if (
                validate_complete
                and decision != "copy"
                and _is_required_job_file(relative_path, run_mode=run_mode, jobs=jobs)
            ):
                raise ValueError(
                    f"required run artifact would be skipped: {relative_path} ({decision})"
                )
            if decision == "copy":
                copied_entry = {**entry, "sha256": _sha256_file(path)}
                copied.append(copied_entry)
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
                    {
                        "relative_path": relative_path,
                        "size_bytes": size_bytes,
                        "sha256": _sha256_file(path),
                    }
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
            "validate_complete": validate_complete,
            "expected_per_task_count": expected_per_task_count,
            "scientific_jobs": scientific_jobs,
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
        validate_complete=not args.allow_incomplete,
        expected_per_task_count=args.expected_per_task_count,
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
    parser.add_argument(
        "--expected-per-task-count",
        type=int,
        help="Fail unless every job contains exactly this many unique per-task rows.",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Permit historical/debug trees that do not satisfy the complete-run contract.",
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


def _validate_complete_jobs(
    source: Path,
    *,
    run_mode: str,
    jobs: Sequence[str],
    expected_per_task_count: int | None,
) -> list[dict[str, Any]]:
    if expected_per_task_count is not None and expected_per_task_count <= 0:
        raise ValueError("expected_per_task_count must be positive")
    roots = (
        [(job, source / job) for job in jobs]
        if run_mode == "multirun"
        else [(source.name, source)]
    )
    if not roots:
        raise ValueError(f"multirun has no completed job outputs: {source}")
    validated: list[dict[str, Any]] = []
    for job_name, root in roots:
        missing = [
            relative
            for relative in REQUIRED_JOB_FILES
            if not (root / relative).is_file()
        ]
        if missing:
            raise ValueError(
                f"run job={job_name!r} is incomplete; missing required artifacts={missing}"
            )
        summary = _read_yaml_mapping(root / "workflow" / "summary.yaml")
        output_schema_version = summary.get("output_schema_version", 1)
        if not isinstance(output_schema_version, int) or isinstance(
            output_schema_version, bool
        ):
            raise ValueError(f"run job={job_name!r} has invalid output_schema_version")
        if output_schema_version >= 2:
            missing_v2 = [
                relative
                for relative in REQUIRED_SCHEMA_V2_JOB_FILES
                if not (root / relative).is_file()
            ]
            if missing_v2:
                raise ValueError(
                    f"run job={job_name!r} is incomplete for output schema "
                    f"v{output_schema_version}; missing={missing_v2}"
                )
        for key in ("method", "dataset", "profile", "seed"):
            if key not in summary:
                raise ValueError(f"run job={job_name!r} summary is missing {key!r}")
        assets = _read_yaml_mapping(root / "assets" / "manifest.yaml")
        if not isinstance(assets.get("assets"), list) or not assets["assets"]:
            raise ValueError(f"run job={job_name!r} has no scientific asset references")
        with (root / "metrics" / "final.metrics.csv").open(
            encoding="utf-8", newline=""
        ) as stream:
            final_rows = list(csv.DictReader(stream))
        if len(final_rows) != 1:
            raise ValueError(
                f"run job={job_name!r} requires exactly one final metric row"
            )
        if final_rows[0].get("Method") != summary["method"]:
            raise ValueError(
                f"run job={job_name!r} summary/final metric methods do not match"
            )
        task_ids: list[str] = []
        per_task_path = root / "metrics" / "per_task.jsonl"
        with per_task_path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"invalid per-task JSON: {per_task_path}:{line_number}"
                    ) from error
                task_id = record.get("task_id") if isinstance(record, dict) else None
                if not isinstance(task_id, str) or not task_id:
                    raise ValueError(
                        f"per-task row lacks task_id: {per_task_path}:{line_number}"
                    )
                task_ids.append(task_id)
        if not task_ids:
            raise ValueError(f"run job={job_name!r} has no per-task metric rows")
        if len(task_ids) != len(set(task_ids)):
            raise ValueError(f"run job={job_name!r} has duplicate per-task task IDs")
        if (
            expected_per_task_count is not None
            and len(task_ids) != expected_per_task_count
        ):
            raise ValueError(
                f"run job={job_name!r} expected {expected_per_task_count} per-task "
                f"rows, observed {len(task_ids)}"
            )
        validated.append(
            {
                "job": job_name,
                "method": summary["method"],
                "variant": summary.get("variant"),
                "dataset": summary["dataset"],
                "profile": summary["profile"],
                "seed": summary["seed"],
                "output_schema_version": output_schema_version,
                "per_task_count": len(task_ids),
                "task_id_sha256": _sha256_text("\n".join(sorted(task_ids)) + "\n"),
            }
        )
    return validated


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected YAML mapping: {path}")
    return value


def _is_required_job_file(
    relative_path: str,
    *,
    run_mode: str,
    jobs: Sequence[str],
) -> bool:
    required_files = (*REQUIRED_JOB_FILES, *REQUIRED_SCHEMA_V2_JOB_FILES)
    if run_mode == "single":
        return relative_path in required_files
    return any(
        relative_path == f"{job}/{required}"
        for job in jobs
        for required in required_files
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


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

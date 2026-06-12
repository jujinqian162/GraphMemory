from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

DEFAULT_MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024


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
    copied: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    total_copied_bytes = 0

    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        relative_path = path.relative_to(source).as_posix()
        size_bytes = path.stat().st_size
        decision = _classify_run_file(relative_path, size_bytes, max_file_size_bytes)
        entry = {"relative_path": relative_path, "size_bytes": size_bytes}
        if decision == "copy":
            copied.append(entry)
            total_copied_bytes += size_bytes
            if not dry_run:
                destination = output_dir / relative_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)
        else:
            skipped.append({**entry, "reason": decision})

    if include_report:
        report_source = Path(report_dir)
        for path in sorted(report_source.rglob("*")) if report_source.exists() else []:
            if not path.is_file():
                continue
            relative_path = f"report/{path.relative_to(report_source).as_posix()}"
            size_bytes = path.stat().st_size
            if size_bytes > max_file_size_bytes:
                skipped.append({"relative_path": relative_path, "size_bytes": size_bytes, "reason": "too_large"})
                continue
            copied.append({"relative_path": relative_path, "size_bytes": size_bytes})
            total_copied_bytes += size_bytes
            if not dry_run:
                destination = output_dir / relative_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)

    manifest = {
        "source_run_dir": str(source.resolve()),
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
    if not dry_run:
        _write_json(output_dir / "delivery_manifest.json", manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    run_dir = Path("runs") / args.name
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
            "Collect compact report-relevant artifacts for transfer. "
            "Contract: --name <train_id> reads runs/<name> and writes results/<name> by default."
        )
    )
    parser.add_argument("--name", required=True, help="Run name under runs/, e.g. rgcn_full_train.")
    parser.add_argument("--output-root", default="results", help="Destination root; run name is appended.")
    parser.add_argument(
        "--max-file-size-mb",
        type=float,
        default=DEFAULT_MAX_FILE_SIZE_BYTES / (1024 * 1024),
        help="Maximum size for files that match include rules.",
    )
    parser.add_argument("--include-report", action="store_true", help="Also copy files from --report-dir under report/.")
    parser.add_argument("--report-dir", default="report", help="Report directory used with --include-report.")
    parser.add_argument("--dry-run", action="store_true", help="Print manifest summary without copying files.")
    return parser


def _classify_run_file(relative_path: str, size_bytes: int, max_file_size_bytes: int) -> str:
    path = Path(relative_path)
    parts = path.parts
    name = path.name

    exclusion = _known_exclusion(parts, name)
    if exclusion is not None:
        return exclusion
    if not _is_included(parts, name, relative_path):
        return "not_selected"
    if size_bytes > max_file_size_bytes:
        return "too_large"
    return "copy"


def _known_exclusion(parts: tuple[str, ...], name: str) -> str | None:
    if not parts:
        return "not_selected"
    top = parts[0]
    if top == "inputs":
        return "excluded_input"
    if top == "graphs" and name.endswith(".graphs.json"):
        return "excluded_graph"
    if (top == "predictions" or "predictions" in parts) and name.endswith(".ranked.json"):
        return "excluded_prediction"
    if "checkpoints" in parts or Path(name).suffix in {".pt", ".ckpt"}:
        return "excluded_checkpoint"
    if name == "train.pairs.json":
        return "excluded_train_pairs"
    if name.endswith(".dev_selected.candidates.json"):
        return "excluded_tuning_candidates"
    if Path(name).suffix in {".bin", ".safetensors", ".npy", ".npz"}:
        return "excluded_model_or_embedding"
    return None


def _is_included(parts: tuple[str, ...], name: str, relative_path: str) -> bool:
    if relative_path == "manifest.json":
        return True
    if name.endswith(".run_summary.json"):
        return True
    if _is_included_ablation_file(parts, name):
        return True
    if not parts:
        return False
    top = parts[0]
    if top == "config" and name.endswith(".json"):
        return True
    if top == "tables":
        return True
    if top == "metrics" and name.endswith(".csv"):
        return True
    if top == "graphs" and name.endswith(".stats.json"):
        return True
    if top == "debug" and name.startswith("failure_cases_") and name.endswith(".jsonl"):
        return True
    if top == "tuned" and name.endswith(".dev_selected.json"):
        return True
    if top == "learned" and name in {
        "effective_method_config.json",
        "train_metrics.jsonl",
        "train_run_summary.json",
        "train.pairs.summary.json",
        "train.pairs.run_summary.json",
    }:
        return True
    return False


def _is_included_ablation_file(parts: tuple[str, ...], name: str) -> bool:
    if len(parts) < 4 or parts[0] != "ablations":
        return False
    if name in {
        "effective_method_config.json",
        "train_metrics.jsonl",
        "train_run_summary.json",
        "train.pairs.summary.json",
        "train.pairs.run_summary.json",
    }:
        return True
    if "metrics" in parts and name.endswith(".csv"):
        return True
    if "debug" in parts and name.startswith("failure_cases") and name.endswith(".jsonl"):
        return True
    return False


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

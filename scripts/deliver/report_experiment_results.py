from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeGuard, cast

import yaml

DEFAULT_METRICS = (
    "Recall@5",
    "MRR",
    "Full Support@2048 Tokens",
    "Coverage@2048 Tokens",
    "Full Support Budget-AUC",
)
METHOD_NAME_TOKENS = {
    "bm25": "bm25",
    "dense": "dense",
    "dense_ft": "dft",
    "cross_encoder": "ce",
    "graphrag": "graphrag",
    "provenance_path": "ppath",
    "provenance_rgcn": "rgcn",
    "dense_rgcn_graph_retriever": "dense_rgcn",
    "dense_ft_rgcn_graph_retriever": "dft_rgcn",
}
VARIANT_NAME_TOKENS = {
    "flat": "flat",
    "provenance_unit": "pu",
    "full_rgcn": "full",
    "wo_graph": "nograph",
    "homogeneous_gcn": "homog",
    "wo_feeds": "nofeeds",
    "wo_execution_ownership": "noexec",
    "wo_artifact_io": "noio",
    "wo_chunk_adjacency": "nochunk",
    "random_edges": "randedge",
    "wo_hard_negatives": "nohardneg",
}


@dataclass(frozen=True)
class RunDiagnostic:
    label: str
    seed: int
    selected_epoch: int | None
    best_dev_metric: float | None
    physical_edges: int | None
    active_edges: int | None
    rewired_edges: int | None
    changed_graphs: int | None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Render aggregate_main_results.py JSON as a readable Markdown report. "
            "Optional run discovery adds checkpoint-selection and graph-control diagnostics."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--title", default="Experiment Results")
    parser.add_argument(
        "--metrics",
        default=",".join(DEFAULT_METRICS),
        help="Comma-separated metric names. Defaults to the compact retrieval set.",
    )
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        help="Run directory or glob used for diagnostics (repeatable).",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("runs"),
        help="Root used with --name-prefix (default: runs).",
    )
    parser.add_argument(
        "--name-prefix",
        help=(
            "Discover diagnostic runs whose canonical run name starts with this prefix, "
            "for example isetrace_v7_e2rel_."
        ),
    )
    parser.add_argument("--digits", type=int, default=2)
    parser.add_argument(
        "--raw-values",
        action="store_true",
        help="Do not convert unit-interval metrics and deltas to percentages/percentage points.",
    )
    args = parser.parse_args(argv)

    result = _read_mapping(args.input)
    metrics = tuple(value.strip() for value in args.metrics.split(",") if value.strip())
    if not metrics:
        raise ValueError("--metrics must select at least one metric")
    run_dirs = _discover_runs(
        args.run,
        run_root=args.run_root,
        name_prefix=args.name_prefix,
    )
    diagnostics = _load_run_diagnostics(run_dirs, result=result)
    report = render_report(
        result,
        title=args.title,
        metrics=metrics,
        diagnostics=diagnostics,
        digits=args.digits,
        percentage=not args.raw_values,
    )
    if args.output is None:
        print(report, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8", newline="\n")
        print(f"report={args.output} runs={len(run_dirs)}", file=sys.stderr)
    return 0


def render_report(
    result: Mapping[str, object],
    *,
    title: str,
    metrics: Sequence[str] = DEFAULT_METRICS,
    diagnostics: Sequence[RunDiagnostic] = (),
    digits: int = 2,
    percentage: bool = True,
) -> str:
    if digits < 0:
        raise ValueError("digits must be non-negative")
    summary = _mapping(result.get("summary"), path="summary")
    baseline = _string(result.get("baseline_method"), path="baseline_method")
    direction = _string(result.get("delta_direction"), path="delta_direction")
    selected_metrics = _shared_selected_metrics(summary, metrics)
    if not selected_metrics:
        raise ValueError("None of the selected metrics occur in every summary row.")

    lines = [f"# {title}", ""]
    lines.extend(
        (
            "## Scope",
            "",
            f"- Baseline: `{baseline}`",
            f"- Delta direction: `{direction}`",
            f"- Test tasks: {_integer(result.get('test_task_count'), path='test_task_count'):,}",
            f"- Test clusters: {_integer(result.get('test_cluster_count'), path='test_cluster_count'):,}",
            f"- Cluster unit: `{result.get('cluster_unit', 'unknown')}`",
            "",
        )
    )
    digest = result.get("test_artifact_digest")
    if isinstance(digest, str) and digest:
        lines.insert(len(lines) - 1, f"- Test artifact: `{digest}`")

    lines.extend(("## Main metrics", ""))
    main_headers = ["Method", "Seeds", *selected_metrics]
    main_rows: list[list[str]] = []
    for method in _ordered_methods(summary, baseline=baseline):
        record = _mapping(summary[method], path=f"summary.{method}")
        seeds = record.get("seeds", [])
        seed_text = (
            ", ".join(str(seed) for seed in seeds) if isinstance(seeds, list) else ""
        )
        values = _mapping(record.get("metrics"), path=f"summary.{method}.metrics")
        main_rows.append(
            [
                f"**{method}**" if method == baseline else method,
                seed_text,
                *[
                    _format_summary_metric(
                        _mapping(
                            values[metric], path=f"summary.{method}.metrics.{metric}"
                        ),
                        digits=digits,
                        percentage=percentage,
                    )
                    for metric in selected_metrics
                ],
            ]
        )
    lines.extend((*_markdown_table(main_headers, main_rows), ""))

    paired_value = result.get("paired_analysis", {})
    paired = _mapping(paired_value, path="paired_analysis")
    if paired:
        lines.extend(("## Paired effects", ""))
        paired_headers = ["Method", *selected_metrics]
        paired_rows: list[list[str]] = []
        for method in sorted(key for key in paired if isinstance(key, str)):
            method_metrics = _mapping(
                _mapping(paired[method], path=f"paired_analysis.{method}").get(
                    "metrics"
                ),
                path=f"paired_analysis.{method}.metrics",
            )
            paired_rows.append(
                [
                    method,
                    *[
                        _format_effect(
                            _mapping(
                                method_metrics[metric],
                                path=f"paired_analysis.{method}.metrics.{metric}",
                            ),
                            digits=digits,
                            percentage=percentage,
                        )
                        for metric in selected_metrics
                    ],
                ]
            )
        lines.extend((*_markdown_table(paired_headers, paired_rows), ""))
        lines.append(
            "`↑` means the reported delta is significantly positive, `↓` significantly "
            "negative, and `≈` has a 95% interval crossing zero."
        )
        lines.append("")

    stratified_value = result.get("stratified_paired_analysis", {})
    stratified = _mapping(stratified_value, path="stratified_paired_analysis")
    if stratified:
        lines.extend(("## Stratified paired effects", ""))
        strata = sorted(
            {
                stratum
                for method_value in stratified.values()
                for stratum in _mapping(method_value, path="stratified method")
                if isinstance(stratum, str)
            }
        )
        stratified_headers = ["Method", "Stratum", *selected_metrics]
        stratified_rows: list[list[str]] = []
        for method in sorted(key for key in stratified if isinstance(key, str)):
            method_strata = _mapping(stratified[method], path=f"stratified.{method}")
            for stratum in strata:
                if stratum not in method_strata:
                    continue
                stratum_metrics = _mapping(
                    _mapping(
                        method_strata[stratum], path=f"stratified.{method}.{stratum}"
                    ).get("metrics"),
                    path=f"stratified.{method}.{stratum}.metrics",
                )
                stratified_rows.append(
                    [
                        method,
                        stratum,
                        *[
                            _format_effect(
                                _mapping(
                                    stratum_metrics[metric],
                                    path=(
                                        f"stratified.{method}.{stratum}.metrics.{metric}"
                                    ),
                                ),
                                digits=digits,
                                percentage=percentage,
                            )
                            for metric in selected_metrics
                        ],
                    ]
                )
        lines.extend((*_markdown_table(stratified_headers, stratified_rows), ""))

    if diagnostics:
        lines.extend(("## Training and control diagnostics", ""))
        diagnostic_rows = [
            [
                item.label,
                str(item.seed),
                "—" if item.selected_epoch is None else str(item.selected_epoch),
                _format_optional(item.best_dev_metric, digits=4),
                _format_optional_integer(item.physical_edges),
                _format_optional_integer(item.active_edges),
                _format_optional_integer(item.rewired_edges),
                _format_optional_integer(item.changed_graphs),
            ]
            for item in sorted(diagnostics, key=lambda value: (value.label, value.seed))
        ]
        lines.extend(
            (
                *_markdown_table(
                    [
                        "Method/variant",
                        "Seed",
                        "Selected epoch",
                        "Best dev",
                        "Physical edges",
                        "Active edges",
                        "Rewired edges",
                        "Changed graphs",
                    ],
                    diagnostic_rows,
                ),
                "",
            )
        )
        fallback = _epoch_zero_summary(diagnostics)
        if fallback:
            lines.extend(("### Automatic observations", ""))
            lines.extend(f"- {message}" for message in fallback)
            lines.append("")

    unit = (
        "percent; paired deltas are percentage points" if percentage else "raw values"
    )
    lines.extend(("## Formatting", "", f"Values are reported as {unit}.", ""))
    return "\n".join(lines)


def _discover_runs(
    patterns: Sequence[str],
    *,
    run_root: Path,
    name_prefix: str | None,
) -> list[Path]:
    candidates: set[Path] = set()
    for pattern in patterns:
        matches = [Path(value) for value in glob.glob(pattern)]
        if not matches and Path(pattern).exists():
            matches = [Path(pattern)]
        if not matches:
            raise FileNotFoundError(f"run pattern matched nothing: {pattern}")
        candidates.update(path.resolve() for path in matches if path.is_dir())
    if name_prefix is not None:
        if not name_prefix or Path(name_prefix).name != name_prefix:
            raise ValueError("--name-prefix must be one path-free run-name prefix")
        candidates.update(
            path.resolve() for path in run_root.glob(f"{name_prefix}*") if path.is_dir()
        )
    return sorted(candidates)


def _load_run_diagnostics(
    run_dirs: Sequence[Path],
    *,
    result: Mapping[str, object],
) -> list[RunDiagnostic]:
    if not run_dirs:
        return []
    summary = _mapping(result.get("summary"), path="summary")
    labels = {key for key in summary if isinstance(key, str)}
    diagnostics: list[RunDiagnostic] = []
    seen: set[tuple[str, int]] = set()
    for run in run_dirs:
        run_summary = _read_mapping(run / "workflow" / "summary.yaml")
        method = _string(run_summary.get("method"), path=f"{run}.method")
        variant_value = run_summary.get("variant")
        variant = (
            variant_value if isinstance(variant_value, str) and variant_value else None
        )
        seed = _integer(run_summary.get("seed"), path=f"{run}.seed")
        label = _match_label(labels, method=method, variant=variant, run_name=run.name)
        if label is None:
            continue
        identity = (label, seed)
        if identity in seen:
            raise ValueError(
                f"duplicate diagnostic run for label={label!r}, seed={seed}"
            )
        seen.add(identity)
        selected_epoch, best_dev_metric = _training_selection(run)
        control = _control_diagnostic(run)
        diagnostics.append(
            RunDiagnostic(
                label=label,
                seed=seed,
                selected_epoch=selected_epoch,
                best_dev_metric=best_dev_metric,
                physical_edges=_optional_integer(control.get("physical_edge_count")),
                active_edges=_optional_integer(control.get("active_edge_count")),
                rewired_edges=_optional_integer(control.get("rewired_edge_count")),
                changed_graphs=_optional_integer(control.get("changed_graph_count")),
            )
        )
    return diagnostics


def _training_selection(run: Path) -> tuple[int | None, float | None]:
    path = run / "training" / "train_metrics.jsonl"
    if not path.is_file():
        return None, None
    rows: list[Mapping[str, object]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError(f"invalid training metric row: {path}:{line_number}")
            rows.append(cast(Mapping[str, object], value))
    if not rows:
        return None, None
    first = rows[0]
    initial = _optional_number(first.get("initial_selection_metric_value"))
    best = _optional_number(rows[-1].get("best_dev_metric"))
    if best is None:
        candidates = [
            value
            for row in rows
            if (value := _optional_number(row.get("selection_metric_value")))
            is not None
        ]
        best = max(candidates) if candidates else initial
    if initial is not None and best is not None and math.isclose(best, initial):
        return 0, best
    for row in rows:
        value = _optional_number(row.get("selection_metric_value"))
        epoch = _optional_integer(row.get("epoch"))
        if (
            value is not None
            and best is not None
            and epoch is not None
            and math.isclose(value, best)
        ):
            return epoch, best
    return None, best


def _control_diagnostic(run: Path) -> Mapping[str, object]:
    path = run / "training" / "control_diagnostics.json"
    if not path.is_file():
        return {}
    value = _read_mapping(path)
    train = value.get("train", {})
    return cast(Mapping[str, object], train) if isinstance(train, Mapping) else {}


def _match_label(
    labels: set[str],
    *,
    method: str,
    variant: str | None,
    run_name: str,
) -> str | None:
    method_token = METHOD_NAME_TOKENS.get(method, method)
    variant_token = (
        None if variant is None else VARIANT_NAME_TOKENS.get(variant, variant)
    )
    candidates = [
        variant,
        f"{method}:{variant}" if variant else None,
        method,
        variant_token,
        f"{method_token}_{variant_token}" if variant_token else None,
        method_token,
    ]
    exact = list(
        dict.fromkeys(candidate for candidate in candidates if candidate in labels)
    )
    if len(exact) == 1:
        return exact[0]
    stem, separator, raw_seed = run_name.rpartition("_s")
    seed_suffix = separator != "" and raw_seed.isdigit()
    suffix_matches = [
        label
        for label in labels
        if run_name.endswith(f"_{label}")
        or (seed_suffix and stem.endswith(f"_{label}"))
    ]
    return suffix_matches[0] if len(suffix_matches) == 1 else None


def _shared_selected_metrics(
    summary: Mapping[str, object], metrics: Sequence[str]
) -> tuple[str, ...]:
    metric_sets = []
    for method, value in summary.items():
        record = _mapping(value, path=f"summary.{method}")
        metric_sets.append(set(_mapping(record.get("metrics"), path="metrics")))
    shared = set.intersection(*metric_sets) if metric_sets else set()
    return tuple(metric for metric in metrics if metric in shared)


def _ordered_methods(summary: Mapping[str, object], *, baseline: str) -> list[str]:
    methods = sorted(key for key in summary if isinstance(key, str))
    return [baseline, *(method for method in methods if method != baseline)]


def _format_summary_metric(
    value: Mapping[str, object], *, digits: int, percentage: bool
) -> str:
    mean = _optional_number(value.get("mean"))
    direct = _optional_number(value.get("value"))
    number = mean if mean is not None else direct
    if number is None:
        return "—"
    scale = 100.0 if percentage else 1.0
    rendered = f"{number * scale:.{digits}f}"
    std = _optional_number(value.get("std"))
    return rendered if std is None else f"{rendered} ± {std * scale:.{digits}f}"


def _format_effect(
    value: Mapping[str, object], *, digits: int, percentage: bool
) -> str:
    delta = _number(value.get("mean_delta"), path="mean_delta")
    interval = value.get("ci_95")
    if not isinstance(interval, list) or len(interval) != 2:
        raise ValueError("ci_95 must contain [lower, upper]")
    lower = _number(interval[0], path="ci_95[0]")
    upper = _number(interval[1], path="ci_95[1]")
    marker = "↑" if lower > 0 else "↓" if upper < 0 else "≈"
    scale = 100.0 if percentage else 1.0
    return (
        f"{delta * scale:+.{digits}f} "
        f"[{lower * scale:+.{digits}f}, {upper * scale:+.{digits}f}] {marker}"
    )


def _epoch_zero_summary(diagnostics: Sequence[RunDiagnostic]) -> list[str]:
    by_label: dict[str, list[RunDiagnostic]] = defaultdict(list)
    for item in diagnostics:
        by_label[item.label].append(item)
    messages = []
    for label, values in sorted(by_label.items()):
        selected = [
            value.selected_epoch for value in values if value.selected_epoch is not None
        ]
        if selected and all(epoch == 0 for epoch in selected):
            messages.append(
                f"`{label}` selected the initial epoch-0 checkpoint for all "
                f"{len(selected)} observed seeds."
            )
    return messages


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    escaped_headers = [_escape_cell(value) for value in headers]
    lines = [
        "| " + " | ".join(escaped_headers) + " |",
        "| " + " | ".join("---" for _ in escaped_headers) + " |",
    ]
    for row in rows:
        if len(row) != len(headers):
            raise ValueError("Markdown table row width does not match headers")
        lines.append("| " + " | ".join(_escape_cell(value) for value in row) + " |")
    return lines


def _escape_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _format_optional(value: float | None, *, digits: int) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _format_optional_integer(value: int | None) -> str:
    return "—" if value is None else f"{value:,}"


def _read_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as stream:
        if path.suffix in {".yaml", ".yml"}:
            value = yaml.safe_load(stream)
        else:
            value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected mapping: {path}")
    return value


def _mapping(value: object, *, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping")
    return cast(Mapping[str, object], value)


def _string(value: object, *, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path} must be a non-empty string")
    return value


def _integer(value: object, *, path: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{path} must be an integer")
    return value


def _optional_integer(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _is_number(value: object) -> TypeGuard[int | float]:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _number(value: object, *, path: str) -> float:
    if not _is_number(value):
        raise ValueError(f"{path} must be numeric")
    return float(value)


def _optional_number(value: object) -> float | None:
    return float(value) if _is_number(value) else None


if __name__ == "__main__":
    raise SystemExit(main())

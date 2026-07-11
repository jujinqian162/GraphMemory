from __future__ import annotations

import logging
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from pydantic import TypeAdapter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graph_memory.evaluation.tables import (
    EFFICIENCY_RESULT_COLUMNS,
    MAIN_RESULT_COLUMNS,
    PATH_RESULT_COLUMNS,
    split_metric_tables,
)
from graph_memory.contracts.metrics import MetricRow
from graph_memory.experiment.persistence import read_yaml
from graph_memory.experiment.stage_cli import load_stage_execution
from graph_memory.experiment.stage_models import (
    AblationAggregateStageConfig,
    AblationSelection,
    AggregateStageConfig,
)
from graph_memory.experiment.state import stage_lifecycle
from graph_memory.io import read_csv, write_csv

LOGGER = logging.getLogger("aggregate_tables")
ABLATION_RESULT_COLUMNS = [
    "Method",
    "Variant",
    "Recall@5",
    "Full Support@5",
    "Connected Evidence Recall@10",
    "Path Recall@10",
    "Retrieval Latency / Query",
]


def main(argv: Sequence[str] | None = None) -> int:
    execution = load_stage_execution(
        None if argv is None else list(argv),
        TypeAdapter(AggregateStageConfig),
        description="Aggregate experiment metrics from a resolved stage YAML.",
        script=Path(__file__),
    )
    config = execution.config
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s"
    )

    start_time = time.perf_counter()
    with stage_lifecycle(execution.invocation) as observations:
        metric_files = config.metrics
        rows: list[MetricRow] = []
        for metric_file in metric_files:
            rows.extend(cast(list[MetricRow], read_csv(metric_file)))
        main_rows, path_rows, efficiency_rows = split_metric_tables(rows)
        write_csv(config.main, main_rows, MAIN_RESULT_COLUMNS)
        write_csv(config.path, path_rows, PATH_RESULT_COLUMNS)
        write_csv(config.efficiency, efficiency_rows, EFFICIENCY_RESULT_COLUMNS)
        if isinstance(config, AblationAggregateStageConfig):
            ablation_rows = _indexed_ablation_rows(
                config.ablation_index,
                config.ablation_selections,
            )
            write_csv(config.ablation, ablation_rows, ABLATION_RESULT_COLUMNS)

        observations.count("metric_files", len(metric_files))
        observations.count("rows", len(rows))
        observations.count("included", [str(path) for path in metric_files])
        observations.timing("total_seconds", time.perf_counter() - start_time)
        LOGGER.info("wrote main results: %s", config.main)
        LOGGER.info("wrote path results: %s", config.path)
        LOGGER.info("wrote efficiency results: %s", config.efficiency)
        if isinstance(config, AblationAggregateStageConfig):
            LOGGER.info("wrote ablation results: %s", config.ablation)
    return 0


def _indexed_ablation_rows(
    index_path: Path,
    selections: Sequence[AblationSelection],
) -> list[dict[str, object]]:
    index = read_yaml(index_path)
    entries = index.get("metrics") if isinstance(index, dict) else None
    if not isinstance(index, dict) or not isinstance(entries, list):
        raise ValueError(
            f"Ablation metric index must contain a metrics list: {index_path}"
        )
    unsupported_fields = sorted(set(index) - {"metrics"})
    if unsupported_fields:
        fields = ", ".join(unsupported_fields)
        raise ValueError(f"Ablation metric index has unsupported fields: {fields}")
    requested = {(value.method.value, value.variant) for value in selections}
    matched: set[tuple[str, str]] = set()
    rows: list[dict[str, object]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError(
                f"Ablation metric index entries must be objects: {index_path}"
            )
        method = entry.get("method")
        variant = entry.get("variant")
        metrics_path = entry.get("metrics_path")
        if (
            not isinstance(method, str)
            or not method
            or not isinstance(variant, str)
            or not variant
            or not isinstance(metrics_path, str)
            or not metrics_path
        ):
            raise ValueError(
                f"Ablation metric index entry requires method, variant, and metrics_path: {entry}"
            )
        selection = (method, variant)
        if requested and selection not in requested:
            continue
        matched.add(selection)
        metric_rows = read_csv(metrics_path)
        if len(metric_rows) != 1:
            raise ValueError(
                f"Expected one metric row for ablation method={method} variant={variant}: {metrics_path}"
            )
        metric_row = metric_rows[0]
        rows.append(
            {
                "Method": method,
                "Variant": variant,
                **{
                    column: metric_row[column] for column in ABLATION_RESULT_COLUMNS[2:]
                },
            }
        )
    missing = sorted(requested - matched)
    if missing:
        values = ", ".join(f"{method}={variant}" for method, variant in missing)
        raise ValueError(
            f"Ablation selections are missing from the metric index: {values}"
        )
    return rows


if __name__ == "__main__":
    raise SystemExit(main())

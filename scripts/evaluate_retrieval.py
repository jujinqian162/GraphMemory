from __future__ import annotations

import logging
import sys
import time
from collections.abc import Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graph_memory.experiment.stage_cli import load_stage_execution
from graph_memory.experiment.stage_models import EvaluateStageConfig
from graph_memory.experiment.state import stage_lifecycle
from graph_memory.evaluation.tables import WIDE_METRIC_COLUMNS
from graph_memory.io import read_json, write_csv, write_jsonl
from graph_memory.stages.evaluate import run_evaluate_stage
from graph_memory.validation import validate_metric_rows

LOGGER = logging.getLogger("evaluate_retrieval")


def main(argv: Sequence[str] | None = None) -> int:
    execution = load_stage_execution(
        None if argv is None else list(argv),
        EvaluateStageConfig,
        description="Evaluate retrieval from a resolved stage YAML.",
        script=Path(__file__),
    )
    config = execution.config
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s"
    )

    start_time = time.perf_counter()
    with stage_lifecycle(execution.invocation) as observations:
        predictions = read_json(config.predictions)
        labels = read_json(config.labels)
        graphs = (
            read_json(config.evidence_graphs)
            if config.evidence_graphs is not None
            else []
        )
        result = run_evaluate_stage(
            config,
            predictions=predictions,
            labels=labels,
            graphs=graphs,
        )
        validate_metric_rows(result.metric_rows)
        write_csv(config.metrics, result.metric_rows, WIDE_METRIC_COLUMNS)

        write_jsonl(config.failure_cases, result.failure_cases)

        observations.count("predictions", len(predictions))
        observations.count("metric_rows", len(result.metric_rows))
        observations.count("failure_cases", len(result.failure_cases))
        observations.timing("total_seconds", time.perf_counter() - start_time)
        LOGGER.info("wrote metrics: %s", config.metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

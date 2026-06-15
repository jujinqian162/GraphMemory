from __future__ import annotations

import argparse
import hashlib
import logging
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graph_memory.contracts.graphs import MemoryGraph
from graph_memory.contracts.common import JsonValue
from graph_memory.contracts.tasks import MemoryTaskInput, MemoryTaskLabels
from graph_memory.io import read_json, write_json
from graph_memory.observability import (
    build_run_summary,
    collect_environment,
    now_iso,
    write_run_summary,
)
from graph_memory.retrieval.methods.flat.dense import DenseConfig
from graph_memory.retrieval.methods.memory_stream.contracts import (
    ImportanceArtifact,
)
from graph_memory.retrieval.requests import DenseRuntime
from graph_memory.retrieval.tuning import (
    memory_stream_grid_from_record,
    tune_memory_stream,
)
from graph_memory.validation import (
    select_importance_records,
    validate_graphs,
    validate_memory_task_inputs,
    validate_memory_task_labels,
)

LOGGER = logging.getLogger("tune_memory_stream")
DEFAULT_GRID_CONFIG = "configs/search_spaces/memory_stream.json"


@dataclass(frozen=True)
class TuneMemoryStreamArgs:
    tasks: str
    labels: str
    graphs: str
    importance: str
    output_config: str
    encoder_model: str
    query_prefix: str
    passage_prefix: str
    top_k: int
    grid_config: str


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s [%(name)s] %(message)s",
    )

    started_at = now_iso()
    start_time = time.perf_counter()
    output_config_path = Path(args.output_config)
    summary_path = output_config_path.with_name(
        f"{output_config_path.stem}.run_summary.json"
    )
    candidates_path = output_config_path.with_name(
        f"{output_config_path.stem}.candidates.json"
    )
    dense_config = DenseConfig(
        model_name=args.encoder_model,
        query_prefix=args.query_prefix,
        passage_prefix=args.passage_prefix,
    )
    effective_config: dict[str, JsonValue] = {
        "encoder_model": args.encoder_model,
        "query_prefix": args.query_prefix,
        "passage_prefix": args.passage_prefix,
        "batch_size": dense_config.batch_size,
        "top_k": args.top_k,
        "grid_config": args.grid_config,
    }
    inputs = {
        "tasks": args.tasks,
        "labels": args.labels,
        "graphs": args.graphs,
        "importance": args.importance,
        "grid_config": args.grid_config,
    }
    outputs = {
        "selected_config": args.output_config,
        "candidate_rows": str(candidates_path),
        "run_summary": str(summary_path),
    }

    try:
        task_inputs = cast(list[MemoryTaskInput], read_json(args.tasks))
        labels = cast(list[MemoryTaskLabels], read_json(args.labels))
        graphs = cast(list[MemoryGraph], read_json(args.graphs))
        importance_bytes = Path(args.importance).read_bytes()
        importance_sha256 = hashlib.sha256(importance_bytes).hexdigest()
        importance_artifact = cast(
            ImportanceArtifact,
            read_json(args.importance),
        )
        grid = memory_stream_grid_from_record(
            cast(Mapping[str, object], read_json(args.grid_config))
        )

        validate_memory_task_inputs(task_inputs)
        inputs_by_task_id = {
            task_input["task_id"]: task_input for task_input in task_inputs
        }
        validate_memory_task_labels(labels, inputs_by_task_id)
        validate_graphs(graphs, inputs_by_task_id)
        _ = select_importance_records(importance_artifact, task_inputs)

        effective_config["importance_sha256"] = importance_sha256
        selected_config, candidate_rows = tune_memory_stream(
            task_inputs=task_inputs,
            labels=labels,
            graphs=graphs,
            importance_artifact=importance_artifact,
            grid=grid,
            top_k=args.top_k,
            dense_runtime=DenseRuntime(config=dense_config),
        )
        write_json(output_config_path, selected_config)
        write_json(candidates_path, candidate_rows)
        effective_config["selected_scoring_config"] = cast(
            JsonValue,
            cast(object, selected_config),
        )

        summary = build_run_summary(
            script="tune_memory_stream.py",
            started_at=started_at,
            finished_at=now_iso(),
            status="success",
            effective_config=effective_config,
            inputs=inputs,
            outputs=outputs,
            counts={
                "tasks": len(task_inputs),
                "grid_size": len(grid),
                "candidate_rows": len(candidate_rows),
            },
            timings={"total_seconds": time.perf_counter() - start_time},
            environment=collect_environment(),
            notes=[],
        )
        write_run_summary(summary_path, summary)
        LOGGER.info("selected config: %s", selected_config)
        LOGGER.info("wrote selected config: %s", output_config_path)
        return 0
    except Exception as error:
        LOGGER.error("%s", error)
        summary = build_run_summary(
            script="tune_memory_stream.py",
            started_at=started_at,
            finished_at=now_iso(),
            status="failed",
            effective_config=effective_config,
            inputs=inputs,
            outputs=outputs,
            counts={},
            timings={"total_seconds": time.perf_counter() - start_time},
            environment=collect_environment(),
            notes=[],
            error=str(error),
        )
        write_run_summary(summary_path, summary)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tune Memory Stream scoring parameters on dev labels."
    )
    _ = parser.add_argument(
        "--tasks",
        required=True,
        help="Path to dev *_memory_tasks.input.json.",
    )
    _ = parser.add_argument(
        "--labels",
        required=True,
        help="Path to dev *_memory_tasks.labels.json.",
    )
    _ = parser.add_argument(
        "--graphs",
        required=True,
        help="Path to dev *_graphs.json.",
    )
    _ = parser.add_argument(
        "--importance",
        required=True,
        help="Path to the aligned cleaned importance artifact.",
    )
    _ = parser.add_argument(
        "--output_config",
        required=True,
        help="Path to write selected Memory Stream scoring config JSON.",
    )
    _ = parser.add_argument(
        "--encoder_model",
        default="intfloat/e5-base-v2",
    )
    _ = parser.add_argument("--query_prefix", default="query: ")
    _ = parser.add_argument("--passage_prefix", default="passage: ")
    _ = parser.add_argument("--top_k", type=int, default=10)
    _ = parser.add_argument("--grid_config", default=DEFAULT_GRID_CONFIG)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> TuneMemoryStreamArgs:
    namespace = build_parser().parse_args(argv)
    return TuneMemoryStreamArgs(
        tasks=cast(str, namespace.tasks),
        labels=cast(str, namespace.labels),
        graphs=cast(str, namespace.graphs),
        importance=cast(str, namespace.importance),
        output_config=cast(str, namespace.output_config),
        encoder_model=cast(str, namespace.encoder_model),
        query_prefix=cast(str, namespace.query_prefix),
        passage_prefix=cast(str, namespace.passage_prefix),
        top_k=cast(int, namespace.top_k),
        grid_config=cast(str, namespace.grid_config),
    )


if __name__ == "__main__":
    raise SystemExit(main())

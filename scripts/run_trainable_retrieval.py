from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graph_memory.application.run_retrieval import RunRetrievalRequest, run_retrieval
from graph_memory.io import read_json, write_json
from graph_memory.models.graph_retriever.contracts import TextEmbeddingProvider
from graph_memory.observability import build_run_summary, collect_environment, now_iso, write_run_summary
from graph_memory.retrieval.requests import TrainableGraphRuntime
from graph_memory.retrieval.signals import SeedSignalProvider
from graph_memory.validation import validate_memory_task_inputs, validate_ranked_results

LOGGER = logging.getLogger("run_trainable_retrieval")


@dataclass(frozen=True)
class RunTrainableRetrievalArgs:
    """
    Parsed CLI arguments for trainable graph retrieval inference.
    可训练图检索推理脚本的 CLI 参数。

    Fields / 字段:
    - tasks: Path to `*_memory_tasks.input.json`.
      tasks：`*_memory_tasks.input.json` 路径。
    - graphs: Path to `*_graphs.json`.
      graphs：`*_graphs.json` 路径。
    - checkpoint: Path to `best.pt`.
      checkpoint：`best.pt` 路径。
    - output: Path to ranked result JSON.
      output：ranked result JSON 输出路径。
    - top_k: Number of top nodes used for retrieved_subgraph.
      top_k：用于 retrieved_subgraph 的 top node 数量。
    - device: Torch inference device.
      device：torch 推理 device。
    """

    tasks: str
    graphs: str
    checkpoint: str
    output: str
    top_k: int
    device: str


def main(
    argv: Sequence[str] | None = None,
    *,
    text_embedding_provider: TextEmbeddingProvider | None = None,
    seed_signal_provider: SeedSignalProvider | None = None,
) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")

    started_at = now_iso()
    start_time = time.perf_counter()
    output_path = Path(args.output)
    summary_path = output_path.with_name(f"{output_path.stem}.run_summary.json")
    effective_config = {
        "method": "dense_rgcn_graph_retriever",
        "top_k": args.top_k,
        "checkpoint": args.checkpoint,
        "device": args.device,
    }
    inputs = {"tasks": args.tasks, "graphs": args.graphs, "checkpoint": args.checkpoint}
    outputs = {"predictions": args.output, "run_summary": str(summary_path)}

    try:
        task_inputs = read_json(args.tasks)
        graphs = read_json(args.graphs)
        validate_memory_task_inputs(task_inputs)
        predictions = run_retrieval(
            RunRetrievalRequest(
                method="dense_rgcn_graph_retriever",
                task_inputs=task_inputs,
                graphs=graphs,
                top_k=args.top_k,
                trainable_runtime=TrainableGraphRuntime(
                    checkpoint_path=args.checkpoint,
                    device=args.device,
                    text_embedding_provider=text_embedding_provider,
                    seed_signal_provider=seed_signal_provider,
                ),
            )
        )
        inputs_by_task_id = {task_input["task_id"]: task_input for task_input in task_inputs}
        validate_ranked_results(predictions, inputs_by_task_id)
        write_json(args.output, predictions)

        avg_latency = (
            sum(prediction["latency_ms"] for prediction in predictions) / len(predictions)
            if predictions
            else 0.0
        )
        summary = build_run_summary(
            script="run_trainable_retrieval.py",
            started_at=started_at,
            finished_at=now_iso(),
            status="success",
            effective_config=effective_config,
            inputs=inputs,
            outputs=outputs,
            counts={"tasks": len(task_inputs), "predictions": len(predictions)},
            timings={"total_seconds": time.perf_counter() - start_time, "avg_latency_ms": avg_latency},
            environment=collect_environment(),
            notes=[],
        )
        write_run_summary(summary_path, summary)
        LOGGER.info("wrote predictions: %s", args.output)
        LOGGER.info("wrote run summary: %s", summary_path)
        return 0
    except Exception as error:
        LOGGER.error("%s", error)
        summary = build_run_summary(
            script="run_trainable_retrieval.py",
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
    parser = argparse.ArgumentParser(description="Run checkpoint-backed trainable graph retrieval.")
    parser.add_argument("--tasks", required=True, help="Path to *_memory_tasks.input.json.")
    parser.add_argument("--graphs", required=True, help="Path to *_graphs.json.")
    parser.add_argument("--checkpoint", required=True, help="Path to best.pt.")
    parser.add_argument("--output", required=True, help="Path to write ranked result JSON.")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> RunTrainableRetrievalArgs:
    namespace = build_parser().parse_args(argv)
    return RunTrainableRetrievalArgs(
        tasks=namespace.tasks,
        graphs=namespace.graphs,
        checkpoint=namespace.checkpoint,
        output=namespace.output,
        top_k=namespace.top_k,
        device=namespace.device,
    )


if __name__ == "__main__":
    raise SystemExit(main())

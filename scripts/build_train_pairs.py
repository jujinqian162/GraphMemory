from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graph_memory.io import read_json, write_json
from graph_memory.learned.data import build_train_pairs
from graph_memory.observability import build_run_summary, collect_environment, now_iso, write_run_summary
from graph_memory.types import NegativeSamplingConfig

LOGGER = logging.getLogger("build_train_pairs")


@dataclass(frozen=True)
class BuildTrainPairsArgs:
    """
    Parsed CLI arguments for train pair artifact construction.
    训练 pair artifact 构造脚本的 CLI 参数。

    Fields / 字段:
    - tasks: Path to `*_memory_tasks.input.json`.
      tasks：`*_memory_tasks.input.json` 路径。
    - labels: Path to `*_memory_tasks.labels.json`.
      labels：`*_memory_tasks.labels.json` 路径。
    - graphs: Path to `*_graphs.json`.
      graphs：`*_graphs.json` 路径。
    - output: Path to write `*_pairs.json`.
      output：写入 `*_pairs.json` 的路径。
    - random_seed: Seed used by deterministic negative sampling.
      random_seed：确定性负采样使用的种子。
    - easy_random_per_positive: Easy random negatives per positive.
      easy_random_per_positive：每个正例对应的 easy random 负例数量。
    - hard_bm25_per_positive: Hard BM25 negatives per positive.
      hard_bm25_per_positive：每个正例对应的 hard BM25 负例数量。
    - hard_dense_per_positive: Hard dense negatives per positive.
      hard_dense_per_positive：每个正例对应的 hard dense 负例数量。
    - hard_graph_neighbor_per_positive: Graph-neighbor negatives per positive.
      hard_graph_neighbor_per_positive：每个正例对应的 graph-neighbor 负例数量。
    - hard_pool_size: Candidate pool size for hard retriever negatives.
      hard_pool_size：hard retriever 负例候选池大小。
    """

    tasks: str
    labels: str
    graphs: str
    output: str
    random_seed: int
    easy_random_per_positive: int
    hard_bm25_per_positive: int
    hard_dense_per_positive: int
    hard_graph_neighbor_per_positive: int
    hard_pool_size: int


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")

    started_at = now_iso()
    start_time = time.perf_counter()
    output_path = Path(args.output)
    summary_path = output_path.with_name(f"{output_path.stem}.summary.json")
    run_summary_path = output_path.with_name(f"{output_path.stem}.run_summary.json")
    sampling_config = NegativeSamplingConfig(
        random_seed=args.random_seed,
        easy_random_per_positive=args.easy_random_per_positive,
        hard_bm25_per_positive=args.hard_bm25_per_positive,
        hard_dense_per_positive=args.hard_dense_per_positive,
        hard_graph_neighbor_per_positive=args.hard_graph_neighbor_per_positive,
        hard_pool_size=args.hard_pool_size,
    )
    effective_config = {
        "random_seed": sampling_config.random_seed,
        "easy_random_per_positive": sampling_config.easy_random_per_positive,
        "hard_bm25_per_positive": sampling_config.hard_bm25_per_positive,
        "hard_dense_per_positive": sampling_config.hard_dense_per_positive,
        "hard_graph_neighbor_per_positive": sampling_config.hard_graph_neighbor_per_positive,
        "hard_pool_size": sampling_config.hard_pool_size,
    }
    inputs = {"tasks": args.tasks, "labels": args.labels, "graphs": args.graphs}
    outputs = {"pairs": args.output, "summary": str(summary_path), "run_summary": str(run_summary_path)}

    try:
        task_inputs = read_json(args.tasks)
        labels = read_json(args.labels)
        graphs = read_json(args.graphs)
        result = build_train_pairs(task_inputs, labels, graphs, sampling_config)
        write_json(args.output, result.pairs)
        write_json(summary_path, result.summary)

        summary = build_run_summary(
            script="build_train_pairs.py",
            started_at=started_at,
            finished_at=now_iso(),
            status="success",
            effective_config=effective_config,
            inputs=inputs,
            outputs=outputs,
            counts={
                "task_inputs": len(task_inputs),
                "labels": len(labels),
                "graphs": len(graphs),
                "pairs": len(result.pairs),
                "positive_count": result.summary["positive_count"],
                "negative_count_by_type": result.summary["negative_count_by_type"],
            },
            timings={"total_seconds": time.perf_counter() - start_time},
            environment=collect_environment(),
            notes=[],
        )
        write_run_summary(run_summary_path, summary)
        LOGGER.info("wrote train pairs: %s", args.output)
        LOGGER.info("wrote train pair summary: %s", summary_path)
        LOGGER.info("wrote run summary: %s", run_summary_path)
        return 0
    except Exception as error:
        LOGGER.error("%s", error)
        summary = build_run_summary(
            script="build_train_pairs.py",
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
        write_run_summary(run_summary_path, summary)
        raise


def build_parser() -> argparse.ArgumentParser:
    defaults = NegativeSamplingConfig()
    parser = argparse.ArgumentParser(description="Build train pair artifacts for the trainable graph retriever.")
    parser.add_argument("--tasks", required=True, help="Path to *_memory_tasks.input.json.")
    parser.add_argument("--labels", required=True, help="Path to *_memory_tasks.labels.json.")
    parser.add_argument("--graphs", required=True, help="Path to *_graphs.json.")
    parser.add_argument("--output", required=True, help="Path to write *_pairs.json.")
    parser.add_argument("--random_seed", type=int, default=defaults.random_seed)
    parser.add_argument("--easy_random_per_positive", type=int, default=defaults.easy_random_per_positive)
    parser.add_argument("--hard_bm25_per_positive", type=int, default=defaults.hard_bm25_per_positive)
    parser.add_argument("--hard_dense_per_positive", type=int, default=defaults.hard_dense_per_positive)
    parser.add_argument(
        "--hard_graph_neighbor_per_positive",
        type=int,
        default=defaults.hard_graph_neighbor_per_positive,
    )
    parser.add_argument("--hard_pool_size", type=int, default=defaults.hard_pool_size)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> BuildTrainPairsArgs:
    namespace = build_parser().parse_args(argv)
    return BuildTrainPairsArgs(
        tasks=namespace.tasks,
        labels=namespace.labels,
        graphs=namespace.graphs,
        output=namespace.output,
        random_seed=namespace.random_seed,
        easy_random_per_positive=namespace.easy_random_per_positive,
        hard_bm25_per_positive=namespace.hard_bm25_per_positive,
        hard_dense_per_positive=namespace.hard_dense_per_positive,
        hard_graph_neighbor_per_positive=namespace.hard_graph_neighbor_per_positive,
        hard_pool_size=namespace.hard_pool_size,
    )


if __name__ == "__main__":
    raise SystemExit(main())

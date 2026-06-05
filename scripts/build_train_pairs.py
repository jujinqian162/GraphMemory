from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graph_memory.config import CONFIG_LOADER
from graph_memory.contracts.common import JsonValue
from graph_memory.io import read_json, write_json
from graph_memory.observability import build_run_summary, collect_environment, now_iso, write_run_summary
from graph_memory.registry import Registry
from graph_memory.registry.stage_configs import PairBuildStageConfig
from graph_memory.retrieval.methods.flat.dense import DenseConfig
from graph_memory.training_pairs import build_train_pairs

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
    - config: Optional resolved trainable training config path.
      config：可选的已解析可训练 training config 路径。
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
    config: str | None


def main(argv: Sequence[str] | None = None) -> int:
    config = CONFIG_LOADER.load(Registry.configs.PAIRS, argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")

    started_at = now_iso()
    start_time = time.perf_counter()
    summary_path = config.io.summary
    run_summary_path = config.io.run_summary
    sampling_config = config.job.sampling.to_negative_sampling_config()
    dense_config = _dense_config_from_config(config)
    effective_config = _effective_config(config)
    inputs = {"tasks": str(config.io.tasks), "labels": str(config.io.labels), "graphs": str(config.io.graphs)}
    outputs = {"pairs": str(config.io.output), "summary": str(summary_path), "run_summary": str(run_summary_path)}

    try:
        task_inputs = read_json(config.io.tasks)
        labels = read_json(config.io.labels)
        graphs = read_json(config.io.graphs)
        result = build_train_pairs(task_inputs, labels, graphs, sampling_config, dense_config=dense_config)
        write_json(config.io.output, result.pairs)
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
        LOGGER.info("wrote train pairs: %s", config.io.output)
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
    return Registry.configs.PAIRS.parser_factory()


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
        config=namespace.config,
    )


def _effective_config(config: PairBuildStageConfig) -> dict[str, JsonValue]:
    sampling_config = config.job.sampling
    effective_config: dict[str, JsonValue] = {
        "random_seed": sampling_config.random_seed,
        "easy_random_per_positive": sampling_config.easy_random_per_positive,
        "hard_bm25_per_positive": sampling_config.hard_bm25_per_positive,
        "hard_dense_per_positive": sampling_config.hard_dense_per_positive,
        "hard_graph_neighbor_per_positive": sampling_config.hard_graph_neighbor_per_positive,
        "hard_pool_size": sampling_config.hard_pool_size,
    }
    dense_config = _dense_config_from_config(config)
    if dense_config is not None:
        effective_config["encoder_model"] = dense_config.model_name
    return effective_config


def _dense_config_from_config(config: PairBuildStageConfig) -> DenseConfig | None:
    job = config.job
    if job.sampling.hard_dense_per_positive <= 0 or job.hard_dense_encoder is None:
        if config.io.config is not None and job.sampling.hard_dense_per_positive > 0:
            raise ValueError("Pair build config with hard dense negatives requires encoder settings.")
        return None
    return DenseConfig(
        model_name=job.hard_dense_encoder.model_name,
        query_prefix=job.hard_dense_encoder.query_prefix,
        passage_prefix=job.hard_dense_encoder.passage_prefix,
        batch_size=job.hard_dense_encoder.batch_size,
    )


if __name__ == "__main__":
    raise SystemExit(main())

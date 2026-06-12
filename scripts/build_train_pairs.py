from __future__ import annotations

import logging
import sys
import time
from collections.abc import Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graph_memory.config import CONFIG_LOADER
from graph_memory.contracts.common import JsonValue
from graph_memory.io import read_json, write_json
from graph_memory.observability import build_run_summary, collect_environment, now_iso, write_run_summary
from graph_memory.registry import Registry
from graph_memory.registry.conversions import dense_config_from_encoder_settings
from graph_memory.registry.stage_configs import PairBuildStageConfig
from graph_memory.retrieval.methods.flat.dense import DenseConfig
from graph_memory.training_pairs import build_train_pairs

LOGGER = logging.getLogger("build_train_pairs")


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
        if job.sampling.hard_dense_per_positive > 0:
            raise ValueError("Pair build config with hard dense negatives requires encoder settings.")
        return None
    return dense_config_from_encoder_settings(job.hard_dense_encoder)


if __name__ == "__main__":
    raise SystemExit(main())

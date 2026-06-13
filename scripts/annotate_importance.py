from __future__ import annotations

import logging
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import cast


def _prepare_local_transformers_environment() -> None:
    for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
        os.environ.pop(key, None)
    os.environ["ACCELERATE_USE_DEEPSPEED"] = "false"


_prepare_local_transformers_environment()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graph_memory.config import CONFIG_LOADER  # noqa: E402
from graph_memory.contracts.common import JsonObject  # noqa: E402
from graph_memory.observability import build_run_summary, collect_environment, now_iso, write_run_summary  # noqa: E402
from graph_memory.registry import Registry  # noqa: E402
from graph_memory.registry.stage_configs import ImportanceStageConfig  # noqa: E402
from graph_memory.stages.importance import run_importance_stage  # noqa: E402

LOGGER = logging.getLogger("annotate_importance")


def main(argv: Sequence[str] | None = None) -> int:
    config = CONFIG_LOADER.load(Registry.configs.IMPORTANCE, argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")

    started_at = now_iso()
    start_time = time.perf_counter()
    effective_config = _effective_config(config)
    inputs = {"tasks": str(config.io.tasks), "cache_dir": str(config.io.cache_dir)}
    outputs = {"importance": str(config.io.output), "run_summary": str(config.io.summary)}

    try:
        result = run_importance_stage(config)
        annotation = result.annotation
        memory_items = sum(len(task["scores"]) for task in annotation.artifact["tasks"])
        summary = build_run_summary(
            script="annotate_importance.py",
            started_at=started_at,
            finished_at=now_iso(),
            status="success",
            effective_config=effective_config,
            inputs=inputs,
            outputs=outputs,
            counts={
                "tasks": len(annotation.artifact["tasks"]),
                "memory_items": memory_items,
                "cache_hits": annotation.cache_stats.hits,
                "cache_misses": annotation.cache_stats.misses,
                "cache_writes": annotation.cache_stats.writes,
                "model_load_count": annotation.model_load_count,
                "generation_calls": annotation.generation_calls,
                "generated_tokens": annotation.generated_tokens,
            },
            timings={
                "total_seconds": time.perf_counter() - start_time,
                "model_load_seconds": annotation.model_load_seconds,
                "generation_seconds": annotation.generation_seconds,
            },
            environment=collect_environment(),
            notes=[],
        )
        write_run_summary(config.io.summary, summary)
        LOGGER.info("tasks=%s cache_hits=%s generations=%s", len(annotation.artifact["tasks"]), annotation.cache_stats.hits, annotation.generation_calls)
        LOGGER.info("wrote importance artifact: %s", config.io.output)
        LOGGER.info("wrote run summary: %s", config.io.summary)
        return 0
    except Exception as error:
        LOGGER.error("%s", error)
        summary = build_run_summary(
            script="annotate_importance.py",
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
        write_run_summary(config.io.summary, summary)
        raise


def _effective_config(config: ImportanceStageConfig) -> JsonObject:
    value = dict(cast(dict[str, object], CONFIG_LOADER.to_json(config.job)))
    value["model_path"] = config.job.model_path.as_posix()
    return cast(JsonObject, value)


if __name__ == "__main__":
    raise SystemExit(main())

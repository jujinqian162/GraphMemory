from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast


def _prepare_local_transformers_environment() -> None:
    for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
        os.environ.pop(key, None)
    os.environ["ACCELERATE_USE_DEEPSPEED"] = "false"


_prepare_local_transformers_environment()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graph_memory.contracts.common import JsonObject  # noqa: E402
from graph_memory.contracts.tasks import MemoryTaskInput  # noqa: E402
from graph_memory.infrastructure.io import read_json, write_json_atomic  # noqa: E402
from graph_memory.observability import build_run_summary, collect_environment, now_iso, write_run_summary  # noqa: E402
from graph_memory.validation import validate_memory_task_inputs  # noqa: E402
from graph_memory.retrieval.methods.memory_stream.annotation import RuntimeFactory, annotate_importance_tasks  # noqa: E402
from graph_memory.retrieval.methods.memory_stream.prompt import IMPORTANCE_PROMPT_VERSION  # noqa: E402
from graph_memory.retrieval.methods.memory_stream.runtime import LocalTransformersImportanceRuntime  # noqa: E402
from graph_memory.retrieval.methods.memory_stream.settings import ImportanceAnnotationSettings  # noqa: E402

LOGGER = logging.getLogger("annotate_importance")

DEFAULT_TASKS = Path("data/hotpotqa/processed/dev_memory_tasks.input.json")
DEFAULT_OUTPUT = Path("data/hotpotqa/processed/memory_stream/dev.importance.json")
DEFAULT_CACHE_DIR = Path("data/cache/memory_stream_importance")
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_MODEL_PATH = Path(os.environ.get("MEMORY_STREAM_MODEL_PATH", "models/Qwen2.5-7B-Instruct"))


@dataclass(frozen=True)
class ImportancePrepareArgs:
    tasks: Path
    output: Path
    summary: Path
    cache_dir: Path
    model_id: str
    model_path: Path
    prompt_version: str
    device: Literal["auto", "cuda", "cpu"]
    max_new_tokens: int


def main(
    argv: Sequence[str] | None = None,
    *,
    runtime_factory: RuntimeFactory | None = None,
) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")

    settings = _settings(args)
    started_at = now_iso()
    start_time = time.perf_counter()
    effective_config = _effective_config(settings)
    inputs = {"tasks": args.tasks.as_posix(), "cache_dir": args.cache_dir.as_posix()}
    outputs = {"importance": args.output.as_posix(), "run_summary": args.summary.as_posix()}

    try:
        loaded = read_json(args.tasks)
        validate_memory_task_inputs(loaded)
        task_inputs = cast(list[MemoryTaskInput], loaded)
        active_runtime_factory = runtime_factory or (
            lambda active_settings: LocalTransformersImportanceRuntime(active_settings)
        )
        annotation = annotate_importance_tasks(
            task_inputs,
            settings,
            cache_dir=args.cache_dir,
            runtime_factory=active_runtime_factory,
        )
        write_json_atomic(args.output, annotation.artifact)
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
        write_run_summary(args.summary, summary)
        LOGGER.info(
            "tasks=%s cache_hits=%s generations=%s",
            len(annotation.artifact["tasks"]),
            annotation.cache_stats.hits,
            annotation.generation_calls,
        )
        LOGGER.info("wrote importance artifact: %s", args.output)
        LOGGER.info("wrote run summary: %s", args.summary)
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
        write_run_summary(args.summary, summary)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the globally shared Memory Stream importance artifact.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS, help="Canonical task input JSON.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Global importance artifact.")
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Run summary path. Defaults beside --output.",
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR, help="Reusable per-task cache.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="Semantic model identity for cache keys.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH, help="Local Transformers model path.")
    parser.add_argument("--prompt-version", default=IMPORTANCE_PROMPT_VERSION, help="Importance prompt contract.")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto", help="Runtime device policy.")
    parser.add_argument("--max-new-tokens", type=int, default=2048, help="Maximum generated tokens per task.")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> ImportancePrepareArgs:
    namespace = build_parser().parse_args(argv)
    output = cast(Path, namespace.output)
    summary = cast(Path | None, namespace.summary)
    return ImportancePrepareArgs(
        tasks=cast(Path, namespace.tasks),
        output=output,
        summary=summary or output.with_name(f"{output.stem}.run_summary.json"),
        cache_dir=cast(Path, namespace.cache_dir),
        model_id=str(namespace.model_id),
        model_path=cast(Path, namespace.model_path),
        prompt_version=str(namespace.prompt_version),
        device=cast(Literal["auto", "cuda", "cpu"], namespace.device),
        max_new_tokens=int(namespace.max_new_tokens),
    )


def _settings(args: ImportancePrepareArgs) -> ImportanceAnnotationSettings:
    return ImportanceAnnotationSettings(
        model_id=args.model_id,
        model_path=args.model_path,
        prompt_version=args.prompt_version,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
    )


def _effective_config(settings: ImportanceAnnotationSettings) -> JsonObject:
    return {
        "model_id": settings.model_id,
        "model_path": settings.model_path.as_posix(),
        "prompt_version": settings.prompt_version,
        "device": settings.device,
        "trust_remote_code": settings.trust_remote_code,
        "torch_dtype": settings.torch_dtype,
        "low_cpu_mem_usage": settings.low_cpu_mem_usage,
        "tp_plan": settings.tp_plan,
        "do_sample": settings.do_sample,
        "use_cache": settings.use_cache,
        "max_new_tokens": settings.max_new_tokens,
    }


if __name__ == "__main__":
    raise SystemExit(main())

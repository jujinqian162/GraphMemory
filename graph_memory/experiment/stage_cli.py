from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, TypeVar

from pydantic import BaseModel, TypeAdapter

from graph_memory.experiment.invocation import StageInvocation
from graph_memory.experiment.persistence import read_yaml_model

T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True)
class StageExecution(Generic[T]):
    config: T
    invocation: StageInvocation


def load_stage_execution(
    argv: list[str] | tuple[str, ...] | None,
    expected: type[T] | TypeAdapter[T],
    *,
    description: str,
    script: Path,
) -> StageExecution[T]:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args(argv)
    config_path = arguments.config.resolve()
    if config_path.suffix.lower() not in {".yaml", ".yml"}:
        parser.error("--config must reference a resolved YAML file")

    invocation = read_yaml_model(config_path, StageInvocation)
    if invocation.config_path != config_path:
        raise ValueError(
            f"stage execution config path mismatch: "
            f"expected={invocation.config_path} actual={config_path}"
        )
    actual_script = script.resolve()
    if invocation.script != actual_script:
        raise ValueError(
            f"stage execution script mismatch: "
            f"expected={invocation.script} actual={actual_script}"
        )

    adapter = expected if isinstance(expected, TypeAdapter) else TypeAdapter(expected)
    config = adapter.validate_python(invocation.config)
    return StageExecution(config=config, invocation=invocation)


__all__ = ["StageExecution", "load_stage_execution"]

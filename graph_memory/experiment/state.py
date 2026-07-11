from __future__ import annotations

import traceback
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from graph_memory.experiment.config import (
    ArtifactRef,
    ClosedModel,
    PublicStageName,
    ResolvedExperimentConfig,
)
from graph_memory.experiment.layout import RunLayout, RunMode
from graph_memory.experiment.persistence import read_yaml_model, write_yaml_atomic
from graph_memory.experiment.planning import StageInvocation, WorkflowPlan
from graph_memory.registry.retrieval import RetrievalMethodId

StageAttemptStatus = Literal["running", "success", "failed"]


class ErrorRecord(ClosedModel):
    type: str
    message: str
    traceback: str


class RunState(ClosedModel):
    version: Literal[1] = 1
    name: str
    mode: RunMode
    created_at: datetime
    updated_at: datetime
    config: ResolvedExperimentConfig
    resolved_config_path: Path
    overrides_path: Path
    selected_methods: tuple[RetrievalMethodId, ...]
    selected_stages: tuple[PublicStageName, ...]
    plan: tuple[str, ...]
    artifacts: tuple[ArtifactRef, ...]
    mlflow_parent_run_id: str | None = None


class StageRunSummary(ClosedModel):
    version: Literal[1] = 1
    identifier: str
    stage: PublicStageName
    script: Path
    method: RetrievalMethodId | None
    split: str | None
    variant: str | None
    status: StageAttemptStatus
    attempt: int = Field(ge=1)
    started_at: datetime
    ended_at: datetime | None
    effective_config: dict[str, Any]
    inputs: tuple[ArtifactRef, ...]
    outputs: tuple[ArtifactRef, ...]
    counts: dict[str, Any]
    timings: dict[str, float]
    error: ErrorRecord | None
    mlflow_child_run_id: str | None = None


@dataclass
class StageObservations:
    counts: dict[str, Any] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)

    def count(self, name: str, value: Any) -> None:
        self.counts[name] = value

    def timing(self, name: str, seconds: float) -> None:
        self.timings[name] = seconds


def create_run_state(
    *,
    layout: RunLayout,
    config: ResolvedExperimentConfig,
    plan: WorkflowPlan,
) -> RunState:
    now = _now()
    stages: tuple[PublicStageName, ...] = tuple(
        dict.fromkeys(item.stage for item in plan.invocations)
    )
    artifacts = _unique_artifacts(
        artifact
        for artifact in (
            *(
                artifact
                for item in plan.invocations
                for artifact in (*item.inputs, *item.outputs)
            ),
            *(alias.artifact for alias in plan.aliases),
        )
    )
    return RunState(
        name=layout.name,
        mode=layout.mode,
        created_at=now,
        updated_at=now,
        config=config,
        resolved_config_path=layout.resolved_config.resolve(),
        overrides_path=layout.overrides.resolve(),
        selected_methods=tuple(config.methods),
        selected_stages=stages,
        plan=tuple(item.identifier for item in plan.invocations),
        artifacts=artifacts,
    )


def read_run_state(path: Path) -> RunState:
    return read_yaml_model(path, RunState)


def write_run_state(path: Path, state: RunState) -> None:
    write_yaml_atomic(path, state)


def update_run_state(
    path: Path,
    state: RunState,
    *,
    mlflow_parent_run_id: str | None = None,
) -> RunState:
    if mlflow_parent_run_id is None or mlflow_parent_run_id == state.mlflow_parent_run_id:
        return state
    updated = state.model_copy(
        update={
            "updated_at": _now(),
            "mlflow_parent_run_id": mlflow_parent_run_id,
        }
    )
    write_run_state(path, updated)
    return updated


def validate_run_identity(
    *,
    layout: RunLayout,
    config: ResolvedExperimentConfig,
    existing: RunState,
) -> None:
    if existing.name != layout.name:
        raise ValueError(
            f"run identity name mismatch: expected={layout.name} actual={existing.name}"
        )
    if existing.mode != layout.mode:
        raise ValueError(
            f"run identity mode mismatch: expected={layout.mode} actual={existing.mode}"
        )
    if existing.config.normalized() != config.normalized():
        raise ValueError(
            f"run identity config mismatch for name={layout.name}; choose another name or reset the run"
        )


def assert_named_root_mode(layout: RunLayout) -> None:
    single_state = layout.named_root / "run_state.yaml"
    if layout.mode == "multirun" and single_state.is_file():
        state = read_run_state(single_state)
        if state.mode == "single":
            raise ValueError(
                f"run identity mode mismatch: name={layout.name} already owns a single run"
            )
    if (
        layout.mode == "single"
        and layout.named_root.is_dir()
        and not single_state.exists()
    ):
        child_states = tuple(layout.named_root.glob("*/run_state.yaml"))
        if child_states and any(
            read_run_state(path).mode == "multirun" for path in child_states
        ):
            raise ValueError(
                f"run identity mode mismatch: name={layout.name} already owns multirun jobs"
            )


def read_stage_summary(path: Path) -> StageRunSummary:
    return read_yaml_model(path, StageRunSummary)


def write_stage_summary(path: Path, summary: StageRunSummary) -> None:
    write_yaml_atomic(path, summary)


@contextmanager
def stage_lifecycle(
    invocation: StageInvocation,
) -> Iterator[StageObservations]:
    destination = summary_path_for(invocation)
    attempt = _next_attempt(destination)
    started = _now()
    observations = StageObservations()
    running = _stage_summary(
        invocation,
        status="running",
        attempt=attempt,
        started_at=started,
        ended_at=None,
        observations=observations,
        error=None,
    )
    write_stage_summary(destination, running)
    try:
        yield observations
        _validate_declared_outputs(invocation)
    except BaseException as error:
        failed = _stage_summary(
            invocation,
            status="failed",
            attempt=attempt,
            started_at=started,
            ended_at=_now(),
            observations=observations,
            error=ErrorRecord(
                type=type(error).__name__,
                message=str(error),
                traceback="".join(traceback.format_exception(error)),
            ),
        )
        write_stage_summary(destination, failed)
        raise
    succeeded = _stage_summary(
        invocation,
        status="success",
        attempt=attempt,
        started_at=started,
        ended_at=_now(),
        observations=observations,
        error=None,
    )
    write_stage_summary(destination, succeeded)


def _stage_summary(
    invocation: StageInvocation,
    *,
    status: StageAttemptStatus,
    attempt: int,
    started_at: datetime,
    ended_at: datetime | None,
    observations: StageObservations,
    error: ErrorRecord | None,
) -> StageRunSummary:
    return StageRunSummary(
        identifier=invocation.identifier,
        stage=invocation.stage,
        script=invocation.script,
        method=invocation.method,
        split=invocation.split,
        variant=invocation.variant,
        status=status,
        attempt=attempt,
        started_at=started_at,
        ended_at=ended_at,
        effective_config=invocation.config.model_dump(mode="json", by_alias=True),
        inputs=invocation.inputs,
        outputs=invocation.outputs,
        counts=dict(observations.counts),
        timings=dict(observations.timings),
        error=error,
        mlflow_child_run_id=None,
    )


def _next_attempt(path: Path) -> int:
    if not path.is_file():
        return 1
    return read_stage_summary(path).attempt + 1


def summary_path_for(invocation: StageInvocation) -> Path:
    value = getattr(invocation.config, "summary", None)
    if isinstance(value, Path):
        return value
    outputs = getattr(invocation.config, "outputs", None)
    nested = getattr(outputs, "summary", None)
    if isinstance(nested, Path):
        return nested
    raise ValueError(f"stage config has no summary path: {invocation.identifier}")


def _validate_declared_outputs(invocation: StageInvocation) -> None:
    for output in invocation.outputs:
        valid = output.path.is_file() if output.kind == "file" else output.path.is_dir()
        if not valid:
            raise RuntimeError(
                f"stage={invocation.identifier} did not produce {output.kind} "
                f"output role={output.role} path={output.path}"
            )


def _unique_artifacts(artifacts: Iterator[ArtifactRef]) -> tuple[ArtifactRef, ...]:
    result: list[ArtifactRef] = []
    seen: set[tuple[str, Path, str, Path | None]] = set()
    for artifact in artifacts:
        key = (artifact.role, artifact.path, artifact.kind, artifact.alias_of)
        if key not in seen:
            seen.add(key)
            result.append(artifact)
    return tuple(result)


def _now() -> datetime:
    return datetime.now(timezone.utc)


__all__ = [
    "ErrorRecord",
    "RunState",
    "StageObservations",
    "StageRunSummary",
    "assert_named_root_mode",
    "create_run_state",
    "read_run_state",
    "read_stage_summary",
    "stage_lifecycle",
    "summary_path_for",
    "update_run_state",
    "validate_run_identity",
    "write_run_state",
    "write_stage_summary",
]

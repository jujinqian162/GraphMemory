from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, TypeVar, cast

from pydantic import BaseModel, TypeAdapter

from graph_memory.experiment.config import ArtifactKind, ArtifactRef
from graph_memory.experiment.persistence import read_yaml
from graph_memory.experiment.planning import StageInvocation, stage_identifier
from graph_memory.experiment.stage_models import (
    AblationAggregateStageConfig,
    Bm25GraphRerankRetrieveStageConfig,
    Bm25RetrieveStageConfig,
    DenseFinetuneRetrieveStageConfig,
    DenseFinetuneTrainStageConfig,
    DenseGraphRerankRetrieveStageConfig,
    DenseRetrieveStageConfig,
    EvaluateStageConfig,
    GraphRerankTuneStageConfig,
    GraphStageConfig,
    ImportancePrepareStageConfig,
    MemoryStreamRetrieveStageConfig,
    MemoryStreamTuneStageConfig,
    OrdinaryAggregateStageConfig,
    PairStageConfig,
    RawPrepareStageConfig,
    RgcnRetrieveStageConfig,
    RgcnTrainStageConfig,
    StageConfig,
)
from graph_memory.registry.retrieval import RetrievalMethodId

T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True)
class StageExecution(Generic[T]):
    config_path: Path
    config: T
    invocation: StageInvocation


def load_stage_execution(
    argv: list[str] | tuple[str, ...] | None,
    expected: type[T] | TypeAdapter[T],
    *,
    description: str,
    script: Path | None = None,
) -> StageExecution[T]:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args(argv)
    config_path = arguments.config.resolve()
    if config_path.suffix.lower() not in {".yaml", ".yml"}:
        parser.error("--config must reference a resolved YAML file")
    adapter = expected if isinstance(expected, TypeAdapter) else TypeAdapter(expected)
    config = adapter.validate_python(read_yaml(config_path))
    _require_absolute_paths(config)
    invocation = invocation_from_stage_config(
        cast(StageConfig, config),
        config_path=config_path,
        script=(script or Path(sys.argv[0])).resolve(),
    )
    return StageExecution(config_path=config_path, config=config, invocation=invocation)


def invocation_from_stage_config(
    config: StageConfig,
    *,
    config_path: Path,
    script: Path,
) -> StageInvocation:
    method_value = getattr(config, "method", None)
    method = RetrievalMethodId(method_value) if method_value is not None else None
    split = getattr(config, "split", None)
    variant = getattr(config, "variant", None)
    inputs, outputs = _artifact_bindings(config)
    return StageInvocation(
        identifier=stage_identifier(
            config.stage,
            method=method,
            split=split,
            variant=variant,
        ),
        stage=config.stage,
        script=script.resolve(),
        config_path=config_path.resolve(),
        config=config,
        inputs=inputs,
        outputs=outputs,
        dependencies=(),
        method=method,
        split=split,
        variant=variant,
    )


def _artifact_bindings(
    config: StageConfig,
) -> tuple[tuple[ArtifactRef, ...], tuple[ArtifactRef, ...]]:
    if isinstance(config, RawPrepareStageConfig):
        return (
            (_ref("raw", config.source),),
            _prepare_outputs(config),
        )
    if isinstance(config, ImportancePrepareStageConfig):
        return (
            (
                _ref("canonical_inputs", config.canonical_inputs),
                _ref("canonical_labels", config.canonical_labels),
                _ref("importance", config.importance),
            ),
            _prepare_outputs(config),
        )
    if isinstance(config, GraphStageConfig):
        return ((_ref("inputs", config.tasks),), (_ref("graphs", config.output),))
    if isinstance(config, PairStageConfig):
        return (
            (
                _ref("inputs", config.tasks),
                _ref("labels", config.labels),
                _ref("graphs", config.graphs),
            ),
            (
                _ref("train_pairs", config.outputs.pairs),
                _ref("train_pair_summary", config.outputs.pair_summary),
            ),
        )
    if isinstance(config, GraphRerankTuneStageConfig):
        return (
            (
                _ref("inputs", config.tasks),
                _ref("labels", config.labels),
                _ref("graphs", config.graphs),
            ),
            (
                _ref("selected_config", config.selected_config),
                _ref("candidate_table", config.candidates),
            ),
        )
    if isinstance(config, MemoryStreamTuneStageConfig):
        return (
            (
                _ref("inputs", config.tasks),
                _ref("labels", config.labels),
                _ref("graphs", config.graphs),
                _ref("importance", config.importance),
            ),
            (
                _ref("selected_config", config.selected_config),
                _ref("candidate_table", config.candidates),
            ),
        )
    if isinstance(config, RgcnTrainStageConfig):
        inputs = (
            _ref("inputs", config.train_tasks),
            _ref("labels", config.train_labels),
            _ref("graphs", config.train_graphs),
            _ref("train_pairs", config.train_pairs),
            _ref("dev_inputs", config.dev_tasks),
            _ref("dev_labels", config.dev_labels),
            _ref("dev_graphs", config.dev_graphs),
        )
        if config.seed_checkpoint is not None:
            inputs = (
                *inputs,
                _ref("seed_checkpoint", config.seed_checkpoint, kind="directory"),
            )
        return (
            inputs,
            (
                _ref("checkpoint", config.checkpoint_dir / "best.pt"),
                _ref("train_metrics", config.metrics),
            ),
        )
    if isinstance(config, DenseFinetuneTrainStageConfig):
        return (
            (
                _ref("inputs", config.train_tasks),
                _ref("labels", config.train_labels),
                _ref("train_pairs", config.train_pairs),
                _ref("dev_inputs", config.dev_tasks),
                _ref("dev_labels", config.dev_labels),
            ),
            (
                _ref("checkpoint", config.model_dir, kind="directory"),
                _ref("train_metrics", config.metrics),
            ),
        )
    if isinstance(config, Bm25RetrieveStageConfig):
        return ((_ref("inputs", config.tasks),), (_ref("predictions", config.output),))
    if isinstance(config, DenseRetrieveStageConfig):
        return ((_ref("inputs", config.tasks),), (_ref("predictions", config.output),))
    if isinstance(config, MemoryStreamRetrieveStageConfig):
        return (
            (
                _ref("inputs", config.tasks),
                _ref("selected_config", config.selected_config),
                _ref("importance", config.importance),
            ),
            (_ref("predictions", config.output),),
        )
    if isinstance(
        config,
        (Bm25GraphRerankRetrieveStageConfig, DenseGraphRerankRetrieveStageConfig),
    ):
        return (
            (
                _ref("inputs", config.tasks),
                _ref("graphs", config.graphs),
                _ref("selected_config", config.selected_config),
            ),
            (_ref("predictions", config.output),),
        )
    if isinstance(config, RgcnRetrieveStageConfig):
        return (
            (
                _ref("inputs", config.tasks),
                _ref("graphs", config.graphs),
                _ref("checkpoint", config.checkpoint),
            ),
            (_ref("predictions", config.output),),
        )
    if isinstance(config, DenseFinetuneRetrieveStageConfig):
        return (
            (
                _ref("inputs", config.tasks),
                _ref("checkpoint", config.model_dir, kind="directory"),
            ),
            (_ref("predictions", config.output),),
        )
    if isinstance(config, EvaluateStageConfig):
        return (
            (
                _ref("predictions", config.predictions),
                _ref("labels", config.labels),
                _ref("graphs", config.graphs),
            ),
            (
                _ref("metrics", config.metrics),
                _ref("failure_cases", config.failure_cases),
            ),
        )
    if isinstance(config, (OrdinaryAggregateStageConfig, AblationAggregateStageConfig)):
        outputs = (
            _ref("main_table", config.main),
            _ref("path_table", config.path),
            _ref("efficiency_table", config.efficiency),
        )
        if isinstance(config, AblationAggregateStageConfig):
            outputs = (*outputs, _ref("ablation_table", config.ablation))
        inputs = tuple(_ref("metrics", path) for path in config.metrics)
        if isinstance(config, AblationAggregateStageConfig):
            inputs = (*inputs, _ref("ablation_index", config.ablation_index))
            inputs = (
                *inputs,
                *(_ref("ablation_metrics", path) for path in config.ablation_metrics),
            )
        return (inputs, outputs)
    raise TypeError(f"unsupported stage config: {type(config).__name__}")


def _prepare_outputs(
    config: RawPrepareStageConfig | ImportancePrepareStageConfig,
) -> tuple[ArtifactRef, ...]:
    return (
        _ref("inputs", config.outputs.input),
        _ref("labels", config.outputs.labels),
        _ref("combined", config.outputs.combined),
    )


def _ref(role: str, path: Path, *, kind: ArtifactKind = "file") -> ArtifactRef:
    return ArtifactRef(role=role, path=path.resolve(), kind=kind, alias_of=None)


def _require_absolute_paths(value: object) -> None:
    if isinstance(value, Path):
        if not value.is_absolute():
            raise ValueError(f"stage YAML paths must be absolute: {value}")
        return
    if isinstance(value, BaseModel):
        for field_name in type(value).model_fields:
            _require_absolute_paths(getattr(value, field_name))
        return
    if isinstance(value, dict):
        for item in value.values():
            _require_absolute_paths(item)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _require_absolute_paths(item)


__all__ = [
    "StageExecution",
    "invocation_from_stage_config",
    "load_stage_execution",
]

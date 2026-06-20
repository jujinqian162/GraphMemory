from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Literal, TypeAlias

from graph_memory.config.patches import ConfigPatch, deep_merge_patch
from graph_memory.contracts.common import JsonValue
from graph_memory.registry.ids import StageId
from graph_memory.registry.method_configs import TrainableMethodConfig
from graph_memory.registry.method_configs import (
    DenseFinetuneMethodSettings,
    RgcnMethodSettings,
    validate_complete_method_config_record,
)
from graph_memory.registry.retrieval import (
    DenseEncoderSettings,
    RetrievalJobSettings,
    RetrievalMethodId,
)
from graph_memory.registry.specs import StageConfigSpec
from graph_memory.training_pairs.config import NegativeSamplingConfig


@dataclass(frozen=True)
class GenericStageConfig:
    args: dict[str, object]


@dataclass(frozen=True)
class PairBuildIO:
    tasks: Path
    labels: Path
    graphs: Path
    output: Path
    summary: Path
    run_summary: Path


@dataclass(frozen=True)
class PairSamplingSettings:
    random_seed: int
    easy_random_per_positive: int
    hard_bm25_per_positive: int
    hard_dense_per_positive: int
    hard_graph_neighbor_per_positive: int
    hard_pool_size: int

    def to_negative_sampling_config(self) -> NegativeSamplingConfig:
        return NegativeSamplingConfig(
            random_seed=self.random_seed,
            easy_random_per_positive=self.easy_random_per_positive,
            hard_bm25_per_positive=self.hard_bm25_per_positive,
            hard_dense_per_positive=self.hard_dense_per_positive,
            hard_graph_neighbor_per_positive=self.hard_graph_neighbor_per_positive,
            hard_pool_size=self.hard_pool_size,
        )


@dataclass(frozen=True)
class PairBuildJobSettings:
    sampling: PairSamplingSettings
    hard_dense_encoder: DenseEncoderSettings | None = None


@dataclass(frozen=True)
class PairBuildStageConfig:
    io: PairBuildIO
    job: PairBuildJobSettings

    io_type: ClassVar[type[PairBuildIO]] = PairBuildIO


@dataclass(frozen=True)
class RetrieveIO:
    tasks: Path
    graphs: Path | None
    output: Path
    summary: Path
    selected_config: Path | None = None
    importance: Path | None = None


@dataclass(frozen=True)
class RetrieveStageConfig:
    io: RetrieveIO
    job: RetrievalJobSettings

    io_type: ClassVar[type[RetrieveIO]] = RetrieveIO


@dataclass(frozen=True)
class RgcnTrainIO:
    train_tasks: Path
    train_labels: Path | None
    train_graphs: Path
    train_pairs: Path
    dev_tasks: Path
    dev_labels: Path
    dev_graphs: Path
    output_dir: Path
    checkpoint_dir: Path
    metrics: Path
    run_summary: Path


@dataclass(frozen=True)
class DenseFinetuneTrainIO:
    train_tasks: Path
    train_labels: Path
    train_pairs: Path
    dev_tasks: Path
    dev_labels: Path
    output_dir: Path
    model_dir: Path
    metrics: Path
    run_summary: Path


@dataclass(frozen=True)
class RgcnTrainStageConfig:
    method: Literal[RetrievalMethodId.DENSE_RGCN_GRAPH_RETRIEVER]
    io: RgcnTrainIO
    job: RgcnMethodSettings

    io_type: ClassVar[type[RgcnTrainIO]] = RgcnTrainIO


@dataclass(frozen=True)
class DenseFinetuneTrainStageConfig:
    method: Literal[RetrievalMethodId.DENSE_FT]
    io: DenseFinetuneTrainIO
    job: DenseFinetuneMethodSettings

    io_type: ClassVar[type[DenseFinetuneTrainIO]] = DenseFinetuneTrainIO


TrainStageConfig: TypeAlias = RgcnTrainStageConfig | DenseFinetuneTrainStageConfig


@dataclass(frozen=True)
class EvaluateIO:
    predictions: Path
    labels: Path
    graphs: Path
    output: Path
    failure_cases_output: Path | None = None


@dataclass(frozen=True)
class EvaluateStageConfig:
    io: EvaluateIO
    failure_case_limit: int = 0

    io_type: ClassVar[type[EvaluateIO]] = EvaluateIO


@dataclass(frozen=True)
class StageConfigRegistry:
    TRAINABLE_METHOD: StageConfigSpec[TrainableMethodConfig]
    PREPARE: StageConfigSpec[GenericStageConfig]
    GRAPHS: StageConfigSpec[GenericStageConfig]
    PAIRS: StageConfigSpec[PairBuildStageConfig]
    TUNE: StageConfigSpec[GenericStageConfig]
    TRAIN: StageConfigSpec[TrainStageConfig]
    RETRIEVE: StageConfigSpec[RetrieveStageConfig]
    EVALUATE: StageConfigSpec[EvaluateStageConfig]
    AGGREGATE: StageConfigSpec[GenericStageConfig]
    EXPERIMENT_INIT: StageConfigSpec[GenericStageConfig]


def build_stage_config_registry() -> StageConfigRegistry:
    return StageConfigRegistry(
        TRAINABLE_METHOD=StageConfigSpec(
            stage=StageId.EXPERIMENT_INIT,
            config_type=TrainableMethodConfig,
            parser_factory=_method_config_parser,
            config_path=_config_path_from_attr("config"),
            profile_name=_method_config_profile_name,
            cli_patch=_empty_cli_patch,
            registry_patch=_validate_method_config_profiles,
        ),
        PREPARE=_generic_spec(StageId.PREPARE, _prepare_parser),
        GRAPHS=_generic_spec(StageId.GRAPHS, _graphs_parser),
        PAIRS=_stage_file_spec(StageId.PAIRS, PairBuildStageConfig, "Build train pair artifacts."),
        TUNE=_generic_spec(StageId.TUNE, _tune_parser),
        TRAIN=_stage_file_spec(StageId.TRAIN, TrainStageConfig, "Train a retrieval method."),
        RETRIEVE=_stage_file_spec(StageId.RETRIEVE, RetrieveStageConfig, "Run a retrieval method."),
        EVALUATE=_stage_file_spec(StageId.EVALUATE, EvaluateStageConfig, "Evaluate retrieval predictions."),
        AGGREGATE=_generic_spec(StageId.AGGREGATE, _aggregate_parser),
        EXPERIMENT_INIT=_generic_spec(
            StageId.EXPERIMENT_INIT,
            _experiment_init_parser,
            config_attr="config",
        ),
    )


def _stage_file_spec(
    stage: StageId,
    config_type: type[Any] | object,
    description: str,
) -> StageConfigSpec[Any]:
    return StageConfigSpec(
        stage=stage,
        config_type=config_type,
        parser_factory=lambda: _stage_file_parser(description),
        config_path=_config_path_from_attr("config"),
        cli_patch=_empty_cli_patch,
        registry_patch=_empty_registry_patch,
    )


def _generic_spec(
    stage: StageId,
    parser_factory: Any,
    *,
    config_attr: str | None = None,
) -> StageConfigSpec[GenericStageConfig]:
    return StageConfigSpec(
        stage=stage,
        config_type=GenericStageConfig,
        parser_factory=parser_factory,
        config_path=_config_path_from_attr(config_attr),
        cli_patch=_generic_cli_patch,
        registry_patch=_empty_registry_patch,
    )


def _method_config_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Load a current trainable retrieval method config.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--profile", default=None)
    return parser


def _method_config_profile_name(
    namespace: argparse.Namespace,
    raw: Mapping[str, JsonValue],
) -> str | None:
    if namespace.profile is not None:
        return str(namespace.profile)
    configured = raw.get("default_profile")
    return str(configured) if configured is not None else None


def _validate_method_config_profiles(
    namespace: argparse.Namespace,
    raw: Mapping[str, JsonValue],
) -> ConfigPatch:
    _ = namespace
    if "default_profile" not in raw:
        raise ValueError("Method config requires field: default_profile")
    if "profiles" not in raw:
        raise ValueError("Method config requires field: profiles")
    default_profile = raw["default_profile"]
    profiles = raw["profiles"]
    if not isinstance(default_profile, str) or not default_profile:
        raise ValueError("Method config default_profile must be a non-empty string.")
    if not isinstance(profiles, Mapping):
        raise ValueError("Method config profiles must be an object.")
    if default_profile not in profiles:
        raise ValueError(f"Unknown default method config profile: {default_profile}")

    base = {
        key: value
        for key, value in raw.items()
        if key not in {"default_profile", "profiles"}
    }
    validate_complete_method_config_record(base)
    for name, profile in profiles.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Method config profile names must be non-empty strings.")
        if not isinstance(profile, Mapping):
            raise ValueError(f"Method config profile must be an object: {name}")
        validate_complete_method_config_record(deep_merge_patch(base, profile))
    return {}


def _stage_file_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True)
    return parser


def _prepare_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert labeled HotpotQA examples into HotpotQA ranking and label artifacts.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_input", required=True)
    parser.add_argument("--output_labels", required=True)
    parser.add_argument("--output_combined", default=None)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--strict_invalid_examples", action="store_true")
    return parser


def _graphs_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build typed memory graphs from HotpotQA ranking records.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_query_overlap", type=int, default=20)
    parser.add_argument("--max_entity_neighbors", type=int, default=10)
    parser.add_argument("--max_bridge_edges", type=int, default=50)
    parser.add_argument("--use_spacy", action="store_true")
    return parser


def _tune_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tune graph rerank config.")
    parser.add_argument("--method", required=True)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--graphs", required=True)
    parser.add_argument("--output_config", required=True)
    parser.add_argument("--encoder_model", default="intfloat/e5-base-v2")
    parser.add_argument("--query_prefix", default="query: ")
    parser.add_argument("--passage_prefix", default="passage: ")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--grid_config", default=None)
    return parser


def _aggregate_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate retrieval metrics into report tables.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_main", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--output_efficiency", required=True)
    parser.add_argument("--ablation_index", default=None)
    parser.add_argument("--output_ablation", default=None)
    parser.add_argument("--ablation_selection", action="append", default=[])
    return parser


def _experiment_init_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Initialize a named graph-memory experiment.")
    parser.add_argument("name")
    parser.add_argument("--run-root", dest="run_root", default="runs")
    parser.add_argument("--profile", default=None)
    parser.add_argument("--method", action="append", default=None)
    parser.add_argument("--methods", default=None)
    parser.add_argument("--top-k", dest="top_k", type=int, default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--force", action="store_true")
    return parser


def _generic_cli_patch(namespace: argparse.Namespace) -> ConfigPatch:
    return {"args": dict(vars(namespace))}


def _empty_cli_patch(namespace: argparse.Namespace) -> ConfigPatch:
    return {}


def _config_path_from_attr(name: str | None) -> Any:
    def config_path(namespace: argparse.Namespace) -> Path | None:
        if name is None:
            return None
        value = getattr(namespace, name)
        return Path(value) if value is not None else None

    return config_path


def _empty_registry_patch(
    namespace: argparse.Namespace,
    raw: Mapping[str, JsonValue],
) -> ConfigPatch:
    return {}


__all__ = [
    "DenseFinetuneTrainStageConfig",
    "EvaluateStageConfig",
    "PairBuildStageConfig",
    "RetrieveStageConfig",
    "RgcnTrainStageConfig",
    "StageConfigRegistry",
    "TrainStageConfig",
    "build_stage_config_registry",
]

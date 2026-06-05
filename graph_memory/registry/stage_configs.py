from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, cast

from graph_memory.config.patches import ConfigPatch, deep_merge_patch
from graph_memory.contracts.common import JsonValue
from graph_memory.registry.ids import StageId
from graph_memory.registry.retrieval import (
    RETRIEVAL_METHOD_METADATA,
    DenseEncoderSettings,
    RetrievalJobSettings,
    RetrievalMethodId,
)
from graph_memory.registry.specs import StageConfigSpec
from graph_memory.registry.training import (
    ModelSelectionSettings,
    RgcnMethodSettings,
    RgcnModelSettings,
    RgcnPairSamplingSettings,
    RgcnTrainerSettings,
    TrainingReportingSettings,
    TrainJobSettings,
)
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
    config: Path | None = None


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
    graph_config: Path | None = None
    encoder_model: str = "intfloat/e5-base-v2"
    query_prefix: str = "query: "
    passage_prefix: str = "passage: "


@dataclass(frozen=True)
class RetrieveStageConfig:
    io: RetrieveIO
    job: RetrievalJobSettings

    io_type: ClassVar[type[RetrieveIO]] = RetrieveIO


@dataclass(frozen=True)
class TrainIO:
    train_tasks: Path
    train_labels: Path
    train_graphs: Path
    train_pairs: Path
    dev_tasks: Path
    dev_labels: Path
    dev_graphs: Path
    output_dir: Path
    checkpoint_dir: Path
    metrics: Path
    run_summary: Path
    config: Path | None = None


@dataclass(frozen=True)
class TrainStageConfig:
    io: TrainIO
    job: TrainJobSettings

    io_type: ClassVar[type[TrainIO]] = TrainIO


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


_PAIR_SAMPLING_FIELDS = (
    "random_seed",
    "easy_random_per_positive",
    "hard_bm25_per_positive",
    "hard_dense_per_positive",
    "hard_graph_neighbor_per_positive",
    "hard_pool_size",
)


@dataclass(frozen=True)
class StageConfigRegistry:
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
        PREPARE=_generic_spec(StageId.PREPARE, _prepare_parser),
        GRAPHS=_generic_spec(StageId.GRAPHS, _graphs_parser),
        PAIRS=StageConfigSpec(
            stage=StageId.PAIRS,
            config_type=PairBuildStageConfig,
            parser_factory=_pairs_parser,
            config_path=_config_path_from_attr("config"),
            profile_name=_no_profile,
            cli_patch=_pairs_cli_patch,
            registry_patch=_empty_registry_patch,
            normalize_raw_config=_normalize_pairs_raw_config,
        ),
        TUNE=_generic_spec(StageId.TUNE, _tune_parser),
        TRAIN=StageConfigSpec(
            stage=StageId.TRAIN,
            config_type=TrainStageConfig,
            parser_factory=_train_parser,
            config_path=_config_path_from_attr("config"),
            profile_name=_no_profile,
            cli_patch=_train_cli_patch,
            registry_patch=_empty_registry_patch,
            normalize_raw_config=_normalize_train_raw_config,
        ),
        RETRIEVE=StageConfigSpec(
            stage=StageId.RETRIEVE,
            config_type=RetrieveStageConfig,
            parser_factory=_retrieve_parser,
            config_path=_no_config_path,
            profile_name=_no_profile, # HUMAN REVIEW POINT, profile name 有何意义？没有任何使用者
            cli_patch=_retrieve_cli_patch,
            registry_patch=_empty_registry_patch,
        ),
        EVALUATE=StageConfigSpec(
            stage=StageId.EVALUATE,
            config_type=EvaluateStageConfig,
            parser_factory=_evaluate_parser,
            config_path=_no_config_path,
            profile_name=_no_profile,
            cli_patch=_evaluate_cli_patch,
            registry_patch=_empty_registry_patch,
        ),
        AGGREGATE=_generic_spec(StageId.AGGREGATE, _aggregate_parser),
        EXPERIMENT_INIT=_generic_spec(StageId.EXPERIMENT_INIT, _experiment_init_parser, config_attr="config"),
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
        profile_name=_no_profile,
        cli_patch=_generic_cli_patch,
        registry_patch=_empty_registry_patch,
    )


def _retrieve_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Phase 1 retrieval methods.")
    parser.add_argument("--method", required=True, choices=tuple(RETRIEVAL_METHOD_METADATA))
    parser.add_argument("--tasks", required=True, help="Path to *_memory_tasks.input.json.")
    parser.add_argument("--graphs", default=None, help="Path to *_graphs.json. Required for graph rerank methods.")
    parser.add_argument("--output", required=True, help="Path to write ranked result JSON.")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--encoder_model", default="intfloat/e5-base-v2")
    parser.add_argument("--query_prefix", default="query: ")
    parser.add_argument("--passage_prefix", default="passage: ")
    parser.add_argument("--graph_config", default=None, help="Path to graph rerank config JSON.")
    parser.add_argument("--checkpoint", default=None, help="Path to trainable retriever checkpoint.")
    parser.add_argument("--device", default="cpu", help="Torch device for trainable retriever inference.")
    return parser


def _retrieve_cli_patch(namespace: argparse.Namespace) -> ConfigPatch:
    output = Path(namespace.output)
    return {
        "io": {
            "tasks": namespace.tasks,
            "graphs": namespace.graphs,
            "graph_config": namespace.graph_config,
            "output": namespace.output,
            "summary": output.with_name(f"{output.stem}.run_summary.json"),
            "encoder_model": namespace.encoder_model,
            "query_prefix": namespace.query_prefix,
            "passage_prefix": namespace.passage_prefix,
        },
        "job": _retrieval_job_patch(namespace),
    }


def _retrieval_job_patch(namespace: argparse.Namespace) -> dict[str, object]:
    encoder = {
        "model_name": namespace.encoder_model,
        "query_prefix": namespace.query_prefix,
        "passage_prefix": namespace.passage_prefix,
    }
    method = RetrievalMethodId(namespace.method)
    if method is RetrievalMethodId.BM25:
        return {"method": method.value, "top_k": namespace.top_k}
    if method is RetrievalMethodId.DENSE:
        return {"method": method.value, "top_k": namespace.top_k, "encoder": encoder}
    if method in {RetrievalMethodId.BM25_GRAPH_RERANK, RetrievalMethodId.DENSE_GRAPH_RERANK}:
        seed_method = RetrievalMethodId.DENSE if method is RetrievalMethodId.DENSE_GRAPH_RERANK else RetrievalMethodId.BM25
        seed: dict[str, object] = {"method": seed_method.value}
        if seed_method is RetrievalMethodId.DENSE:
            seed["encoder"] = encoder
        return {
            "method": method.value,
            "top_k": namespace.top_k,
            "seed": seed,
            "rerank": {},
        }
    return {
        "method": method.value,
        "top_k": namespace.top_k,
        "checkpoint": namespace.checkpoint,
        "device": namespace.device,
    }


def _prepare_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert labeled HotpotQA examples into graph-memory artifacts.")
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
    parser = argparse.ArgumentParser(description="Build typed memory graphs from task inputs.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_query_overlap", type=int, default=20)
    parser.add_argument("--max_entity_neighbors", type=int, default=10)
    parser.add_argument("--max_bridge_edges", type=int, default=50)
    parser.add_argument("--use_spacy", action="store_true")
    return parser


def _pairs_parser() -> argparse.ArgumentParser:
    defaults = NegativeSamplingConfig()
    parser = argparse.ArgumentParser(description="Build train pair artifacts for the trainable graph retriever.")
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--graphs", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--random_seed", type=int, default=defaults.random_seed)
    parser.add_argument("--easy_random_per_positive", type=int, default=defaults.easy_random_per_positive)
    parser.add_argument("--hard_bm25_per_positive", type=int, default=defaults.hard_bm25_per_positive)
    parser.add_argument("--hard_dense_per_positive", type=int, default=defaults.hard_dense_per_positive)
    parser.add_argument("--hard_graph_neighbor_per_positive", type=int, default=defaults.hard_graph_neighbor_per_positive)
    parser.add_argument("--hard_pool_size", type=int, default=defaults.hard_pool_size)
    parser.add_argument("--config", default=None)
    return parser


def _pairs_cli_patch(namespace: argparse.Namespace) -> ConfigPatch:
    output = Path(namespace.output)
    sampling_patch: dict[str, object] = {}
    for field_name in _PAIR_SAMPLING_FIELDS:
        if namespace.config is None or _cli_option_was_provided(namespace, field_name):
            sampling_patch[field_name] = getattr(namespace, field_name)
    return {
        "io": {
            "tasks": namespace.tasks,
            "labels": namespace.labels,
            "graphs": namespace.graphs,
            "output": namespace.output,
            "summary": output.with_name(f"{output.stem}.summary.json"),
            "run_summary": output.with_name(f"{output.stem}.run_summary.json"),
            "config": namespace.config,
        },
        "job": {"sampling": sampling_patch},
    }


def _normalize_pairs_raw_config(
    namespace: argparse.Namespace,
    raw: Mapping[str, JsonValue],
) -> Mapping[str, JsonValue]:
    if not raw or "io" in raw or "job" in raw:
        return raw
    resolved = _resolve_legacy_training_config(raw) if "defaults" in raw else raw
    job: dict[str, JsonValue] = {}
    pair_sampling = resolved.get("pair_sampling")
    if pair_sampling is not None:
        job["sampling"] = _json_object(pair_sampling, name="Pair sampling config")
    encoder = resolved.get("encoder")
    if encoder is not None:
        encoder_config = _json_object(encoder, name="Pair encoder config")
        job["hard_dense_encoder"] = {
            "model_name": _string_config_value(encoder_config, "model"),
            "query_prefix": _string_config_value(encoder_config, "query_prefix"),
            "passage_prefix": _string_config_value(encoder_config, "passage_prefix"),
        }
    return {"job": job} if job else {}


def _resolve_legacy_training_config(raw: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
    method = _string_config_value(raw, "method")
    defaults = _json_object(raw.get("defaults"), name="Training config defaults")
    profile_name = _legacy_profile_name(raw)
    profiles = raw.get("profiles", {})
    if not isinstance(profiles, Mapping):
        raise ValueError("Training config profiles must be an object.")
    if profile_name not in profiles:
        raise ValueError(f"Unknown training config profile: {profile_name}")
    profile = _json_object(profiles[profile_name], name=f"Training config profile: {profile_name}")
    resolved = deep_merge_patch(defaults, profile)
    return {
        "schema_version": raw.get("schema_version", 1),
        "method": method,
        "profile": profile_name,
        **resolved,
    }


def _legacy_profile_name(raw: Mapping[str, JsonValue]) -> str:
    value = raw.get("default_profile", "quick")
    if not isinstance(value, str) or not value:
        raise ValueError("Training config requires a non-empty default_profile.")
    return value


def _cli_option_was_provided(namespace: argparse.Namespace, name: str) -> bool:
    provided = getattr(namespace, "_provided_options", frozenset())
    return name in provided


def _json_object(value: JsonValue, *, name: str) -> dict[str, JsonValue]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object.")
    return {str(key): cast(JsonValue, item) for key, item in value.items()}


def _string_config_value(config: Mapping[str, JsonValue], name: str) -> str:
    value = config.get(name)
    if not isinstance(value, str):
        raise ValueError(f"Config field must be a string: {name}")
    return value


def _string_config_alias(config: Mapping[str, JsonValue], primary: str, legacy: str) -> str:
    if primary in config:
        return _string_config_value(config, primary)
    return _string_config_value(config, legacy)


def _json_int(config: Mapping[str, JsonValue], name: str) -> int:
    value = config.get(name)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"Config field must be an integer: {name}")
    return value


def _json_float(config: Mapping[str, JsonValue], name: str) -> float:
    value = config.get(name)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"Config field must be numeric: {name}")
    return float(value)


def _json_bool_alias(config: Mapping[str, JsonValue], primary: str, legacy: str) -> bool:
    name = primary if primary in config else legacy
    value = config.get(name)
    if not isinstance(value, bool):
        raise ValueError(f"Config field must be boolean: {name}")
    return value


def _tune_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tune graph rerank config.")
    parser.add_argument("--method", required=True, choices=("bm25_graph_rerank", "dense_graph_rerank"))
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


def _train_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the Phase 2 R-GCN graph retriever.")
    parser.add_argument("--train_tasks", required=True)
    parser.add_argument("--train_labels", required=True)
    parser.add_argument("--train_graphs", required=True)
    parser.add_argument("--train_pairs", required=True)
    parser.add_argument("--dev_tasks", required=True)
    parser.add_argument("--dev_labels", required=True)
    parser.add_argument("--dev_graphs", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--encoder_model", default="intfloat/e5-base-v2")
    parser.add_argument("--query_prefix", default="query: ")
    parser.add_argument("--passage_prefix", default="passage: ")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--ablation",
        default="full_rgcn",
        choices=["full_rgcn", "wo_graph", "wo_edge_type", "wo_bridge", "wo_edge_weight", "wo_seed_score"],
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--random_seed", type=int, default=13)
    parser.add_argument("--pos_weight", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--config", default=None)
    return parser


def _train_cli_patch(namespace: argparse.Namespace) -> ConfigPatch:
    output_dir = Path(namespace.output_dir)
    return {
        "io": {
            "train_tasks": namespace.train_tasks,
            "train_labels": namespace.train_labels,
            "train_graphs": namespace.train_graphs,
            "train_pairs": namespace.train_pairs,
            "dev_tasks": namespace.dev_tasks,
            "dev_labels": namespace.dev_labels,
            "dev_graphs": namespace.dev_graphs,
            "output_dir": namespace.output_dir,
            "checkpoint_dir": output_dir / "checkpoints",
            "metrics": output_dir / "train_metrics.jsonl",
            "run_summary": output_dir / "train_run_summary.json",
            "config": namespace.config,
        },
        "job": _train_job_cli_patch(namespace),
    }


def _train_job_cli_patch(namespace: argparse.Namespace) -> dict[str, object]:
    job: dict[str, object] = {}
    encoder = _train_cli_section_patch(
        namespace,
        {
            "encoder_model": "model_name",
            "query_prefix": "query_prefix",
            "passage_prefix": "passage_prefix",
        },
    )
    if encoder:
        job["encoder"] = encoder
    model = _train_cli_section_patch(
        namespace,
        {
            "hidden_dim": "hidden_dim",
            "num_layers": "num_layers",
            "dropout": "dropout",
            "ablation": "ablation",
        },
    )
    if model:
        job["model"] = model
    trainer = _train_cli_section_patch(
        namespace,
        {
            "learning_rate": "learning_rate",
            "batch_size": "batch_size",
            "max_grad_norm": "max_grad_norm",
            "random_seed": "random_seed",
            "epochs": "epochs",
            "device": "device",
            "pos_weight": "pos_weight_enabled",
        },
    )
    if namespace.config is None:
        trainer["optimizer_name"] = "AdamW"
    if trainer:
        job["trainer"] = trainer
    if namespace.config is None:
        defaults = RgcnPairSamplingSettings()
        job["pairs"] = {
            "random_seed": defaults.random_seed,
            "easy_random_per_positive": defaults.easy_random_per_positive,
            "hard_bm25_per_positive": defaults.hard_bm25_per_positive,
            "hard_dense_per_positive": defaults.hard_dense_per_positive,
            "hard_graph_neighbor_per_positive": defaults.hard_graph_neighbor_per_positive,
            "hard_pool_size": defaults.hard_pool_size,
        }
    return {"method": RetrievalMethodId.DENSE_RGCN_GRAPH_RETRIEVER.value, **job}


def _train_cli_section_patch(namespace: argparse.Namespace, fields: Mapping[str, str]) -> dict[str, object]:
    patch: dict[str, object] = {}
    for source, target in fields.items():
        if namespace.config is None or _cli_option_was_provided(namespace, source):
            patch[target] = getattr(namespace, source)
    return patch


def _normalize_train_raw_config(
    namespace: argparse.Namespace,
    raw: Mapping[str, JsonValue],
) -> Mapping[str, JsonValue]:
    if not raw or "io" in raw or "job" in raw:
        return raw
    resolved = _resolve_legacy_training_config(raw) if "defaults" in raw else raw
    return {"job": _train_job_from_resolved_config(resolved)}


def _train_job_from_resolved_config(raw: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    job: dict[str, JsonValue] = {
        "method": _string_config_value(raw, "method"),
        "encoder": _train_encoder_from_config(_json_object(raw.get("encoder"), name="Training encoder config")),
        "model": _train_model_from_config(_json_object(raw.get("model"), name="Training model config")),
    }
    trainer_source = raw.get("trainer", raw.get("optimization"))
    if trainer_source is None:
        raise ValueError("Training config requires object section: optimization")
    job["trainer"] = _train_trainer_from_config(_json_object(trainer_source, name="Training trainer config"))
    pair_sampling = raw.get("pairs", raw.get("pair_sampling"))
    if pair_sampling is not None:
        job["pairs"] = _json_object(pair_sampling, name="Training pair sampling config")
    reporting = raw.get("reporting")
    if reporting is not None:
        job["reporting"] = _json_object(reporting, name="Training reporting config")
    selection = raw.get("selection")
    if selection is not None:
        job["selection"] = _json_object(selection, name="Training selection config")
    return job


def _train_encoder_from_config(config: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    return {
        "model_name": _string_config_alias(config, "model_name", "model"),
        "query_prefix": _string_config_value(config, "query_prefix"),
        "passage_prefix": _string_config_value(config, "passage_prefix"),
    }


def _train_model_from_config(config: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    return {
        "hidden_dim": _json_int(config, "hidden_dim"),
        "num_layers": _json_int(config, "num_layers"),
        "dropout": _json_float(config, "dropout"),
        "ablation": _string_config_alias(config, "ablation", "ablation_name"),
    }


def _train_trainer_from_config(config: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    trainer: dict[str, JsonValue] = {
        "optimizer_name": _string_config_alias(config, "optimizer_name", "optimizer"),
        "learning_rate": _json_float(config, "learning_rate"),
        "batch_size": _json_int(config, "batch_size"),
        "max_grad_norm": _json_float(config, "max_grad_norm"),
        "random_seed": _json_int(config, "random_seed"),
        "pos_weight_enabled": _json_bool_alias(config, "pos_weight_enabled", "pos_weight"),
        "epochs": _json_int(config, "epochs"),
    }
    device = config.get("device")
    if device is not None:
        if not isinstance(device, str) or not device:
            raise ValueError("Training config field must be a non-empty string: device")
        trainer["device"] = device
    return trainer


def _evaluate_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Phase 1 ranked retrieval results.")
    parser.add_argument("--pred", required=True)
    parser.add_argument("--labels", default=None)
    parser.add_argument("--gold", default=None)
    parser.add_argument("--graphs", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--failure_cases_output", default=None)
    parser.add_argument("--failure_case_limit", type=int, default=0)
    return parser


def _evaluate_cli_patch(namespace: argparse.Namespace) -> ConfigPatch:
    label_path = namespace.labels or namespace.gold
    if label_path is None:
        raise ValueError("--labels is required; --gold is accepted as a compatibility alias.")
    return {
        "io": {
            "predictions": namespace.pred,
            "labels": label_path,
            "graphs": namespace.graphs,
            "output": namespace.output,
            "failure_cases_output": namespace.failure_cases_output,
        },
        "failure_case_limit": namespace.failure_case_limit,
    }


def _aggregate_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate per-method metric CSVs into final tables.")
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


def _config_path_from_attr(name: str | None) -> Any:
    def config_path(namespace: argparse.Namespace) -> Path | None:
        if name is None:
            return None
        value = getattr(namespace, name)
        return Path(value) if value is not None else None

    return config_path


def _no_config_path(namespace: argparse.Namespace) -> Path | None:
    return None


def _no_profile(namespace: argparse.Namespace, raw: Mapping[str, JsonValue]) -> str | None:
    return None


def _empty_registry_patch(namespace: argparse.Namespace, raw: Mapping[str, JsonValue]) -> ConfigPatch:
    return {}


__all__ = [
    "GenericStageConfig",
    "EvaluateIO",
    "EvaluateStageConfig",
    "ModelSelectionSettings",
    "PairBuildIO",
    "PairBuildJobSettings",
    "PairBuildStageConfig",
    "PairSamplingSettings",
    "RgcnMethodSettings",
    "RgcnModelSettings",
    "RgcnPairSamplingSettings",
    "RgcnTrainerSettings",
    "RetrieveIO",
    "RetrieveStageConfig",
    "StageConfigRegistry",
    "TrainIO",
    "TrainStageConfig",
    "TrainingReportingSettings",
    "build_stage_config_registry",
]

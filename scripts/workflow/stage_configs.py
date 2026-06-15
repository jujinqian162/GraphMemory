from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from typing_extensions import assert_never

from graph_memory.config import CONFIG_LOADER
from graph_memory.io import write_json
from graph_memory.registry import Registry
from graph_memory.registry.method_configs import (
    DenseFinetuneMethodConfig,
    DenseFinetuneMethodSettings,
    RgcnMethodConfig,
    RgcnMethodSettings,
    TrainableMethodConfig,
)
from graph_memory.registry.methods import (
    GraphConfigSource,
    GraphInputSource,
    ModelSource,
)
from graph_memory.registry.retrieval import (
    Bm25RetrievalSettings,
    CheckpointGraphRetrievalSettings,
    DenseEncoderSettings,
    DenseFinetunedRetrievalSettings,
    DenseRetrievalSettings,
    GraphRerankRetrievalSettings,
    GraphRerankSettings,
    MemoryStreamRetrievalSettings,
    RetrievalJobSettings,
    RetrievalMethodId,
    SeedRetrievalSettings,
)
from graph_memory.retrieval.methods.memory_stream.config import MemoryStreamScoringConfig
from graph_memory.registry.stage_configs import (
    DenseFinetuneTrainIO,
    DenseFinetuneTrainStageConfig,
    EvaluateIO,
    EvaluateStageConfig,
    PairBuildIO,
    PairBuildJobSettings,
    PairBuildStageConfig,
    PairSamplingSettings,
    RetrieveIO,
    RetrieveStageConfig,
    RgcnTrainIO,
    RgcnTrainStageConfig,
    TrainStageConfig,
)


def load_trainable_method_configs(
    effective_config: Mapping[str, Any],
    selected_methods: Sequence[str],
) -> dict[str, TrainableMethodConfig]:
    configured_paths = effective_config.get("method_configs", {})
    if not isinstance(configured_paths, Mapping):
        raise ValueError("Experiment config method_configs must be an object.")
    profile = str(effective_config["profile"])
    loaded: dict[str, TrainableMethodConfig] = {}
    for method in selected_methods:
        definition = Registry.methods.get(method)
        if definition.method_config_type is None:
            continue
        configured_path = configured_paths.get(method)
        if not isinstance(configured_path, str) or not configured_path:
            raise ValueError(f"Trainable method={method} requires a method config path.")
        config = CONFIG_LOADER.load(
            Registry.configs.TRAINABLE_METHOD,
            ["--config", configured_path, "--profile", profile],
        )
        if config.method.value != method:
            raise ValueError(
                f"Method config method={config.method.value} does not match selected method={method}."
            )
        if not isinstance(config, definition.method_config_type):
            raise TypeError(
                f"Method config for {method} must be {definition.method_config_type.__name__}, "
                f"got {type(config).__name__}."
            )
        loaded[method] = config
    return loaded


def write_main_stage_configs(
    manifest: dict[str, Any],
    method_configs: Mapping[str, TrainableMethodConfig],
) -> None:
    stage_paths: dict[str, dict[str, str]] = {
        "pairs": {},
        "train": {},
        "retrieve": {},
        "evaluate": {},
    }
    root = Path(manifest["paths"]["run_dir"]) / "config" / "stages"
    for method in manifest["selected_methods"]:
        configs = _build_method_stage_configs(manifest, method, method_configs.get(method))
        for stage, config in configs.items():
            path = root / stage / f"{method}.json"
            _write_stage_config(path, config)
            stage_paths[stage][method] = path.as_posix()
    manifest["stage_configs"] = stage_paths


def write_variant_stage_configs(
    manifest: dict[str, Any],
    *,
    method: str,
    variant: str,
    method_config: TrainableMethodConfig,
    record: dict[str, Any],
) -> dict[str, str]:
    variant_manifest = _variant_manifest(manifest, method, record)
    configs = _build_method_stage_configs(variant_manifest, method, method_config)
    root = (
        Path(manifest["paths"]["run_dir"])
        / "config"
        / "stages"
        / "ablations"
        / method
        / variant
    )
    paths: dict[str, str] = {}
    for stage, config in configs.items():
        path = root / f"{stage}.json"
        _write_stage_config(path, config)
        paths[stage] = path.as_posix()
    return paths


def _build_method_stage_configs(
    manifest: dict[str, Any],
    method: str,
    method_config: TrainableMethodConfig | None,
) -> dict[str, object]:
    configs: dict[str, object] = {
        "retrieve": _retrieve_stage_config(manifest, method, method_config),
        "evaluate": _evaluate_stage_config(manifest, method),
    }
    if method_config is not None:
        configs["pairs"] = _pair_stage_config(manifest, method, method_config)
        configs["train"] = _train_stage_config(manifest, method, method_config)
    return configs


def _pair_stage_config(
    manifest: dict[str, Any],
    method: str,
    method_config: TrainableMethodConfig,
) -> PairBuildStageConfig:
    learned = manifest["artifacts"]["learned"][method]
    pairs = method_config.pairs
    return PairBuildStageConfig(
        io=PairBuildIO(
            tasks=Path(manifest["artifacts"]["inputs"]["train"]["input"]),
            labels=Path(manifest["artifacts"]["inputs"]["train"]["labels"]),
            graphs=Path(manifest["artifacts"]["graphs"]["train"]),
            output=Path(learned["train_pairs"]),
            summary=Path(learned["train_pair_summary"]),
            run_summary=Path(learned["train_pair_run_summary"]),
        ),
        job=PairBuildJobSettings(
            sampling=PairSamplingSettings(
                random_seed=pairs.random_seed,
                easy_random_per_positive=pairs.easy_random_per_positive,
                hard_bm25_per_positive=pairs.hard_bm25_per_positive,
                hard_dense_per_positive=pairs.hard_dense_per_positive,
                hard_graph_neighbor_per_positive=pairs.hard_graph_neighbor_per_positive,
                hard_pool_size=pairs.hard_pool_size,
            ),
            hard_dense_encoder=method_config.encoder,
        ),
    )


def _train_stage_config(
    manifest: dict[str, Any],
    method: str,
    method_config: TrainableMethodConfig,
) -> TrainStageConfig:
    learned = manifest["artifacts"]["learned"][method]
    if isinstance(method_config, RgcnMethodConfig):
        return RgcnTrainStageConfig(
            method=RetrievalMethodId.DENSE_RGCN_GRAPH_RETRIEVER,
            io=RgcnTrainIO(
                train_tasks=Path(manifest["artifacts"]["inputs"]["train"]["input"]),
                train_labels=Path(manifest["artifacts"]["inputs"]["train"]["labels"]),
                train_graphs=Path(manifest["artifacts"]["graphs"]["train"]),
                train_pairs=Path(learned["train_pairs"]),
                dev_tasks=Path(manifest["artifacts"]["inputs"]["dev"]["input"]),
                dev_labels=Path(manifest["artifacts"]["inputs"]["dev"]["labels"]),
                dev_graphs=Path(manifest["artifacts"]["graphs"]["dev"]),
                output_dir=Path(learned["training_output_dir"]),
                checkpoint_dir=Path(learned["best_checkpoint"]).parent,
                metrics=Path(learned["train_metrics"]),
                run_summary=Path(learned["train_run_summary"]),
            ),
            job=RgcnMethodSettings(
                encoder=method_config.encoder,
                model=method_config.train.model,
                trainer=method_config.train.trainer,
                pairs=method_config.pairs,
                reporting=method_config.train.reporting,
                selection=method_config.train.selection,
            ),
        )
    if isinstance(method_config, DenseFinetuneMethodConfig):
        return DenseFinetuneTrainStageConfig(
            method=RetrievalMethodId.DENSE_FT,
            io=DenseFinetuneTrainIO(
                train_tasks=Path(manifest["artifacts"]["inputs"]["train"]["input"]),
                train_labels=Path(manifest["artifacts"]["inputs"]["train"]["labels"]),
                train_pairs=Path(learned["train_pairs"]),
                dev_tasks=Path(manifest["artifacts"]["inputs"]["dev"]["input"]),
                dev_labels=Path(manifest["artifacts"]["inputs"]["dev"]["labels"]),
                output_dir=Path(learned["training_output_dir"]),
                model_dir=Path(learned["best_checkpoint"]),
                metrics=Path(learned["train_metrics"]),
                run_summary=Path(learned["train_run_summary"]),
            ),
            job=DenseFinetuneMethodSettings(
                encoder=method_config.encoder,
                data=method_config.train.data,
                trainer=method_config.train.trainer,
                selection=method_config.train.selection,
            ),
        )
    assert_never(method_config)


def _retrieve_stage_config(
    manifest: dict[str, Any],
    method: str,
    method_config: TrainableMethodConfig | None,
) -> RetrieveStageConfig:
    definition = Registry.methods.get(method)
    graph_path = (
        Path(manifest["artifacts"]["graphs"]["test"])
        if definition.dependencies.graphs is GraphInputSource.GRAPH_ARTIFACT
        else None
    )
    graph_config = (
        Path(manifest["artifacts"]["tuned"][method])
        if definition.dependencies.graph_config is GraphConfigSource.TUNED_ARTIFACT
        else None
    )
    return RetrieveStageConfig(
        io=RetrieveIO(
            tasks=Path(manifest["artifacts"]["inputs"]["test"]["input"]),
            graphs=graph_path,
            graph_config=graph_config,
            importance=_memory_stream_importance_path(manifest, method),
            output=Path(manifest["artifacts"]["predictions"][method]),
            summary=Path(manifest["artifacts"]["predictions"][method]).with_name(
                f"{Path(manifest['artifacts']['predictions'][method]).stem}.run_summary.json"
            ),
        ),
        job=_retrieval_job(manifest, method, method_config),
    )


def _retrieval_job(
    manifest: dict[str, Any],
    method: str,
    method_config: TrainableMethodConfig | None,
) -> RetrievalJobSettings:
    method_id = RetrievalMethodId(method)
    top_k = int(manifest["effective_config"]["top_k"])
    definition = Registry.methods.get(method_id)
    if method_id is RetrievalMethodId.BM25:
        return Bm25RetrievalSettings(top_k=top_k)
    if method_id is RetrievalMethodId.DENSE:
        return DenseRetrievalSettings(top_k=top_k, encoder=_experiment_encoder(manifest))
    if method_id is RetrievalMethodId.MEMORY_STREAM:
        return MemoryStreamRetrievalSettings(
            top_k=top_k,
            encoder=_experiment_encoder(manifest),
            scoring=MemoryStreamScoringConfig(
                relevance_weight=float(
                    manifest["effective_config"].get(
                        "memory_stream_relevance_weight",
                        1.0,
                    )
                ),
                recency_weight=float(
                    manifest["effective_config"].get(
                        "memory_stream_recency_weight",
                        0.0,
                    )
                ),
                importance_weight=float(
                    manifest["effective_config"].get(
                        "memory_stream_importance_weight",
                        0.01,
                    )
                ),
                recency_decay=float(
                    manifest["effective_config"].get(
                        "memory_stream_recency_decay",
                        0.99,
                    )
                ),
            ),
            capped_test_count=int(manifest["effective_config"]["splits"]["test"]["max_examples"]),
        )
    if method_id is RetrievalMethodId.BM25_GRAPH_RERANK:
        seed_method = definition.seed_method
        if seed_method is not RetrievalMethodId.BM25:
            raise ValueError(f"BM25 graph rerank requires BM25 seed method: {method}")
        return GraphRerankRetrievalSettings(
            method=method_id,
            top_k=top_k,
            seed=SeedRetrievalSettings(
                method=seed_method,
                encoder=None,
            ),
            rerank=GraphRerankSettings(),
        )
    if method_id is RetrievalMethodId.DENSE_GRAPH_RERANK:
        seed_method = definition.seed_method
        if seed_method is not RetrievalMethodId.DENSE:
            raise ValueError(f"Dense graph rerank requires dense seed method: {method}")
        return GraphRerankRetrievalSettings(
            method=method_id,
            top_k=top_k,
            seed=SeedRetrievalSettings(
                method=seed_method,
                encoder=_experiment_encoder(manifest),
            ),
            rerank=GraphRerankSettings(),
        )
    if method_config is None:
        raise ValueError(f"Trainable retrieval method requires a method config: {method}")
    checkpoint = Path(manifest["artifacts"]["learned"][method]["best_checkpoint"])
    if definition.dependencies.model is ModelSource.CHECKPOINT_FILE:
        if not isinstance(method_config, RgcnMethodConfig):
            raise TypeError(f"R-GCN retrieval requires RgcnMethodConfig, got {type(method_config).__name__}.")
        return CheckpointGraphRetrievalSettings(
            top_k=top_k,
            checkpoint=checkpoint,
            device=method_config.train.trainer.device,
        )
    if definition.dependencies.model is ModelSource.MODEL_DIRECTORY:
        if not isinstance(method_config, DenseFinetuneMethodConfig):
            raise TypeError(
                f"Dense-FT retrieval requires DenseFinetuneMethodConfig, got {type(method_config).__name__}."
            )
        return DenseFinetunedRetrievalSettings(
            top_k=top_k,
            checkpoint=checkpoint,
            device=method_config.train.trainer.device,
        )
    raise ValueError(f"Unsupported retrieval model source for method={method}: {definition.dependencies.model}")


def _evaluate_stage_config(manifest: dict[str, Any], method: str) -> EvaluateStageConfig:
    return EvaluateStageConfig(
        io=EvaluateIO(
            predictions=Path(manifest["artifacts"]["predictions"][method]),
            labels=Path(manifest["artifacts"]["inputs"]["test"]["labels"]),
            graphs=Path(manifest["artifacts"]["graphs"]["test"]),
            output=Path(manifest["artifacts"]["metrics"][method]),
            failure_cases_output=Path(manifest["artifacts"]["failure_cases"][method]),
        ),
        failure_case_limit=50,
    )


def _memory_stream_importance_path(manifest: Mapping[str, Any], method: str) -> Path | None:
    """Return the external cleaned importance artifact path for Memory Stream only."""
    if method != RetrievalMethodId.MEMORY_STREAM.value:
        return None
    configured = manifest["effective_config"].get("memory_stream_importance_path")
    if configured is not None:
        return Path(str(configured))
    return Path("data/hotpotqa/processed/memory_stream/dev.first_1000.importance.json")


def _experiment_encoder(manifest: Mapping[str, Any]) -> DenseEncoderSettings:
    config = manifest["effective_config"]
    return DenseEncoderSettings(
        model_name=str(config["dense_encoder"]),
        query_prefix=str(config["query_prefix"]),
        passage_prefix=str(config["passage_prefix"]),
    )


def _write_stage_config(path: Path, config: object) -> None:
    value = CONFIG_LOADER.to_json(config)
    if not isinstance(value, dict):
        raise ValueError(f"Stage config must serialize to an object: {path}")
    write_json(path, value)


def _variant_manifest(
    manifest: dict[str, Any],
    method: str,
    record: Mapping[str, Any],
) -> dict[str, Any]:
    artifacts = record["artifacts"]
    if not isinstance(artifacts, Mapping):
        raise ValueError(f"Variant artifacts must be an object: {method}")
    resolved = {
        **manifest,
        "artifacts": {
            **manifest["artifacts"],
            "learned": {
                **manifest["artifacts"]["learned"],
                method: {
                    **manifest["artifacts"]["learned"][method],
                    "train_pairs": artifacts["train_pairs"],
                    "train_pair_summary": artifacts["train_pair_summary"],
                    "train_pair_run_summary": artifacts["train_pair_run_summary"],
                    "effective_method_config": artifacts["effective_method_config"],
                    "training_output_dir": str(Path(str(artifacts["checkpoint"])).parents[1]),
                    "train_metrics": artifacts["train_metrics"],
                    "train_run_summary": artifacts["train_run_summary"],
                    "best_checkpoint": artifacts["checkpoint"],
                },
            },
            "predictions": {
                **manifest["artifacts"]["predictions"],
                method: artifacts["predictions"],
            },
            "metrics": {
                **manifest["artifacts"]["metrics"],
                method: artifacts["metrics"],
            },
            "failure_cases": {
                **manifest["artifacts"]["failure_cases"],
                method: artifacts["failure_cases"],
            },
        },
    }
    return resolved


__all__ = [
    "load_trainable_method_configs",
    "write_main_stage_configs",
    "write_variant_stage_configs",
]

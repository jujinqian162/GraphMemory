from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import TypeVar

import pytest

from graph_memory.config import CONFIG_LOADER
from graph_memory.io import write_json
from graph_memory.registry import Registry
from graph_memory.registry.ids import StageId
from graph_memory.registry.method_configs import (
    DenseFinetuneMethodSettings,
    RgcnMethodSettings,
    RgcnModelSettings,
    RgcnTrainerSettings,
)
from graph_memory.registry.retrieval import (
    Bm25RetrievalSettings,
    DenseEncoderSettings,
    MemoryStreamRetrievalSettings,
    RetrievalMethodId,
)
from graph_memory.retrieval.methods.memory_stream.config import MemoryStreamScoringConfig
from graph_memory.registry.specs import StageConfigSpec
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
)
from graph_memory.registry.training import DenseFinetuneMethodTrainer, RgcnGraphRetrieverTrainer


def _write_config(path: Path, config: object) -> None:
    write_json(path, CONFIG_LOADER.to_json(config))


ConfigT = TypeVar("ConfigT")


def _assert_config_round_trip(path: Path, spec: StageConfigSpec[ConfigT], expected: ConfigT) -> None:
    _write_config(path, expected)
    assert CONFIG_LOADER.load(spec, ["--config", str(path)]) == expected


def test_registry_exposes_stage_root_config_specs() -> None:
    expected = {
        "TRAINABLE_METHOD": StageId.EXPERIMENT_INIT,
        "PREPARE": StageId.PREPARE,
        "GRAPHS": StageId.GRAPHS,
        "PAIRS": StageId.PAIRS,
        "TUNE": StageId.TUNE,
        "TRAIN": StageId.TRAIN,
        "RETRIEVE": StageId.RETRIEVE,
        "EVALUATE": StageId.EVALUATE,
        "AGGREGATE": StageId.AGGREGATE,
        "EXPERIMENT_INIT": StageId.EXPERIMENT_INIT,
    }

    assert {name: getattr(Registry.configs, name).stage for name in expected} == expected


def test_runtime_stage_parsers_accept_only_complete_config_files() -> None:
    for name in ("PAIRS", "TRAIN", "RETRIEVE", "EVALUATE"):
        parser = getattr(Registry.configs, name).parser_factory()
        options = {
            option
            for action in parser._actions
            for option in action.option_strings
            if option.startswith("--")
        }
        assert options == {"--help", "--config"}


def test_pair_retrieve_and_evaluate_configs_round_trip(tmp_path: Path) -> None:
    _assert_config_round_trip(
        tmp_path / "pairs.json",
        Registry.configs.PAIRS,
        PairBuildStageConfig(
            io=PairBuildIO(
                tasks=tmp_path / "tasks.json",
                labels=tmp_path / "labels.json",
                graphs=tmp_path / "graphs.json",
                output=tmp_path / "pairs.json",
                summary=tmp_path / "pairs.summary.json",
                run_summary=tmp_path / "pairs.run_summary.json",
            ),
            job=PairBuildJobSettings(
                sampling=PairSamplingSettings(
                    random_seed=13,
                    easy_random_per_positive=1,
                    hard_bm25_per_positive=1,
                    hard_dense_per_positive=0,
                    hard_graph_neighbor_per_positive=1,
                    hard_pool_size=30,
                )
            ),
        ),
    )
    _assert_config_round_trip(
        tmp_path / "retrieve.json",
        Registry.configs.RETRIEVE,
        RetrieveStageConfig(
            io=RetrieveIO(
                tasks=tmp_path / "tasks.json",
                graphs=None,
                output=tmp_path / "predictions.json",
                summary=tmp_path / "predictions.run_summary.json",
            ),
            job=Bm25RetrievalSettings(top_k=7),
        ),
    )
    _assert_config_round_trip(
        tmp_path / "evaluate.json",
        Registry.configs.EVALUATE,
        EvaluateStageConfig(
            io=EvaluateIO(
                predictions=tmp_path / "predictions.json",
                labels=tmp_path / "labels.json",
                graphs=tmp_path / "graphs.json",
                output=tmp_path / "metrics.csv",
            ),
            failure_case_limit=3,
        ),
    )


def test_memory_stream_retrieve_config_round_trips_importance_and_cap(tmp_path: Path) -> None:
    _assert_config_round_trip(
        tmp_path / "retrieve-memory-stream.json",
        Registry.configs.RETRIEVE,
        RetrieveStageConfig(
            io=RetrieveIO(
                tasks=tmp_path / "tasks.json",
                graphs=None,
                selected_config=tmp_path / "memory_stream.dev_selected.json",
                output=tmp_path / "predictions.json",
                summary=tmp_path / "predictions.run_summary.json",
                importance=tmp_path / "dev.first_1000.importance.json",
            ),
            job=MemoryStreamRetrievalSettings(
                top_k=7,
                encoder=DenseEncoderSettings(
                    model_name="fake-e5",
                    query_prefix="query: ",
                    passage_prefix="passage: ",
                    batch_size=8,
                ),
                scoring=MemoryStreamScoringConfig(
                    relevance_weight=2.0,
                    recency_weight=1.0,
                    importance_weight=3.0,
                    recency_decay=0.95,
                ),
                capped_test_count=1000,
            ),
        ),
    )


def test_train_config_union_round_trips_rgcn_and_dense_ft(tmp_path: Path) -> None:
    encoder = DenseEncoderSettings(
        model_name="fake-e5",
        query_prefix="query: ",
        passage_prefix="passage: ",
        batch_size=8,
    )
    rgcn = RgcnTrainStageConfig(
        method=RetrievalMethodId.DENSE_RGCN_GRAPH_RETRIEVER,
        io=RgcnTrainIO(
            train_tasks=tmp_path / "train.input.json",
            train_labels=None,
            train_graphs=tmp_path / "train.graphs.json",
            train_pairs=tmp_path / "train.pairs.json",
            dev_tasks=tmp_path / "dev.input.json",
            dev_labels=tmp_path / "dev.labels.json",
            dev_graphs=tmp_path / "dev.graphs.json",
            output_dir=tmp_path / "rgcn",
            checkpoint_dir=tmp_path / "rgcn" / "checkpoints",
            metrics=tmp_path / "rgcn" / "train_metrics.jsonl",
            run_summary=tmp_path / "rgcn" / "train_run_summary.json",
        ),
        job=RgcnMethodSettings(
            encoder=encoder,
            model=RgcnModelSettings(hidden_dim=8, num_layers=1, dropout=0.0),
            trainer=RgcnTrainerSettings(device="cpu"),
        ),
    )
    dense_ft = DenseFinetuneTrainStageConfig(
        method=RetrievalMethodId.DENSE_FT,
        io=DenseFinetuneTrainIO(
            train_tasks=tmp_path / "train.input.json",
            train_labels=tmp_path / "train.labels.json",
            train_pairs=tmp_path / "train.pairs.json",
            dev_tasks=tmp_path / "dev.input.json",
            dev_labels=tmp_path / "dev.labels.json",
            output_dir=tmp_path / "dense-ft",
            model_dir=tmp_path / "dense-ft" / "checkpoints" / "best_model",
            metrics=tmp_path / "dense-ft" / "train_metrics.jsonl",
            run_summary=tmp_path / "dense-ft" / "train_run_summary.json",
        ),
        job=DenseFinetuneMethodSettings(encoder=encoder),
    )

    for name, expected in (("rgcn", rgcn), ("dense-ft", dense_ft)):
        path = tmp_path / f"{name}.json"
        _write_config(path, expected)
        assert CONFIG_LOADER.load(Registry.configs.TRAIN, ["--config", str(path)]) == expected


def test_stage_config_unknown_fields_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "retrieve.json"
    write_json(
        path,
        {
            "io": {
                "tasks": "tasks.json",
                "graphs": None,
                "output": "predictions.json",
                "summary": "predictions.run_summary.json",
                "legacy_encoder_model": "old",
            },
            "job": {"method": "bm25", "top_k": 10},
        },
    )

    with pytest.raises(ValueError, match="unsupported fields.*legacy_encoder_model"):
        CONFIG_LOADER.load(Registry.configs.RETRIEVE, ["--config", str(path)])


def test_training_registry_builds_exact_trainers() -> None:
    encoder = DenseEncoderSettings("fake-e5", "query: ", "passage: ")

    assert isinstance(
        Registry.training.build(
            RgcnMethodSettings(
                encoder=encoder,
                model=RgcnModelSettings(),
                trainer=RgcnTrainerSettings(),
            )
        ),
        RgcnGraphRetrieverTrainer,
    )
    assert isinstance(
        Registry.training.build(DenseFinetuneMethodSettings(encoder=encoder)),
        DenseFinetuneMethodTrainer,
    )


def test_stage_config_roots_are_explicit() -> None:
    assert [field.name for field in fields(RetrieveStageConfig)] == ["io", "job"]
    assert [field.name for field in fields(RgcnTrainStageConfig)] == ["method", "io", "job"]

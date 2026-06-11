from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import pytest

from graph_memory.config import CONFIG_LOADER
from graph_memory.registry import Registry
from graph_memory.registry.ids import StageId
from graph_memory.registry.retrieval import Bm25RetrievalSettings, DenseEncoderSettings, RetrievalMethodId
from graph_memory.registry.training import (
    DenseFinetuneMethodSettings,
    ModelSelectionSettings,
    RgcnMethodSettings,
    RgcnModelSettings,
    RgcnPairSamplingSettings,
    RgcnTrainerSettings,
    TrainingReportingSettings,
)
import graph_memory.registry.stage_configs as stage_configs
from graph_memory.registry.stage_configs import (
    DenseFinetuneTrainStageConfig,
    EvaluateStageConfig,
    RetrieveStageConfig,
    RgcnTrainStageConfig,
)


def test_registry_exposes_stage_root_config_specs() -> None:
    expected = {
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


def test_stage_config_specs_do_not_expose_profile_or_defaults_keys() -> None:
    for name in (
        "PREPARE",
        "GRAPHS",
        "PAIRS",
        "TUNE",
        "TRAIN",
        "RETRIEVE",
        "EVALUATE",
        "AGGREGATE",
        "EXPERIMENT_INIT",
    ):
        spec = getattr(Registry.configs, name)
        assert not hasattr(spec, "profile_key")
        assert not hasattr(spec, "defaults_key")


def test_stage_config_specs_do_not_declare_noop_profile_selectors() -> None:
    for name in (
        "PREPARE",
        "GRAPHS",
        "PAIRS",
        "TUNE",
        "TRAIN",
        "RETRIEVE",
        "EVALUATE",
        "AGGREGATE",
        "EXPERIMENT_INIT",
    ):
        spec = getattr(Registry.configs, name)
        assert spec.profile_name is None


def test_retrieve_stage_config_loads_directly_from_existing_cli_contract(tmp_path: Path) -> None:
    tasks_path = tmp_path / "tasks.json"
    output_path = tmp_path / "predictions.json"

    config = CONFIG_LOADER.load(
        Registry.configs.RETRIEVE,
        [
            "--method",
            "bm25",
            "--tasks",
            str(tasks_path),
            "--output",
            str(output_path),
            "--top_k",
            "7",
        ],
    )

    assert config == RetrieveStageConfig(
        io=RetrieveStageConfig.io_type(
            tasks=tasks_path,
            graphs=None,
            output=output_path,
            summary=tmp_path / "predictions.run_summary.json",
            encoder_model="intfloat/e5-base-v2",
            query_prefix="query: ",
            passage_prefix="passage: ",
        ),
        job=Bm25RetrievalSettings(top_k=7),
    )


def test_retrieve_stage_config_has_io_and_job_roots() -> None:
    assert [field.name for field in fields(RetrieveStageConfig)] == ["io", "job"]


def test_pair_build_stage_config_loads_cli_overrides_after_training_config(tmp_path: Path) -> None:
    assert hasattr(stage_configs, "PairBuildStageConfig")
    PairBuildStageConfig = stage_configs.PairBuildStageConfig
    config_path = tmp_path / "effective_training_config.json"
    tasks_path = tmp_path / "tasks.json"
    labels_path = tmp_path / "labels.json"
    graphs_path = tmp_path / "graphs.json"
    output_path = tmp_path / "pairs.json"
    config_path.write_text(
        """
        {
          "method": "dense_rgcn_graph_retriever",
          "profile": "full",
          "encoder": {
            "model": "models/file-e5",
            "query_prefix": "file query: ",
            "passage_prefix": "file passage: ",
            "batch_size": 17
          },
          "pair_sampling": {
            "random_seed": 99,
            "easy_random_per_positive": 0,
            "hard_bm25_per_positive": 0,
            "hard_dense_per_positive": 4,
            "hard_graph_neighbor_per_positive": 0,
            "hard_pool_size": 12
          }
        }
        """,
        encoding="utf-8",
    )

    config = CONFIG_LOADER.load(
        Registry.configs.PAIRS,
        [
            "--tasks",
            str(tasks_path),
            "--labels",
            str(labels_path),
            "--graphs",
            str(graphs_path),
            "--output",
            str(output_path),
            "--config",
            str(config_path),
            "--easy_random_per_positive",
            "1",
            "--hard_dense_per_positive",
            "0",
        ],
    )

    assert isinstance(config, PairBuildStageConfig)
    assert config.io.tasks == tasks_path
    assert config.io.labels == labels_path
    assert config.io.graphs == graphs_path
    assert config.io.output == output_path
    assert config.job.sampling.random_seed == 99
    assert config.job.sampling.easy_random_per_positive == 1
    assert config.job.sampling.hard_dense_per_positive == 0
    assert config.job.sampling.hard_pool_size == 12
    assert config.job.hard_dense_encoder == DenseEncoderSettings(
        model_name="models/file-e5",
        query_prefix="file query: ",
        passage_prefix="file passage: ",
        batch_size=17,
    )


def test_pair_build_stage_config_resolves_legacy_defaults_profiles_before_cli(tmp_path: Path) -> None:
    config_path = tmp_path / "base_training_config.json"
    tasks_path = tmp_path / "tasks.json"
    labels_path = tmp_path / "labels.json"
    graphs_path = tmp_path / "graphs.json"
    output_path = tmp_path / "pairs.json"
    config_path.write_text(
        """
        {
          "schema_version": 1,
          "method": "dense_rgcn_graph_retriever",
          "default_profile": "quick",
          "defaults": {
            "encoder": {
              "model": "models/default-e5",
              "query_prefix": "default query: ",
              "passage_prefix": "default passage: "
            },
            "pair_sampling": {
              "random_seed": 13,
              "easy_random_per_positive": 2,
              "hard_bm25_per_positive": 2,
              "hard_dense_per_positive": 0,
              "hard_graph_neighbor_per_positive": 1,
              "hard_pool_size": 30
            }
          },
          "profiles": {
            "quick": {
              "pair_sampling": {
                "easy_random_per_positive": 1,
                "hard_bm25_per_positive": 0
              }
            }
          }
        }
        """,
        encoding="utf-8",
    )

    config = CONFIG_LOADER.load(
        Registry.configs.PAIRS,
        [
            "--tasks",
            str(tasks_path),
            "--labels",
            str(labels_path),
            "--graphs",
            str(graphs_path),
            "--output",
            str(output_path),
            "--config",
            str(config_path),
            "--hard_dense_per_positive",
            "1",
        ],
    )

    assert config.job.sampling == stage_configs.PairSamplingSettings(
        random_seed=13,
        easy_random_per_positive=1,
        hard_bm25_per_positive=0,
        hard_dense_per_positive=1,
        hard_graph_neighbor_per_positive=1,
        hard_pool_size=30,
    )
    assert config.job.hard_dense_encoder == DenseEncoderSettings(
        model_name="models/default-e5",
        query_prefix="default query: ",
        passage_prefix="default passage: ",
    )


def test_pair_build_stage_config_rejects_config_sampling_missing_fields(tmp_path: Path) -> None:
    config_path = tmp_path / "effective_training_config.json"
    config_path.write_text(
        """
        {
          "method": "dense_rgcn_graph_retriever",
          "profile": "quick",
          "pair_sampling": {
            "random_seed": 7,
            "easy_random_per_positive": 1,
            "hard_bm25_per_positive": 0,
            "hard_graph_neighbor_per_positive": 1,
            "hard_pool_size": 10
          }
        }
        """,
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="hard_dense_per_positive"):
        CONFIG_LOADER.load(
            Registry.configs.PAIRS,
            [
                "--tasks",
                str(tmp_path / "tasks.json"),
                "--labels",
                str(tmp_path / "labels.json"),
                "--graphs",
                str(tmp_path / "graphs.json"),
                "--output",
                str(tmp_path / "pairs.json"),
                "--config",
                str(config_path),
            ],
        )


def test_pair_build_stage_config_has_io_and_job_roots() -> None:
    assert hasattr(stage_configs, "PairBuildStageConfig")
    PairBuildStageConfig = stage_configs.PairBuildStageConfig
    assert [field.name for field in fields(PairBuildStageConfig)] == ["io", "job"]


def test_train_stage_config_loads_directly_from_existing_cli_contract(tmp_path: Path) -> None:
    train_tasks_path = tmp_path / "train.input.json"
    train_labels_path = tmp_path / "train.labels.json"
    train_graphs_path = tmp_path / "train.graphs.json"
    train_pairs_path = tmp_path / "train.pairs.json"
    dev_tasks_path = tmp_path / "dev.input.json"
    dev_labels_path = tmp_path / "dev.labels.json"
    dev_graphs_path = tmp_path / "dev.graphs.json"
    output_dir = tmp_path / "rgcn_run"

    config = CONFIG_LOADER.load(
        Registry.configs.TRAIN,
        [
            "--method",
            "dense_rgcn_graph_retriever",
            "--train_tasks",
            str(train_tasks_path),
            "--train_labels",
            str(train_labels_path),
            "--train_graphs",
            str(train_graphs_path),
            "--train_pairs",
            str(train_pairs_path),
            "--dev_tasks",
            str(dev_tasks_path),
            "--dev_labels",
            str(dev_labels_path),
            "--dev_graphs",
            str(dev_graphs_path),
            "--output_dir",
            str(output_dir),
            "--encoder_model",
            "fake-encoder",
            "--hidden_dim",
            "8",
            "--num_layers",
            "1",
            "--dropout",
            "0.0",
            "--epochs",
            "2",
            "--batch_size",
            "3",
            "--learning_rate",
            "0.01",
            "--device",
            "cpu",
        ],
    )

    assert config == RgcnTrainStageConfig(
        method=RetrievalMethodId.DENSE_RGCN_GRAPH_RETRIEVER,
        io=RgcnTrainStageConfig.io_type(
            train_tasks=train_tasks_path,
            train_labels=train_labels_path,
            train_graphs=train_graphs_path,
            train_pairs=train_pairs_path,
            dev_tasks=dev_tasks_path,
            dev_labels=dev_labels_path,
            dev_graphs=dev_graphs_path,
            output_dir=output_dir,
            checkpoint_dir=output_dir / "checkpoints",
            metrics=output_dir / "train_metrics.jsonl",
            run_summary=output_dir / "train_run_summary.json",
        ),
        job=RgcnMethodSettings(
            encoder=DenseEncoderSettings(
                model_name="fake-encoder",
                query_prefix="query: ",
                passage_prefix="passage: ",
            ),
            model=RgcnModelSettings(
                hidden_dim=8,
                num_layers=1,
                dropout=0.0,
                ablation="full_rgcn",
            ),
            trainer=RgcnTrainerSettings(
                optimizer_name="AdamW",
                learning_rate=0.01,
                batch_size=3,
                max_grad_norm=1.0,
                random_seed=13,
                pos_weight_enabled=False,
                epochs=2,
                device="cpu",
            ),
            pairs=RgcnPairSamplingSettings(),
            reporting=TrainingReportingSettings(),
            selection=ModelSelectionSettings(),
        ),
    )


def test_train_stage_config_allows_omitting_train_labels(tmp_path: Path) -> None:
    output_dir = tmp_path / "rgcn_run"

    config = CONFIG_LOADER.load(
        Registry.configs.TRAIN,
        [
            "--method",
            "dense_rgcn_graph_retriever",
            "--train_tasks",
            str(tmp_path / "train.input.json"),
            "--train_graphs",
            str(tmp_path / "train.graphs.json"),
            "--train_pairs",
            str(tmp_path / "train.pairs.json"),
            "--dev_tasks",
            str(tmp_path / "dev.input.json"),
            "--dev_labels",
            str(tmp_path / "dev.labels.json"),
            "--dev_graphs",
            str(tmp_path / "dev.graphs.json"),
            "--output_dir",
            str(output_dir),
            "--encoder_model",
            "fake-encoder",
        ],
    )

    assert isinstance(config, RgcnTrainStageConfig)
    assert config.io.train_labels is None


def test_train_stage_config_loads_config_without_cli_defaults_clobbering_file_values(tmp_path: Path) -> None:
    config_path = tmp_path / "effective_training_config.json"
    output_dir = tmp_path / "rgcn_run"
    config_path.write_text(
        """
        {
          "method": "dense_rgcn_graph_retriever",
          "profile": "quick",
          "encoder": {
            "model": "models/file-e5",
            "query_prefix": "file query: ",
            "passage_prefix": "file passage: ",
            "batch_size": 17
          },
          "model": {
            "hidden_dim": 8,
            "num_layers": 1,
            "dropout": 0.0,
            "ablation": "wo_bridge"
          },
          "optimization": {
            "optimizer": "AdamW",
            "epochs": 5,
            "batch_size": 4,
            "learning_rate": 0.02,
            "max_grad_norm": 0.5,
            "random_seed": 17,
            "pos_weight": true,
            "device": "cuda"
          },
          "pair_sampling": {
            "random_seed": 17,
            "easy_random_per_positive": 1,
            "hard_bm25_per_positive": 0,
            "hard_dense_per_positive": 0,
            "hard_graph_neighbor_per_positive": 1,
            "hard_pool_size": 10
          }
        }
        """,
        encoding="utf-8",
    )

    config = CONFIG_LOADER.load(
        Registry.configs.TRAIN,
        [
            "--method",
            "dense_rgcn_graph_retriever",
            "--train_tasks",
            str(tmp_path / "train.input.json"),
            "--train_labels",
            str(tmp_path / "train.labels.json"),
            "--train_graphs",
            str(tmp_path / "train.graphs.json"),
            "--train_pairs",
            str(tmp_path / "train.pairs.json"),
            "--dev_tasks",
            str(tmp_path / "dev.input.json"),
            "--dev_labels",
            str(tmp_path / "dev.labels.json"),
            "--dev_graphs",
            str(tmp_path / "dev.graphs.json"),
            "--output_dir",
            str(output_dir),
            "--config",
            str(config_path),
            "--epochs",
            "1",
            "--device",
            "cpu",
        ],
    )

    assert isinstance(config, RgcnTrainStageConfig)
    assert config.job.encoder == DenseEncoderSettings(
        model_name="models/file-e5",
        query_prefix="file query: ",
        passage_prefix="file passage: ",
        batch_size=17,
    )
    assert config.job.model == RgcnModelSettings(
        hidden_dim=8,
        num_layers=1,
        dropout=0.0,
        ablation="wo_bridge",
    )
    assert config.job.trainer == RgcnTrainerSettings(
        optimizer_name="AdamW",
        learning_rate=0.02,
        batch_size=4,
        max_grad_norm=0.5,
        random_seed=17,
        pos_weight_enabled=True,
        epochs=1,
        device="cpu",
    )
    assert config.job.pairs == RgcnPairSamplingSettings(
        random_seed=17,
        easy_random_per_positive=1,
        hard_bm25_per_positive=0,
        hard_dense_per_positive=0,
        hard_graph_neighbor_per_positive=1,
        hard_pool_size=10,
    )


def test_train_stage_config_resolves_legacy_defaults_profiles_before_cli(tmp_path: Path) -> None:
    config_path = tmp_path / "base_training_config.json"
    config_path.write_text(
        """
        {
          "schema_version": 1,
          "method": "dense_rgcn_graph_retriever",
          "default_profile": "quick",
          "defaults": {
            "encoder": {
              "model": "models/default-e5",
              "query_prefix": "default query: ",
              "passage_prefix": "default passage: "
            },
            "model": {
              "hidden_dim": 128,
              "num_layers": 2,
              "dropout": 0.1,
              "ablation": "full_rgcn"
            },
            "optimization": {
              "optimizer": "AdamW",
              "epochs": 5,
              "batch_size": 16,
              "learning_rate": 0.0001,
              "max_grad_norm": 1.0,
              "random_seed": 13,
              "pos_weight": true,
              "device": "cpu"
            },
            "pair_sampling": {
              "random_seed": 13,
              "easy_random_per_positive": 2,
              "hard_bm25_per_positive": 2,
              "hard_dense_per_positive": 0,
              "hard_graph_neighbor_per_positive": 1,
              "hard_pool_size": 30
            }
          },
          "profiles": {
            "quick": {
              "model": {
                "hidden_dim": 32,
                "num_layers": 1
              },
              "optimization": {
                "epochs": 1,
                "batch_size": 1
              }
            }
          }
        }
        """,
        encoding="utf-8",
    )

    config = CONFIG_LOADER.load(
        Registry.configs.TRAIN,
        [
            "--method",
            "dense_rgcn_graph_retriever",
            "--train_tasks",
            str(tmp_path / "train.input.json"),
            "--train_labels",
            str(tmp_path / "train.labels.json"),
            "--train_graphs",
            str(tmp_path / "train.graphs.json"),
            "--train_pairs",
            str(tmp_path / "train.pairs.json"),
            "--dev_tasks",
            str(tmp_path / "dev.input.json"),
            "--dev_labels",
            str(tmp_path / "dev.labels.json"),
            "--dev_graphs",
            str(tmp_path / "dev.graphs.json"),
            "--output_dir",
            str(tmp_path / "rgcn_run"),
            "--config",
            str(config_path),
            "--dropout",
            "0.0",
        ],
    )

    assert isinstance(config, RgcnTrainStageConfig)
    assert config.job.model == RgcnModelSettings(
        hidden_dim=32,
        num_layers=1,
        dropout=0.0,
        ablation="full_rgcn",
    )
    assert config.job.trainer.epochs == 1
    assert config.job.trainer.batch_size == 1


def test_train_stage_config_loads_dense_ft_without_graph_io(tmp_path: Path) -> None:
    output_dir = tmp_path / "dense_ft_run"

    config = CONFIG_LOADER.load(
        Registry.configs.TRAIN,
        [
            "--method",
            "dense_ft",
            "--train_tasks",
            str(tmp_path / "train.input.json"),
            "--train_labels",
            str(tmp_path / "train.labels.json"),
            "--train_pairs",
            str(tmp_path / "train.pairs.json"),
            "--dev_tasks",
            str(tmp_path / "dev.input.json"),
            "--dev_labels",
            str(tmp_path / "dev.labels.json"),
            "--output_dir",
            str(output_dir),
            "--encoder_model",
            "fake-e5",
            "--device",
            "cpu",
        ],
    )

    assert config == DenseFinetuneTrainStageConfig(
        method=RetrievalMethodId.DENSE_FT,
        io=DenseFinetuneTrainStageConfig.io_type(
            train_tasks=tmp_path / "train.input.json",
            train_labels=tmp_path / "train.labels.json",
            train_pairs=tmp_path / "train.pairs.json",
            dev_tasks=tmp_path / "dev.input.json",
            dev_labels=tmp_path / "dev.labels.json",
            output_dir=output_dir,
            model_dir=output_dir / "checkpoints" / "best_model",
            metrics=output_dir / "train_metrics.jsonl",
            run_summary=output_dir / "train_run_summary.json",
        ),
        job=DenseFinetuneMethodSettings(
            encoder=DenseEncoderSettings(model_name="fake-e5", query_prefix="query: ", passage_prefix="passage: "),
            trainer=stage_configs.DenseFinetuneTrainerSettings(device="cpu"),
        ),
    )
    assert not hasattr(config.io, "train_graphs")
    assert not hasattr(config.io, "dev_graphs")


def test_train_stage_config_loads_dense_ft_legacy_defaults_profile(tmp_path: Path) -> None:
    config_path = tmp_path / "dense_ft_base.json"
    config_path.write_text(
        """
        {
          "schema_version": 1,
          "method": "dense_ft",
          "default_profile": "smoke",
          "defaults": {
            "encoder": {
              "model": "models/default-e5",
              "query_prefix": "query: ",
              "passage_prefix": "passage: ",
              "batch_size": 64
            },
            "data": {
              "hard_negatives_per_positive": 1
            },
            "trainer": {
              "learning_rate": 0.00002,
              "train_batch_size": 16,
              "eval_batch_size": 64,
              "epochs": 2,
              "warmup_ratio": 0.1,
              "max_grad_norm": 1.0,
              "random_seed": 13,
              "device": "cuda",
              "fp16": false,
              "bf16": false,
              "logging_steps": 50,
              "save_total_limit": 2
            },
            "selection": {
              "best_metric": "eval_dev_cosine_ndcg@10",
              "higher_is_better": true
            }
          },
          "profiles": {
            "smoke": {
              "trainer": {
                "train_batch_size": 1,
                "eval_batch_size": 4,
                "epochs": 1,
                "device": "cpu",
                "logging_steps": 1
              }
            }
          }
        }
        """,
        encoding="utf-8",
    )

    config = CONFIG_LOADER.load(
        Registry.configs.TRAIN,
        [
            "--method",
            "dense_ft",
            "--train_tasks",
            str(tmp_path / "train.input.json"),
            "--train_labels",
            str(tmp_path / "train.labels.json"),
            "--train_pairs",
            str(tmp_path / "train.pairs.json"),
            "--dev_tasks",
            str(tmp_path / "dev.input.json"),
            "--dev_labels",
            str(tmp_path / "dev.labels.json"),
            "--output_dir",
            str(tmp_path / "dense_ft_run"),
            "--config",
            str(config_path),
        ],
    )

    assert isinstance(config, DenseFinetuneTrainStageConfig)
    assert config.job.encoder == DenseEncoderSettings(
        model_name="models/default-e5",
        query_prefix="query: ",
        passage_prefix="passage: ",
        batch_size=64,
    )
    assert config.job.trainer.device == "cpu"
    assert config.job.trainer.train_batch_size == 1
    assert config.job.trainer.eval_batch_size == 4
    assert config.job.trainer.epochs == 1


def test_train_stage_config_variants_have_method_io_and_job_roots() -> None:
    assert [field.name for field in fields(RgcnTrainStageConfig)] == ["method", "io", "job"]
    assert [field.name for field in fields(DenseFinetuneTrainStageConfig)] == ["method", "io", "job"]


def test_training_registry_builds_method_trainers_without_global_deps(tmp_path: Path) -> None:
    rgcn_config = CONFIG_LOADER.load(
        Registry.configs.TRAIN,
        [
            "--method",
            "dense_rgcn_graph_retriever",
            "--train_tasks",
            str(tmp_path / "train.input.json"),
            "--train_graphs",
            str(tmp_path / "train.graphs.json"),
            "--train_pairs",
            str(tmp_path / "train.pairs.json"),
            "--dev_tasks",
            str(tmp_path / "dev.input.json"),
            "--dev_labels",
            str(tmp_path / "dev.labels.json"),
            "--dev_graphs",
            str(tmp_path / "dev.graphs.json"),
            "--output_dir",
            str(tmp_path / "rgcn_run"),
            "--encoder_model",
            "fake-e5",
        ],
    )
    dense_ft_config = CONFIG_LOADER.load(
        Registry.configs.TRAIN,
        [
            "--method",
            "dense_ft",
            "--train_tasks",
            str(tmp_path / "train.input.json"),
            "--train_labels",
            str(tmp_path / "train.labels.json"),
            "--train_pairs",
            str(tmp_path / "train.pairs.json"),
            "--dev_tasks",
            str(tmp_path / "dev.input.json"),
            "--dev_labels",
            str(tmp_path / "dev.labels.json"),
            "--output_dir",
            str(tmp_path / "dense_ft_run"),
            "--encoder_model",
            "fake-e5",
        ],
    )

    assert callable(Registry.training.build(rgcn_config.job).train)
    assert callable(Registry.training.build(dense_ft_config.job).train)


def test_evaluate_stage_config_loads_directly_from_existing_cli_contract(tmp_path: Path) -> None:
    predictions_path = tmp_path / "predictions.json"
    labels_path = tmp_path / "labels.json"
    graphs_path = tmp_path / "graphs.json"
    output_path = tmp_path / "metrics.csv"
    failures_path = tmp_path / "failure_cases.jsonl"

    config = CONFIG_LOADER.load(
        Registry.configs.EVALUATE,
        [
            "--pred",
            str(predictions_path),
            "--labels",
            str(labels_path),
            "--graphs",
            str(graphs_path),
            "--output",
            str(output_path),
            "--failure_cases_output",
            str(failures_path),
            "--failure_case_limit",
            "5",
        ],
    )

    assert config == EvaluateStageConfig(
        io=EvaluateStageConfig.io_type(
            predictions=predictions_path,
            labels=labels_path,
            graphs=graphs_path,
            output=output_path,
            failure_cases_output=failures_path,
        ),
        failure_case_limit=5,
    )


def test_evaluate_stage_config_maps_gold_alias_to_labels(tmp_path: Path) -> None:
    gold_path = tmp_path / "gold.labels.json"

    config = CONFIG_LOADER.load(
        Registry.configs.EVALUATE,
        [
            "--pred",
            str(tmp_path / "predictions.json"),
            "--gold",
            str(gold_path),
            "--graphs",
            str(tmp_path / "graphs.json"),
            "--output",
            str(tmp_path / "metrics.csv"),
        ],
    )

    assert config.io.labels == gold_path
    assert config.io.failure_cases_output is None
    assert config.failure_case_limit == 0


def test_evaluate_stage_config_requires_label_or_gold(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="--labels is required"):
        CONFIG_LOADER.load(
            Registry.configs.EVALUATE,
            [
                "--pred",
                str(tmp_path / "predictions.json"),
                "--graphs",
                str(tmp_path / "graphs.json"),
                "--output",
                str(tmp_path / "metrics.csv"),
            ],
        )


def test_evaluate_stage_config_has_io_and_failure_limit_roots() -> None:
    assert [field.name for field in fields(EvaluateStageConfig)] == ["io", "failure_case_limit"]

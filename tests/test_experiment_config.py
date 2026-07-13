from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.errors import ConfigCompositionException
from omegaconf import OmegaConf
from omegaconf.errors import MissingMandatoryValue
from pydantic import ValidationError

from graph_memory.experiment.config import (
    ExperimentConfig,
    ModelSelectionConfig,
    resolve_experiment_config,
    validate_composed_config,
)
from graph_memory.models.graph_retriever.selection import RgcnSelectionMetric
from graph_memory.registry.ablations import AblationVariantId


REPO_ROOT = Path(__file__).parents[1]
CONFIG_DIR = REPO_ROOT / "configs"


def _compose(
    *,
    config_name: str = "config",
    overrides: list[str] | None = None,
) -> ExperimentConfig:
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        config = compose(
            config_name=config_name,
            overrides=["name=test-run", *(overrides or [])],
        )
    return validate_composed_config(config)


def test_default_composition_is_closed_complete_and_resolvable() -> None:
    config = _compose()
    resolved = resolve_experiment_config(config, repository_root=REPO_ROOT)

    assert config.dataset.name == "musique"
    assert config.profile.name == "full"
    assert config.profile.trainable.rgcn.num_layers == 3
    assert config.profile.trainable.rgcn.hard_dense_per_positive == 1
    assert [method.value for method in config.methods] == [
        "bm25",
        "dense",
        "bm25_graph_rerank",
        "dense_graph_rerank",
        "dense_rgcn_graph_retriever",
        "dense_ft",
        "dense_ft_rgcn_graph_retriever",
    ]
    assert set(type(config.method_configs).model_fields) == {
        "bm25",
        "dense",
        "memory_stream",
        "bm25_graph_rerank",
        "dense_graph_rerank",
        "dense_rgcn_graph_retriever",
        "dense_ft",
        "dense_ft_rgcn_graph_retriever",
    }
    assert resolved.dataset.prepare_script.is_absolute()
    assert resolved.dataset.splits["test"].source.is_absolute()
    assert resolved.tracking.database.is_absolute()
    assert resolved.tracking.tracking_uri.startswith("sqlite:///")
    assert "???" not in json.dumps(resolved.normalized())


def test_missing_name_unknown_override_and_unknown_model_key_fail_before_resolution() -> (
    None
):
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        missing = compose(config_name="config")
        with pytest.raises(MissingMandatoryValue):
            OmegaConf.to_container(missing, resolve=True, throw_on_missing=True)
        with pytest.raises(ConfigCompositionException):
            compose(config_name="config", overrides=["name=test", "unknown=1"])

    primitive = _compose().model_dump(mode="python", by_alias=True)
    primitive["unknown"] = 1
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ExperimentConfig.model_validate(primitive)


@pytest.mark.parametrize("value", [True, "14", 14.0])
def test_scientific_integer_rejects_bool_string_and_float_coercion(
    value: object,
) -> None:
    primitive = _compose().model_dump(mode="python", by_alias=True)
    primitive["seed"] = value
    with pytest.raises(ValidationError, match="integer"):
        ExperimentConfig.model_validate(primitive)


def test_dataset_group_and_full_policy_resolve_complete_windows() -> None:
    hotpot = resolve_experiment_config(
        _compose(overrides=["dataset=hotpotqa", "profile=full"]),
        repository_root=REPO_ROOT,
    )
    twowiki = resolve_experiment_config(
        _compose(overrides=["dataset=2wiki", "profile=full"]),
        repository_root=REPO_ROOT,
    )
    musique = resolve_experiment_config(
        _compose(overrides=["dataset=musique", "profile=full"]),
        repository_root=REPO_ROOT,
    )

    assert {name: split.count for name, split in hotpot.dataset.splits.items()} == {
        "train": 90025,
        "dev": 500,
        "test": 6869,
    }
    assert {name: split.count for name, split in twowiki.dataset.splits.items()} == {
        "train": 167454,
        "dev": 500,
        "test": 12076,
    }
    assert {name: split.count for name, split in musique.dataset.splits.items()} == {
        "train": 19938,
        "dev": 500,
        "test": 1917,
    }
    assert twowiki.dataset.name == "twowiki"
    assert twowiki.dataset.prepare_script.name == "prepare_2wiki.py"


def test_fixed_count_beyond_dataset_capacity_fails_during_resolution() -> None:
    config = _compose(overrides=["dataset=hotpotqa", "profile=full"])
    primitive = config.model_dump(mode="python", by_alias=True)
    primitive["profile"]["splits"]["test"] = {"kind": "fixed", "count": 7000}
    invalid = ExperimentConfig.model_validate(primitive)
    with pytest.raises(ValueError, match=r"offset\+count=7500 beyond capacity=7369"):
        resolve_experiment_config(invalid, repository_root=REPO_ROOT)


def test_method_subset_seed_device_and_method_override_have_one_path() -> None:
    baseline = _compose()
    config = _compose(
        overrides=[
            "methods=[bm25,dense]",
            "seed=14",
            "device=cpu",
            "method_configs.dense_ft.train.trainer.learning_rate=3e-5",
        ]
    )

    assert [method.value for method in config.methods] == ["bm25", "dense"]
    assert config.method_configs.bm25.model_dump() == {"method": "bm25"}
    assert "device" not in config.method_configs.bm25.model_dump()
    assert config.method_configs.dense_ft.pairs.random_seed == 14
    assert config.method_configs.dense_ft.train.trainer.random_seed == 14
    assert config.method_configs.dense_ft.train.trainer.device == "cpu"
    assert config.method_configs.dense_ft.train.trainer.learning_rate == pytest.approx(
        3e-5
    )
    assert (
        config.method_configs.dense_rgcn_graph_retriever.train.trainer.learning_rate
        == baseline.method_configs.dense_rgcn_graph_retriever.train.trainer.learning_rate
    )


def test_ablation_values_and_explicit_2wiki_overrides_are_closed() -> None:
    config = _compose(
        overrides=[
            "dataset=2wiki",
            "profile=smoke",
            "methods=[bm25,dense,dense_graph_rerank,dense_rgcn_graph_retriever,dense_ft,dense_ft_rgcn_graph_retriever]",
        ]
    )
    assert config.dataset.name == "twowiki"
    assert config.profile.name == "smoke"
    assert [method.value for method in config.methods] == [
        "bm25",
        "dense",
        "dense_graph_rerank",
        "dense_rgcn_graph_retriever",
        "dense_ft",
        "dense_ft_rgcn_graph_retriever",
    ]

    default_ablation = _compose().ablation
    assert default_ablation.enable is False
    expected_variants = [variant.value for variant in AblationVariantId]
    assert [variant.value for variant in default_ablation.variants] == expected_variants

    full_ablation = _compose(overrides=["ablation.enable=true"]).ablation
    assert full_ablation.enable is True
    assert [variant.value for variant in full_ablation.variants] == expected_variants

    selected_ablation = _compose(
        overrides=["ablation.enable=true", "ablation.variants=[wo_graph]"]
    ).ablation
    assert selected_ablation.enable is True
    assert [variant.value for variant in selected_ablation.variants] == ["wo_graph"]
    assert set(expected_variants) == {
        "wo_bridge",
        "wo_entity_overlap",
        "wo_sequential",
        "wo_query_overlap",
        "wo_graph",
        "wo_edge_type",
        "wo_edge_weight",
        "wo_seed_score",
        "wo_hard_negatives",
    }

    with pytest.raises(ValidationError, match="at least 1 item"):
        primitive = _compose().model_dump(mode="python", by_alias=True)
        primitive["ablation"]["variants"] = []
        ExperimentConfig.model_validate(primitive)
    with pytest.raises(ValidationError, match="valid ablation variant"):
        primitive = _compose().model_dump(mode="python", by_alias=True)
        primitive["ablation"]["variants"] = ["full_rgcn"]
        ExperimentConfig.model_validate(primitive)


@pytest.mark.parametrize(
    "method_name",
    ["dense_rgcn_graph_retriever", "dense_ft_rgcn_graph_retriever"],
)
def test_rgcn_methods_require_typed_beam_training_config(method_name: str) -> None:
    config = _compose()
    train = getattr(config.method_configs, method_name).train

    assert train.decoder.hidden_dim > 0
    assert train.beam.training_beam_size == 2
    assert train.beam.inference_beam_size == 2
    assert train.beam.max_steps == 5
    assert train.beam.deduplicate_selected_sets is True
    assert train.loss.next_action_loss_weight == pytest.approx(1.0)
    assert train.loss.stop_loss_weight == pytest.approx(1.0)
    assert train.loss.aux_node_loss_weight == pytest.approx(0.2)
    assert train.optimizer_phases.decoder_learning_rate > 0
    assert train.optimizer_phases.rgcn_learning_rate > 0
    assert train.selection.best_metric == "dev_composite"
    assert train.selection.higher_is_better is True

    primitive = config.model_dump(mode="python", by_alias=True)
    for field_name in ("decoder", "beam", "loss", "optimizer_phases", "selection"):
        incomplete = deepcopy(primitive)
        del incomplete["method_configs"][method_name]["train"][field_name]
        with pytest.raises(ValidationError, match=field_name):
            ExperimentConfig.model_validate(incomplete)


@pytest.mark.parametrize(
    "metric_name",
    [
        "dev_composite",
        "dev_full_support_at_5",
        "dev_full_support_at_10",
        "dev_recall_at_5",
        "dev_mrr",
        "dev_loss",
    ],
)
def test_rgcn_selection_accepts_only_supported_metrics(
    metric_name: RgcnSelectionMetric,
) -> None:
    selection = ModelSelectionConfig(
        best_metric=metric_name,
        higher_is_better=metric_name != "dev_loss",
    )

    assert selection.best_metric == metric_name

    with pytest.raises(ValidationError):
        ModelSelectionConfig.model_validate(
            {
                "best_metric": "full_support_at_5",
                "higher_is_better": True,
            }
        )


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("training_beam_size", 0),
        ("inference_beam_size", 0),
        ("max_steps", 0),
        ("max_steps", 6),
    ],
)
def test_rgcn_beam_config_rejects_invalid_bounds(field_name: str, value: int) -> None:
    primitive = _compose().model_dump(mode="python", by_alias=True)
    primitive["method_configs"]["dense_rgcn_graph_retriever"]["train"]["beam"][
        field_name
    ] = value

    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(primitive)


def test_rgcn_beam_config_rejects_unknown_fields_and_mismatched_beam_sizes() -> None:
    primitive = _compose().model_dump(mode="python", by_alias=True)
    beam = primitive["method_configs"]["dense_rgcn_graph_retriever"]["train"]["beam"]
    beam["legacy_width"] = 2
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ExperimentConfig.model_validate(primitive)

    primitive = _compose().model_dump(mode="python", by_alias=True)
    beam = primitive["method_configs"]["dense_rgcn_graph_retriever"]["train"]["beam"]
    beam["inference_beam_size"] = 4
    with pytest.raises(ValidationError, match="must match"):
        ExperimentConfig.model_validate(primitive)

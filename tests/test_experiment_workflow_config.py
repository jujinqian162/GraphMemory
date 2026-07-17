from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.errors import ConfigCompositionException
from pydantic import ValidationError

from graph_memory.experiment.config import (
    DenseFinetuneMethodConfig,
    DenseFtRgcnMethodConfig,
    RgcnMethodConfig,
    resolve_experiment_config,
    parse_composed_config,
)


ROOT = Path(__file__).resolve().parents[1]


def _compose(*overrides: str):
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        return compose(
            config_name="config",
            overrides=["name=config-test", "profile=smoke", "device=cpu", *overrides],
        )


@pytest.mark.parametrize(
    "method",
    (
        "bm25",
        "dense",
        "graphrag",
        "dense_ft",
        "dense_rgcn_graph_retriever",
        "dense_ft_rgcn_graph_retriever",
    ),
)
def test_evidence_method_configs_compose_as_one_final_method(method: str) -> None:
    config = parse_composed_config(_compose(f"method={method}"))

    assert config.method.method == method
    assert not hasattr(config, "methods")
    assert not hasattr(config, "method_configs")


@pytest.mark.parametrize(
    "method",
    ("execution_provenance_retriever", "execution_provenance_rgcn_retriever"),
)
def test_provenance_method_configs_compose_for_provenance_dataset(method: str) -> None:
    config = parse_composed_config(
        _compose("dataset=twowiki_provenance", f"method={method}")
    )
    resolved = resolve_experiment_config(config, repository_root=ROOT)

    assert resolved.method.method == method
    assert resolved.dataset.name == "twowiki_provenance"


def test_dense_ft_seed_config_is_the_canonical_public_dense_ft_stage() -> None:
    dense_ft = parse_composed_config(_compose("method=dense_ft"))
    composite = parse_composed_config(
        _compose("method=dense_ft_rgcn_graph_retriever")
    )

    assert isinstance(dense_ft.method, DenseFinetuneMethodConfig)
    assert isinstance(composite.method, DenseFtRgcnMethodConfig)
    assert composite.method.seed == dense_ft.method


def test_rgcn_has_one_singular_variant_and_applies_it_at_the_first_change() -> None:
    config = parse_composed_config(
        _compose(
            "method=dense_rgcn_graph_retriever", "method.variant=wo_hard_negatives"
        )
    )

    assert isinstance(config.method, RgcnMethodConfig)
    effective = config.method.effective()
    assert effective.variant == "wo_hard_negatives"
    assert effective.pairs.hard_bm25_per_positive == 0
    assert effective.pairs.hard_dense_per_positive == 0
    assert effective.pairs.hard_graph_neighbor_per_positive == 0
    assert effective.train.model.ablation == "full_rgcn"


def test_rgcn_defaults_to_full_variant() -> None:
    config = parse_composed_config(_compose("method=dense_rgcn_graph_retriever"))

    assert isinstance(config.method, RgcnMethodConfig)
    assert config.method.variant == "full_rgcn"


@pytest.mark.parametrize(
    "override",
    (
        "+methods=[bm25,dense]",
        "+ablation.variants=[full_rgcn,wo_graph]",
        "method=dense_rgcn_graph_retriever",
    ),
)
def test_retired_or_list_valued_selection_is_rejected(override: str) -> None:
    overrides = [override]
    if override == "method=dense_rgcn_graph_retriever":
        overrides.append("method.variant=[full_rgcn,wo_graph]")
    with pytest.raises((ValidationError, ConfigCompositionException)):
        parse_composed_config(_compose(*overrides))


def test_variant_is_rejected_on_non_rgcn_method() -> None:
    with pytest.raises((ValidationError, ConfigCompositionException)):
        parse_composed_config(_compose("method=bm25", "+method.variant=wo_graph"))


def test_provenance_only_method_is_rejected_on_evidence_dataset() -> None:
    config = parse_composed_config(_compose("method=execution_provenance_retriever"))

    with pytest.raises(ValueError, match="does not support"):
        resolve_experiment_config(config, repository_root=ROOT)


def test_model_only_variant_reuses_pair_contract_but_hard_negative_variant_does_not() -> (
    None
):
    full = parse_composed_config(_compose("method=dense_rgcn_graph_retriever"))
    wo_graph = parse_composed_config(
        _compose("method=dense_rgcn_graph_retriever", "method.variant=wo_graph")
    )
    wo_hard_negatives = parse_composed_config(
        _compose(
            "method=dense_rgcn_graph_retriever",
            "method.variant=wo_hard_negatives",
        )
    )

    assert isinstance(full.method, RgcnMethodConfig)
    assert isinstance(wo_graph.method, RgcnMethodConfig)
    assert isinstance(wo_hard_negatives.method, RgcnMethodConfig)
    assert full.method.effective().pairs == wo_graph.method.effective().pairs
    assert full.method.effective().pairs != wo_hard_negatives.method.effective().pairs


def test_hydra_multirun_subdir_uses_only_native_interpolations() -> None:
    config_text = (ROOT / "configs" / "config.yaml").read_text(encoding="utf-8")

    assert "subdir: ${hydra.job.num}_${method.method}" in config_text
    assert "concise_override" not in config_text

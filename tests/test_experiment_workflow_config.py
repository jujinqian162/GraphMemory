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
    ExecutionProvenanceRgcnMethodConfig,
    resolve_experiment_config,
    parse_composed_config,
)
from graph_memory.experiment.inspect import inspect_catalog
from graph_memory.registry.retrieval import RetrievalMethodId


ROOT = Path(__file__).resolve().parents[1]


def _compose(*overrides: str):
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        return compose(
            config_name="config",
            overrides=["name=config-test", "profile=smoke", "device=cpu", *overrides],
        )


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

    defaulted = parse_composed_config(_compose("method=dense_rgcn_graph_retriever"))
    assert isinstance(defaulted.method, RgcnMethodConfig)
    assert defaulted.method.variant == "full_rgcn"


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


def test_provenance_variant_lifecycle_boundaries_are_explicit() -> None:
    base = ("dataset=twowiki_provenance", "method=execution_provenance_rgcn_retriever")
    full = parse_composed_config(_compose(*base))
    wo_graph = parse_composed_config(_compose(*base, "method.variant=wo_graph"))
    wo_hard = parse_composed_config(
        _compose(*base, "method.variant=wo_hard_negatives")
    )
    wo_rerank = parse_composed_config(
        _compose(*base, "method.variant=wo_edge_rerank")
    )

    assert isinstance(full.method, ExecutionProvenanceRgcnMethodConfig)
    assert isinstance(wo_graph.method, ExecutionProvenanceRgcnMethodConfig)
    assert isinstance(wo_hard.method, ExecutionProvenanceRgcnMethodConfig)
    assert isinstance(wo_rerank.method, ExecutionProvenanceRgcnMethodConfig)
    assert full.method.effective().pairs == wo_graph.method.effective().pairs
    assert full.method.effective().pairs == wo_rerank.method.effective().pairs
    assert full.method.effective().pairs != wo_hard.method.effective().pairs
    assert wo_hard.method.effective().pairs.hard_provenance_successor_per_positive == 0
    assert wo_hard.method.effective().pairs.hard_provenance_predecessor_per_positive == 0
    assert full.method.train_stage() == wo_rerank.method.train_stage()
    assert full.method.train_stage() != wo_graph.method.train_stage()

    variants = inspect_catalog("variants", repository_root=ROOT)
    assert isinstance(variants, dict)
    assert variants[
        RetrievalMethodId.EXECUTION_PROVENANCE_RGCN_RETRIEVER
    ] == [
        "full_rgcn",
        "wo_graph",
        "wo_edge_type",
        "wo_edge_weight",
        "wo_hard_negatives",
        "wo_edge_rerank",
    ]

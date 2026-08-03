from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.errors import ConfigCompositionException
from pydantic import ValidationError

from graph_memory.datasets.isetrace.registration import ISETRACE_REVISION
from graph_memory.experiment.config import (
    DenseFinetuneMethodConfig,
    DenseFtRgcnMethodConfig,
    ISETraceDatasetConfig,
    ProvenancePathMethodConfig,
    ProvenanceRgcnMethodConfig,
    RgcnMethodConfig,
    parse_composed_config,
    resolve_experiment_config,
)
from graph_memory.experiment.workflow import _prepare_config


ROOT = Path(__file__).resolve().parents[1]


def _compose(*overrides: str):
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        return compose(
            config_name="config",
            overrides=["name=config-test", "profile=smoke", "device=cpu", *overrides],
        )


def test_isetrace_config_resolves_explicit_query_counts() -> None:
    composed = parse_composed_config(
        _compose("dataset=isetrace", "profile=full", "method=provenance_path")
    )
    resolved = resolve_experiment_config(composed, repository_root=ROOT)

    assert isinstance(composed.dataset, ISETraceDatasetConfig)
    assert isinstance(resolved.method, ProvenancePathMethodConfig)
    assert composed.dataset.queries.splits.model_dump() == {
        "train": {"natural": 2692, "template": 5384},
        "dev": {"natural": 393, "template": 393},
        "test": {"natural": 981, "template": 0},
    }
    assert set(resolved.dataset.splits) == {"train", "dev", "test"}
    natural_source = (
        ROOT / "data/isetrace/query-authoring/isetrace-v7-raw.jsonl"
    ).resolve()
    assert all(
        split.source == natural_source for split in resolved.dataset.splits.values()
    )
    assert resolved.dataset.natural_query_source == natural_source
    assert (
        resolved.dataset.trajectory_source
        == (ROOT / "data/isetrace/raw/trajectories").resolve()
    )
    assert resolved.dataset.strict_invalid_examples is False
    assert resolved.dataset.source_revision == ISETRACE_REVISION
    assert not hasattr(composed.dataset, "source_revision")
    assert not hasattr(composed.dataset, "splits")


@pytest.mark.parametrize(
    "retired_override",
    (
        "+dataset.source_revision=retired",
        "+dataset.strict_invalid_examples=false",
        "+dataset.splits.test.kind=raw",
        "+dataset.allow_unreviewed=true",
        "+dataset.accepted_only=true",
        "+dataset.target_policy=answer_only",
        "+dataset.queries.kind=corpus",
        "+dataset.queries.invalid_policy=drop",
        "+dataset.queries.grouping_policy=trajectory",
        "+dataset.queries.manifest_name=split.json",
        "+dataset.queries.extension_policy=rebuild",
        "+dataset.queries.schema_version=7",
        "+dataset.queries.split_ratio.train=1",
        "+dataset.queries.mix_ratio.train.natural=1",
    ),
)
def test_isetrace_rejects_retired_test_only_and_policy_fields(
    retired_override: str,
) -> None:
    with pytest.raises((ValidationError, ConfigCompositionException)):
        parse_composed_config(
            _compose(
                "dataset=isetrace",
                "method=provenance_path",
                retired_override,
            )
        )


def test_isetrace_accepts_template_only_train_and_dev_counts() -> None:
    composed = parse_composed_config(
        _compose(
            "dataset=isetrace",
            "method=provenance_rgcn",
            "dataset.queries.splits.train.natural=0",
            "dataset.queries.splits.train.template=8076",
            "dataset.queries.splits.dev.natural=0",
            "dataset.queries.splits.dev.template=786",
        )
    )
    assert isinstance(composed.dataset, ISETraceDatasetConfig)
    assert composed.dataset.queries.splits.train.model_dump() == {
        "natural": 0,
        "template": 8076,
    }
    assert composed.dataset.queries.splits.dev.model_dump() == {
        "natural": 0,
        "template": 786,
    }


def test_isetrace_provenance_rgcn_config_requires_current_trainable_lifecycle() -> None:
    composed = parse_composed_config(
        _compose("dataset=isetrace", "method=provenance_rgcn")
    )
    resolved = resolve_experiment_config(composed, repository_root=ROOT)

    assert isinstance(composed.method, ProvenanceRgcnMethodConfig)
    assert composed.method.variant == "full_rgcn"
    assert set(resolved.dataset.splits) == {"train", "dev", "test"}

    no_graph = parse_composed_config(
        _compose(
            "dataset=isetrace",
            "method=provenance_rgcn",
            "method.variant=wo_graph",
        )
    )
    assert isinstance(no_graph.method, ProvenanceRgcnMethodConfig)
    assert no_graph.method.effective().train.model.num_layers == 0


def test_evidence_dataset_rejects_provenance_rgcn_method() -> None:
    composed = parse_composed_config(
        _compose("dataset=hotpotqa", "method=provenance_rgcn")
    )

    with pytest.raises(ValueError, match="does not support"):
        resolve_experiment_config(composed, repository_root=ROOT)


def test_isetrace_rejects_evidence_only_trainable_method() -> None:
    composed = parse_composed_config(
        _compose("dataset=isetrace", "method=dense_rgcn_graph_retriever")
    )

    with pytest.raises(ValueError, match="does not support"):
        resolve_experiment_config(composed, repository_root=ROOT)


def test_frozen_encoding_config_enables_pytorch_gpu_pool() -> None:
    config = parse_composed_config(
        _compose(
            "method=dense_rgcn_graph_retriever",
            "encoding.enable_gpupool=true",
            "encoding.chunk_size=2048",
        )
    )

    assert config.encoding.enable_gpupool is True
    assert config.encoding.chunk_size == 2048


def test_rgcn_profiles_define_true_graph_batches() -> None:
    evidence = parse_composed_config(
        _compose("profile=full", "method=dense_rgcn_graph_retriever")
    )
    provenance = parse_composed_config(
        _compose("dataset=isetrace", "profile=full", "method=provenance_rgcn")
    )
    resolved_provenance = resolve_experiment_config(
        provenance, repository_root=ROOT
    )

    assert isinstance(evidence.method, RgcnMethodConfig)
    assert evidence.method.train.trainer.per_device_graph_batch_size == 128
    assert isinstance(provenance.method, ProvenanceRgcnMethodConfig)
    assert provenance.method.train.trainer.per_device_graph_batch_size == 128
    assert _prepare_config(resolved_provenance, "train").count is None
    assert _prepare_config(resolved_provenance, "dev").count is None
    assert _prepare_config(resolved_provenance, "test").count is None


def test_dense_ft_seed_config_is_the_canonical_public_dense_ft_stage() -> None:
    dense_ft = parse_composed_config(_compose("method=dense_ft"))
    composite = parse_composed_config(_compose("method=dense_ft_rgcn_graph_retriever"))

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

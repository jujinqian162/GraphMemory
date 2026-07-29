from __future__ import annotations

import pytest
from pydantic import ValidationError

from graph_memory.experiment.config import (
    DatasetConfig,
    TwoWikiProvenanceTransformConfig,
)
from graph_memory.stages.transform import transform_version_tag


def _splits(source: str = "s.json") -> dict[str, object]:
    return {
        "train": {"kind": "raw", "source": source, "offset": 0},
        "dev": {"kind": "raw", "source": source, "offset": 0},
        "test": {"kind": "raw", "source": source, "offset": 0},
    }


def _transform_block() -> dict[str, object]:
    return TwoWikiProvenanceTransformConfig().model_dump(mode="json")


@pytest.mark.parametrize(
    "override",
    (
        {"edge_scorer": "hybrid"},
        {"seed": 29},
        {"candidate_cap": 16},
        {"dev_fraction": 0.4},
        {"semantic_temperature": 0.2},
        {"weight_floor": 0.25},
    ),
)
def test_identity_changes_when_any_parameter_changes(
    override: dict[str, object],
) -> None:
    base = TwoWikiProvenanceTransformConfig()
    changed = TwoWikiProvenanceTransformConfig.model_validate(
        {**base.model_dump(mode="python"), **override}
    )
    assert base.identity() != changed.identity()
    # Equal defaults stay equal (stable identity).
    assert TwoWikiProvenanceTransformConfig().identity() == base.identity()


def test_dense_model_affects_identity_only_for_dense_aware_scorers() -> None:
    bm25_base = TwoWikiProvenanceTransformConfig(edge_scorer="bm25")
    bm25_other = TwoWikiProvenanceTransformConfig(
        edge_scorer="bm25", dense_model="models/other"
    )
    assert bm25_base.identity() == bm25_other.identity()
    assert bm25_base.identity()["dense"] is None

    hybrid_base = TwoWikiProvenanceTransformConfig(edge_scorer="hybrid")
    hybrid_other = TwoWikiProvenanceTransformConfig(
        edge_scorer="hybrid", dense_model="models/other"
    )
    assert hybrid_base.identity() != hybrid_other.identity()
    assert hybrid_base.identity()["dense"] is not None


def test_version_tag_shape_and_encoder_sensitivity() -> None:
    config = TwoWikiProvenanceTransformConfig(edge_scorer="hybrid")
    tag = transform_version_tag(config, schema_version=3)
    assert tag.startswith("v3-")
    assert len(tag) == len("v3-") + 16
    with_encoder = transform_version_tag(
        config, schema_version=3, encoder_digest="abc123"
    )
    assert tag != with_encoder
    assert transform_version_tag(config, schema_version=4).startswith("v4-")


@pytest.mark.parametrize(
    ("payload", "match"),
    (
        ({"name": "twowiki_provenance", "splits": _splits()}, "requires a transform"),
        (
            {
                "name": "twowiki",
                "splits": _splits(),
                "transform": _transform_block(),
            },
            "must not define a transform",
        ),
    ),
    ids=("provenance-requires-transform", "evidence-rejects-transform"),
)
def test_dataset_transform_block_guards(
    payload: dict[str, object], match: str
) -> None:
    with pytest.raises(ValidationError, match=match):
        DatasetConfig.model_validate(payload)


def test_capacity_offset_is_bounded_when_present() -> None:
    with pytest.raises(ValidationError):
        DatasetConfig.model_validate(
            {
                "name": "twowiki",
                "splits": {
                    "train": {
                        "kind": "raw",
                        "source": "s.json",
                        "offset": 10,
                        "capacity": 10,
                    },
                    "dev": {"kind": "raw", "source": "s.json", "offset": 0},
                    "test": {"kind": "raw", "source": "s.json", "offset": 0},
                },
            }
        )

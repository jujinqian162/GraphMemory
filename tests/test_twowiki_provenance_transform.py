from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from graph_memory.experiment.config import (
    DatasetConfig,
    TwoWikiProvenanceTransformConfig,
)
from graph_memory.stages.transform import (
    transform_version_tag,
)


def _splits(source: str = "s.json") -> dict[str, object]:
    return {
        "train": {"kind": "raw", "source": source, "offset": 0},
        "dev": {"kind": "raw", "source": source, "offset": 0},
        "test": {"kind": "raw", "source": source, "offset": 0},
    }


def _transform_block() -> dict[str, object]:
    return TwoWikiProvenanceTransformConfig().model_dump(mode="json")


def test_identity_is_stable_for_equal_parameters() -> None:
    a = TwoWikiProvenanceTransformConfig()
    b = TwoWikiProvenanceTransformConfig()
    assert a.identity() == b.identity()

    def canonical(value: object) -> str:
        return json.dumps(value, sort_keys=True, separators=(",", ":"))

    assert canonical(a.identity()) == canonical(b.identity())


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
    changed = TwoWikiProvenanceTransformConfig(**override)
    assert base.identity() != changed.identity()


def test_dense_model_only_affects_identity_when_scorer_uses_dense() -> None:
    # bm25 never consults the dense model, so its identity is model-agnostic.
    bm25_base = TwoWikiProvenanceTransformConfig(edge_scorer="bm25")
    bm25_other = TwoWikiProvenanceTransformConfig(
        edge_scorer="bm25", dense_model="models/other"
    )
    assert bm25_base.identity() == bm25_other.identity()
    # hybrid consumes the dense model, so changing it must change identity.
    hybrid_base = TwoWikiProvenanceTransformConfig(edge_scorer="hybrid")
    hybrid_other = TwoWikiProvenanceTransformConfig(
        edge_scorer="hybrid", dense_model="models/other"
    )
    assert hybrid_base.identity() != hybrid_other.identity()


def test_bm25_identity_has_no_dense_block_but_hybrid_does() -> None:
    assert TwoWikiProvenanceTransformConfig(edge_scorer="bm25").identity()["dense"] is None
    assert (
        TwoWikiProvenanceTransformConfig(edge_scorer="hybrid").identity()["dense"]
        is not None
    )


def test_version_tag_shape_and_encoder_sensitivity() -> None:
    config = TwoWikiProvenanceTransformConfig(edge_scorer="hybrid")
    tag = transform_version_tag(config, schema_version=3)
    assert tag.startswith("v3-")
    assert len(tag) == len("v3-") + 16
    with_encoder = transform_version_tag(
        config, schema_version=3, encoder_digest="abc123"
    )
    assert tag != with_encoder
    # same schema version, different version number
    assert transform_version_tag(config, schema_version=4).startswith("v4-")


def test_invalid_transform_parameters_are_rejected() -> None:
    with pytest.raises(ValidationError):
        TwoWikiProvenanceTransformConfig(dev_fraction=1.5)
    with pytest.raises(ValidationError):
        TwoWikiProvenanceTransformConfig(weight_floor=1.0)
    with pytest.raises(ValidationError):
        TwoWikiProvenanceTransformConfig(successors_per_output=3)
    with pytest.raises(ValidationError):
        TwoWikiProvenanceTransformConfig(candidate_cap=0)


def test_provenance_dataset_requires_transform_block() -> None:
    with pytest.raises(ValidationError, match="requires a transform"):
        DatasetConfig.model_validate(
            {"name": "twowiki_provenance", "splits": _splits()}
        )


def test_non_provenance_dataset_rejects_transform_block() -> None:
    with pytest.raises(ValidationError, match="must not define a transform"):
        DatasetConfig.model_validate(
            {
                "name": "twowiki",
                "splits": _splits(),
                "transform": _transform_block(),
            }
        )


def test_provenance_dataset_accepts_transform_block() -> None:
    config = DatasetConfig.model_validate(
        {
            "name": "twowiki_provenance",
            "splits": _splits(),
            "transform": _transform_block(),
        }
    )
    assert config.transform is not None
    assert config.transform.edge_scorer == "bm25"


def test_capacity_is_optional_and_offset_still_bounded() -> None:
    config = DatasetConfig.model_validate(
        {
            "name": "twowiki",
            "splits": {
                "train": {"kind": "raw", "source": "s.json", "offset": 0},
                "dev": {"kind": "raw", "source": "s.json", "offset": 0},
                "test": {"kind": "raw", "source": "s.json", "offset": 0},
            },
        }
    )
    assert config.splits.train.capacity is None
    # capacity present still validates offset < capacity
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

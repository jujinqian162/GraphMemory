from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from graph_memory.experiment.artifacts import FileSourceRef
from graph_memory.experiment.config import (
    parse_composed_config,
    resolve_experiment_config,
)
from graph_memory.experiment import workflow
from graph_memory.stages.transform import TwoWikiProvenanceTransformResult


ROOT = Path(__file__).resolve().parents[1]


def _resolve(*overrides: str):
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        composed = compose(
            config_name="config",
            overrides=[
                "name=fork-test",
                "profile=smoke",
                "device=cpu",
                *overrides,
            ],
        )
    return resolve_experiment_config(parse_composed_config(composed), repository_root=ROOT)


def _stub_ref(uri: str) -> FileSourceRef:
    return FileSourceRef(uri=uri, digest="0" * 64, size_bytes=1)


def test_provenance_dataset_routes_split_sources_through_transform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolved = _resolve(
        "dataset=twowiki_provenance",
        "method=execution_provenance_rgcn_retriever",
    )

    calls: list[dict[str, object]] = []

    def fake_transform(**kwargs: object) -> TwoWikiProvenanceTransformResult:
        calls.append(kwargs)
        return TwoWikiProvenanceTransformResult(
            version_tag="v3-deadbeefdeadbeef",
            train=_stub_ref("data/twowiki_provenance/raw/v3-x/train.json"),
            dev=_stub_ref("data/twowiki_provenance/raw/v3-x/dev.json"),
            test=_stub_ref("data/twowiki_provenance/raw/v3-x/test.json"),
        )

    monkeypatch.setattr(workflow, "transform_twowiki_task", fake_transform)

    sources = workflow._resolve_split_sources(resolved)

    # The fork ran exactly one transform and derived all three splits from it.
    assert len(calls) == 1
    assert set(sources) == {"train", "dev", "test"}
    assert sources["test"].uri.endswith("v3-x/test.json")
    # hybrid scorer means the encoder identity must be threaded into the transform.
    assert calls[0]["encoder_source"] is not None
    assert calls[0]["config"] is resolved.dataset.transform


def test_non_provenance_dataset_never_calls_transform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolved = _resolve("dataset=2wiki", "method=bm25")

    def forbidden(**_kwargs: object) -> TwoWikiProvenanceTransformResult:
        raise AssertionError("non-provenance dataset must not run the transform")

    monkeypatch.setattr(workflow, "transform_twowiki_task", forbidden)

    sources = workflow._resolve_split_sources(resolved)

    assert set(sources) == {"train", "dev", "test"}
    # Direct sources point at the dataset's own raw files, not a version dir.
    assert "twowiki_provenance" not in sources["train"].uri

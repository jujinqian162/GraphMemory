from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from graph_memory.experiment.inspect import InspectionKind, inspect_catalog
from graph_memory.experiment.reset import reset_named_run

ROOT = Path(__file__).resolve().parents[1]


def test_inspection_exposes_only_supported_typed_catalog_kinds() -> None:
    assert inspect_catalog("stages", repository_root=ROOT) == [
        "prepare",
        "graphs",
        "pairs",
        "tune",
        "train",
        "retrieve",
        "evaluate",
        "aggregate",
    ]
    methods = inspect_catalog("methods", repository_root=ROOT)
    assert isinstance(methods, list) and len(methods) == 8
    datasets = inspect_catalog("datasets", repository_root=ROOT)
    profiles = inspect_catalog("profiles", repository_root=ROOT)
    configs = inspect_catalog("configs", repository_root=ROOT)
    ablations = inspect_catalog("ablations", repository_root=ROOT)
    assert isinstance(datasets, list) and "hotpotqa" in datasets
    assert isinstance(profiles, list) and "smoke" in profiles
    assert isinstance(configs, list) and "config" in configs
    assert isinstance(ablations, dict) and "dense_rgcn_graph_retriever" in ablations
    with pytest.raises(ValueError):
        inspect_catalog(
            cast(InspectionKind, cast(object, "recipes")), repository_root=ROOT
        )


def test_reset_deletes_only_direct_contained_named_run_for_single_or_multirun(
    tmp_path: Path,
) -> None:
    named = tmp_path / "runs" / "demo"
    (named / "0_dataset=hotpotqa").mkdir(parents=True)
    (named / "0_dataset=hotpotqa" / "run_state.yaml").write_text(
        "state", encoding="utf-8"
    )
    neighbor = tmp_path / "runs" / "keep"
    neighbor.mkdir(parents=True)

    target = reset_named_run("demo", repository_root=tmp_path)
    assert target == named.resolve()
    assert not named.exists()
    assert neighbor.is_dir()

    with pytest.raises(ValueError, match="run name"):
        reset_named_run("../escape", repository_root=tmp_path)


def test_reset_refuses_run_root_symlink(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    link = tmp_path / "runs" / "linked"
    link.parent.mkdir()
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symbolic links are unavailable on this Windows host")
    with pytest.raises(ValueError, match="symbolic-link"):
        reset_named_run("linked", repository_root=tmp_path)
    assert outside.is_dir()

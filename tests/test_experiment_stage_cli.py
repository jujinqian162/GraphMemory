from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from pydantic import ValidationError

from graph_memory.experiment.config import (
    resolve_experiment_config,
    validate_composed_config,
)
from graph_memory.experiment.layout import RunLayout
from graph_memory.experiment.persistence import read_yaml, write_yaml_atomic
from graph_memory.experiment.planning import WorkflowPlanner
from graph_memory.experiment.stage_cli import (
    invocation_from_stage_config,
    load_stage_execution,
)
from graph_memory.experiment.stage_models import GraphStageConfig

ROOT = Path(__file__).resolve().parents[1]


def _plan(tmp_path: Path):
    with initialize_config_dir(
        config_dir=str(ROOT / "configs"),
        version_base="1.3",
    ):
        composed = compose(
            config_name="config",
            overrides=[
                "name=stage-cli",
                "profile=smoke",
                "dataset=hotpotqa-memory-stream",
                "methods=[bm25,dense,memory_stream,bm25_graph_rerank,dense_graph_rerank,dense_rgcn_graph_retriever,dense_ft,dense_ft_rgcn_graph_retriever]",
            ],
        )
    config = resolve_experiment_config(
        validate_composed_config(composed),
        repository_root=ROOT,
    )
    return WorkflowPlanner(config, RunLayout(tmp_path, "stage-cli")).build()


def test_direct_stage_invocation_binding_matches_every_planned_method_and_stage(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    for planned in plan.invocations:
        direct = invocation_from_stage_config(
            planned.config,
            config_path=planned.config_path,
            script=planned.script,
        )
        assert direct.identifier == planned.identifier
        assert direct.method == planned.method
        assert direct.split == planned.split
        assert direct.variant == planned.variant
        assert direct.inputs == planned.inputs
        assert direct.outputs == planned.outputs


def test_stage_loader_accepts_only_one_resolved_yaml_contract(tmp_path: Path) -> None:
    graph = next(item for item in _plan(tmp_path).invocations if item.stage == "graphs")
    write_yaml_atomic(graph.config_path, graph.config)
    loaded = load_stage_execution(
        ["--config", str(graph.config_path)],
        GraphStageConfig,
        description="test",
        script=graph.script,
    )
    assert loaded.config == graph.config
    assert loaded.invocation.inputs == graph.inputs
    assert loaded.invocation.outputs == graph.outputs

    with pytest.raises(SystemExit):
        load_stage_execution(
            ["--config", str(graph.config_path), "--dataset", "hotpotqa"],
            GraphStageConfig,
            description="test",
        )

    legacy_json = tmp_path / "legacy.json"
    legacy_json.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit, match="2"):
        load_stage_execution(
            ["--config", str(legacy_json)],
            GraphStageConfig,
            description="test",
        )


def test_stage_loader_rejects_unknown_fields_and_relative_paths(tmp_path: Path) -> None:
    graph = next(item for item in _plan(tmp_path).invocations if item.stage == "graphs")
    primitive = (
        read_yaml(graph.config_path)
        if graph.config_path.exists()
        else graph.config.model_dump(mode="json")
    )
    assert isinstance(primitive, dict)
    primitive["unknown"] = True
    unknown = tmp_path / "unknown.yaml"
    write_yaml_atomic(unknown, primitive)
    with pytest.raises(ValidationError, match="extra_forbidden"):
        load_stage_execution(
            ["--config", str(unknown)], GraphStageConfig, description="test"
        )

    relative_primitive = graph.config.model_dump(mode="json")
    relative_primitive["tasks"] = "relative.json"
    relative = tmp_path / "relative.yaml"
    write_yaml_atomic(relative, relative_primitive)
    with pytest.raises(ValueError, match="paths must be absolute"):
        load_stage_execution(
            ["--config", str(relative)], GraphStageConfig, description="test"
        )

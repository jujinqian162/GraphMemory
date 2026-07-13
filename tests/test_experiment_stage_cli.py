from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from pydantic import ValidationError

from graph_memory.experiment.config import (
    AliasArtifactRef,
    ArtifactRef,
    resolve_experiment_config,
    validate_composed_config,
)
from graph_memory.experiment.layout import RunLayout
from graph_memory.experiment.persistence import read_yaml, write_yaml_atomic
from graph_memory.experiment.planning import WorkflowPlanner
from graph_memory.experiment.stage_cli import load_stage_execution
from graph_memory.experiment.stage_models import (
    Bm25GraphRerankTuneStageConfig,
    DenseGraphRerankTuneStageConfig,
    GraphStageConfig,
    OrdinaryRgcnTrainStageConfig,
    SeededRgcnTrainStageConfig,
)

ROOT = Path(__file__).resolve().parents[1]


def _plan(tmp_path: Path, *extra_overrides: str):
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
                *extra_overrides,
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
    plans = (
        _plan(tmp_path),
        _plan(
            tmp_path,
            "dataset=hotpotqa",
            "methods=[dense_rgcn_graph_retriever]",
            "ablation.enable=true",
            "ablation.variants=[wo_graph]",
        ),
    )
    for plan in plans:
        for planned in plan.invocations:
            write_yaml_atomic(planned.config_path, planned)
            loaded = load_stage_execution(
                ["--config", str(planned.config_path)],
                type(planned.config),
                description="test",
                script=planned.script,
            )
            assert loaded.invocation == planned
            assert loaded.config == planned.config


def test_stage_loader_accepts_only_one_resolved_yaml_contract(tmp_path: Path) -> None:
    graph = next(item for item in _plan(tmp_path).invocations if item.stage == "graphs")
    write_yaml_atomic(graph.config_path, graph)
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
            script=graph.script,
        )

    legacy_json = tmp_path / "legacy.json"
    legacy_json.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit, match="2"):
        load_stage_execution(
            ["--config", str(legacy_json)],
            GraphStageConfig,
            description="test",
            script=graph.script,
        )


def test_stage_loader_rejects_unknown_fields_and_relative_paths(tmp_path: Path) -> None:
    graph = next(item for item in _plan(tmp_path).invocations if item.stage == "graphs")
    primitive = (
        read_yaml(graph.config_path)
        if graph.config_path.exists()
        else graph.model_dump(mode="json")
    )
    assert isinstance(primitive, dict)
    primitive["unknown"] = True
    unknown = tmp_path / "unknown.yaml"
    write_yaml_atomic(unknown, primitive)
    with pytest.raises(ValidationError, match="extra_forbidden"):
        load_stage_execution(
            ["--config", str(unknown)],
            GraphStageConfig,
            description="test",
            script=graph.script,
        )

    relative_primitive = graph.model_dump(mode="json")
    relative = tmp_path / "relative.yaml"
    relative_primitive["config_path"] = str(relative.resolve())
    inputs = relative_primitive["inputs"]
    assert isinstance(inputs, list) and isinstance(inputs[0], dict)
    inputs[0]["path"] = "relative.json"
    write_yaml_atomic(relative, relative_primitive)
    with pytest.raises(ValueError, match="paths must be absolute"):
        load_stage_execution(
            ["--config", str(relative)],
            GraphStageConfig,
            description="test",
            script=graph.script,
        )


def test_variant_models_reject_half_configurations(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    dense_tune = next(
        item.config
        for item in plan.invocations
        if item.stage == "tune" and item.method == "dense_graph_rerank"
    )
    assert isinstance(dense_tune, DenseGraphRerankTuneStageConfig)
    missing_encoder = dense_tune.model_dump()
    encoder = missing_encoder.pop("encoder")
    with pytest.raises(ValidationError, match="encoder"):
        DenseGraphRerankTuneStageConfig.model_validate(missing_encoder)

    bm25_tune = next(
        item.config
        for item in plan.invocations
        if item.stage == "tune" and item.method == "bm25_graph_rerank"
    )
    assert isinstance(bm25_tune, Bm25GraphRerankTuneStageConfig)
    unexpected_encoder = {**bm25_tune.model_dump(), "encoder": encoder}
    with pytest.raises(ValidationError, match="extra_forbidden"):
        Bm25GraphRerankTuneStageConfig.model_validate(unexpected_encoder)

    seeded_train = next(
        item.config
        for item in plan.invocations
        if item.stage == "train" and item.method == "dense_ft_rgcn_graph_retriever"
    )
    assert isinstance(seeded_train, SeededRgcnTrainStageConfig)
    missing_seed = seeded_train.model_dump()
    seed_model_dir = missing_seed.pop("seed_model_dir")
    with pytest.raises(ValidationError, match="seed_model_dir"):
        SeededRgcnTrainStageConfig.model_validate(missing_seed)

    ordinary_train = next(
        item.config
        for item in plan.invocations
        if item.stage == "train" and item.method == "dense_rgcn_graph_retriever"
    )
    assert isinstance(ordinary_train, OrdinaryRgcnTrainStageConfig)
    unexpected_seed = {**ordinary_train.model_dump(), "seed_model_dir": seed_model_dir}
    with pytest.raises(ValidationError, match="extra_forbidden"):
        OrdinaryRgcnTrainStageConfig.model_validate(unexpected_seed)

    artifact = ArtifactRef(role="metrics", path=tmp_path.resolve(), kind="file")
    with pytest.raises(ValidationError, match="extra_forbidden"):
        ArtifactRef.model_validate({**artifact.model_dump(), "alias_of": tmp_path})
    with pytest.raises(ValidationError, match="alias_of"):
        AliasArtifactRef.model_validate(artifact.model_dump())

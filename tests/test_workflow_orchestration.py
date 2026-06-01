from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from graph_memory.retrieval_registry import get_supported_methods
from scripts.workflow.artifacts import build_main_method_artifacts, build_variant_artifact_namespace
from scripts.workflow.manifest import initialize_experiment
from scripts.workflow.planner import build_stage_plan, earliest_invalidated_stage, format_commands
from scripts.workflow.registry import (
    METHOD_WORKFLOW_REGISTRY,
    RGCN_ABLATION_SUITE,
    discover_ablation_variants,
    validate_workflow_registry,
)
from scripts.workflow.types import (
    ArtifactRole,
    ArtifactState,
    ChangeDimension,
    RgcnAblationVariant,
    StageId,
    WorkflowId,
)
from scripts.workflow.workflows import RGCN_WORKFLOW, STATELESS_RETRIEVAL_WORKFLOW
from scripts.workflow.status import inspect_experiment_status
import scripts.experiment as experiment_script
from tests.test_experiment_runner import (
    TRAINABLE_METHOD,
    _write_rgcn_training_config,
    _write_trainable_experiment_config,
)


def test_closed_workflow_values_expose_allowed_choices() -> None:
    assert [stage.value for stage in StageId] == [
        "prepare",
        "graphs",
        "pairs",
        "tune",
        "train",
        "retrieve",
        "evaluate",
        "aggregate",
    ]
    assert {workflow.value for workflow in WorkflowId} == {
        "stateless_retrieval",
        "graph_rerank",
        "rgcn_trainable_retrieval",
    }
    assert {role.value for role in ArtifactRole} >= {
        "inputs",
        "graphs",
        "train_pairs",
        "checkpoint",
        "predictions",
        "metrics",
    }
    assert {dimension.value for dimension in ChangeDimension} == {
        "pair_sampling",
        "model_structure",
        "model_graph_view",
    }
    assert {state.value for state in ArtifactState} == {"missing", "complete", "stale", "alias"}


def test_rgcn_suite_exposes_registered_variants_without_random_edges() -> None:
    rows = discover_ablation_variants("dense_rgcn_graph_retriever")

    assert [row["variant"] for row in rows] == [variant.value for variant in RgcnAblationVariant]
    assert "random_edges" not in {row["variant"] for row in rows}


def test_rgcn_workflow_declares_dimension_driven_invalidation() -> None:
    steps = {step.stage: step for step in RGCN_WORKFLOW.steps}

    assert steps[StageId.PAIRS].invalidated_by == frozenset({ChangeDimension.PAIR_SAMPLING})
    assert steps[StageId.TRAIN].invalidated_by == frozenset(
        {
            ChangeDimension.PAIR_SAMPLING,
            ChangeDimension.MODEL_STRUCTURE,
            ChangeDimension.MODEL_GRAPH_VIEW,
        }
    )


def test_workflow_registry_covers_every_runtime_method() -> None:
    validate_workflow_registry()

    assert set(METHOD_WORKFLOW_REGISTRY) == set(get_supported_methods())


def test_planner_validates_registry_before_building_commands(tmp_path, monkeypatch) -> None:
    config_path = _write_ablation_experiment_config(tmp_path, variants=["wo_graph"])
    manifest = initialize_experiment(
        "quick_rgcn_ablation",
        config=json.loads(config_path.read_text(encoding="utf-8")),
        run_root=tmp_path / "runs",
        profile="quick",
        methods=[TRAINABLE_METHOD],
    )
    monkeypatch.delitem(METHOD_WORKFLOW_REGISTRY, "dense")

    with pytest.raises(ValueError, match="Workflow registry mismatch"):
        build_stage_plan(
            manifest,
            methods=[TRAINABLE_METHOD],
        )


def test_same_lifecycle_method_can_reuse_existing_workflow_without_planner_branch() -> None:
    registrations = {
        **METHOD_WORKFLOW_REGISTRY,
        "test_dense_clone": STATELESS_RETRIEVAL_WORKFLOW,
    }

    validate_workflow_registry(
        runtime_methods=(*get_supported_methods(), "test_dense_clone"),
        registrations=registrations,
        suites={"dense_rgcn_graph_retriever": RGCN_ABLATION_SUITE},
    )


def test_registry_rejects_suite_for_unregistered_method() -> None:
    invalid_suite = replace(RGCN_ABLATION_SUITE, method="missing_method")

    with pytest.raises(ValueError, match="missing_method"):
        validate_workflow_registry(
            suites={"missing_method": invalid_suite},
        )


def test_registry_rejects_suite_without_exactly_one_baseline_alias() -> None:
    invalid_suite = replace(
        RGCN_ABLATION_SUITE,
        variants=tuple(replace(variant, baseline_alias=False) for variant in RGCN_ABLATION_SUITE.variants),
    )

    with pytest.raises(ValueError, match="exactly one baseline alias"):
        validate_workflow_registry(
            suites={RGCN_ABLATION_SUITE.method: invalid_suite},
        )


def test_model_structure_variant_aliases_pairs_and_allocates_train_outputs_locally(tmp_path) -> None:
    main = build_main_method_artifacts(tmp_path, "dense_rgcn_graph_retriever")
    variant = next(
        spec for spec in RGCN_ABLATION_SUITE.variants if spec.identifier is RgcnAblationVariant.WO_GRAPH
    )

    assert earliest_invalidated_stage(RGCN_WORKFLOW, variant) is StageId.TRAIN

    namespace = build_variant_artifact_namespace(tmp_path, RGCN_ABLATION_SUITE.method, variant, main)

    assert namespace.path(ArtifactRole.TRAIN_PAIRS) == main[ArtifactRole.TRAIN_PAIRS]
    assert Path(namespace.path(ArtifactRole.CHECKPOINT)) == (
        tmp_path / "ablations" / RGCN_ABLATION_SUITE.method / "wo_graph" / "checkpoints" / "best.pt"
    )
    assert {alias.role for alias in namespace.aliases} >= {ArtifactRole.TRAIN_PAIRS}


def test_pair_sampling_variant_allocates_pairs_and_all_downstream_outputs_locally(tmp_path) -> None:
    main = build_main_method_artifacts(tmp_path, "dense_rgcn_graph_retriever")
    variant = next(
        spec
        for spec in RGCN_ABLATION_SUITE.variants
        if spec.identifier is RgcnAblationVariant.WO_HARD_NEGATIVES
    )

    assert earliest_invalidated_stage(RGCN_WORKFLOW, variant) is StageId.PAIRS

    namespace = build_variant_artifact_namespace(tmp_path, RGCN_ABLATION_SUITE.method, variant, main)

    assert Path(namespace.path(ArtifactRole.TRAIN_PAIRS)) == (
        tmp_path / "ablations" / RGCN_ABLATION_SUITE.method / "wo_hard_negatives" / "train.pairs.json"
    )
    assert Path(namespace.path(ArtifactRole.CHECKPOINT)) == (
        tmp_path / "ablations" / RGCN_ABLATION_SUITE.method / "wo_hard_negatives" / "checkpoints" / "best.pt"
    )
    assert Path(namespace.path(ArtifactRole.METRICS)) == (
        tmp_path / "ablations" / RGCN_ABLATION_SUITE.method / "wo_hard_negatives" / "metrics" / "test.metrics.csv"
    )


def test_full_rgcn_variant_aliases_all_main_method_outputs(tmp_path) -> None:
    main = build_main_method_artifacts(tmp_path, "dense_rgcn_graph_retriever")
    variant = next(
        spec for spec in RGCN_ABLATION_SUITE.variants if spec.identifier is RgcnAblationVariant.FULL_RGCN
    )

    assert earliest_invalidated_stage(RGCN_WORKFLOW, variant) is None

    namespace = build_variant_artifact_namespace(tmp_path, RGCN_ABLATION_SUITE.method, variant, main)

    assert namespace.paths == main
    assert {alias.role for alias in namespace.aliases} == set(main)


def test_ablation_enabled_manifest_writes_variant_namespaces_configs_and_metric_index(tmp_path) -> None:
    config_path = _write_ablation_experiment_config(tmp_path, variants=["wo_graph", "wo_hard_negatives"])

    manifest = initialize_experiment(
        "quick_rgcn_ablation",
        config=json.loads(config_path.read_text(encoding="utf-8")),
        run_root=tmp_path / "runs",
        profile="quick",
        methods=[TRAINABLE_METHOD],
    )

    run_dir = tmp_path / "runs" / "quick_rgcn_ablation"
    ablations = manifest["artifacts"]["ablations"][TRAINABLE_METHOD]
    main = build_main_method_artifacts(run_dir, TRAINABLE_METHOD)
    assert manifest["schema_version"] == 2
    assert list(ablations) == ["full_rgcn", "wo_graph", "wo_hard_negatives"]
    assert ablations["full_rgcn"]["artifacts"]["checkpoint"] == main[ArtifactRole.CHECKPOINT]
    assert ablations["wo_graph"]["invalidated_from"] == "train"
    assert ablations["wo_graph"]["artifacts"]["train_pairs"] == main[ArtifactRole.TRAIN_PAIRS]
    assert ablations["wo_hard_negatives"]["invalidated_from"] == "pairs"
    assert Path(ablations["wo_hard_negatives"]["artifacts"]["train_pairs"]) == (
        run_dir / "ablations" / TRAINABLE_METHOD / "wo_hard_negatives" / "train.pairs.json"
    )

    main_config = manifest["effective_config"]["training"][TRAINABLE_METHOD]
    wo_graph_config = json.loads(
        (run_dir / "ablations" / TRAINABLE_METHOD / "wo_graph" / "effective_training_config.json").read_text(
            encoding="utf-8"
        )
    )
    hard_negative_config = json.loads(
        (
            run_dir
            / "ablations"
            / TRAINABLE_METHOD
            / "wo_hard_negatives"
            / "effective_training_config.json"
        ).read_text(encoding="utf-8")
    )
    assert wo_graph_config["optimization"] == main_config["optimization"]
    assert wo_graph_config["pair_sampling"] == main_config["pair_sampling"]
    assert wo_graph_config["model"]["ablation"] == "wo_graph"
    assert wo_graph_config["model"]["num_layers"] == 0
    assert hard_negative_config["model"] == main_config["model"]
    assert hard_negative_config["optimization"] == main_config["optimization"]
    assert hard_negative_config["pair_sampling"]["hard_bm25_per_positive"] == 0
    assert hard_negative_config["pair_sampling"]["hard_dense_per_positive"] == 0
    assert hard_negative_config["pair_sampling"]["hard_graph_neighbor_per_positive"] == 0

    metric_index = json.loads((run_dir / "config" / "ablation_metrics_index.json").read_text(encoding="utf-8"))
    assert metric_index["metrics"][0] == {
        "method": TRAINABLE_METHOD,
        "variant": "full_rgcn",
        "metrics_path": main[ArtifactRole.METRICS],
    }
    assert Path(manifest["artifacts"]["tables"]["ablation"]) == run_dir / "tables" / "ablation_results.csv"


def test_ablation_enabled_manifest_rejects_unknown_variant_with_allowed_values(tmp_path) -> None:
    config_path = _write_ablation_experiment_config(tmp_path, variants=["not_a_variant"])

    with pytest.raises(ValueError, match="allowed values:.*full_rgcn.*wo_hard_negatives"):
        initialize_experiment(
            "quick_rgcn_ablation",
            config=json.loads(config_path.read_text(encoding="utf-8")),
            run_root=tmp_path / "runs",
            profile="quick",
            methods=[TRAINABLE_METHOD],
        )


def test_manifest_rejects_non_boolean_enable_ablation(tmp_path) -> None:
    config_path = _write_ablation_experiment_config(tmp_path, variants=["wo_graph"])
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["enable_ablation"] = "yes"

    with pytest.raises(ValueError, match="enable_ablation must be a boolean"):
        initialize_experiment(
            "quick_rgcn_ablation",
            config=config,
            run_root=tmp_path / "runs",
            profile="quick",
            methods=[TRAINABLE_METHOD],
        )


def test_planner_rejects_unknown_stage_with_allowed_values(tmp_path) -> None:
    config_path = _write_ablation_experiment_config(tmp_path, variants=["wo_graph"])
    manifest = initialize_experiment(
        "quick_rgcn_ablation",
        config=json.loads(config_path.read_text(encoding="utf-8")),
        run_root=tmp_path / "runs",
        profile="quick",
        methods=[TRAINABLE_METHOD],
    )

    with pytest.raises(ValueError, match="allowed values:") as exc_info:
        build_stage_plan(
            manifest,
            methods=[TRAINABLE_METHOD],
            from_stage="not_a_stage",
        )
    assert "prepare" in str(exc_info.value)
    assert "aggregate" in str(exc_info.value)


def test_ablation_only_plan_schedules_shared_pairs_and_variant_local_downstream_commands(tmp_path) -> None:
    config_path = _write_ablation_experiment_config(tmp_path, variants=["wo_graph", "wo_hard_negatives"])
    manifest = initialize_experiment(
        "quick_rgcn_ablation",
        config=json.loads(config_path.read_text(encoding="utf-8")),
        run_root=tmp_path / "runs",
        profile="quick",
        methods=[TRAINABLE_METHOD],
    )
    _write_main_rgcn_metrics_placeholder(manifest)

    commands = build_stage_plan(
        manifest,
        methods=[TRAINABLE_METHOD],
        variants=["wo_graph"],
        ablations_only=True,
    )

    assert [command.stage for command in commands] == [
        StageId.PREPARE,
        StageId.PREPARE,
        StageId.PREPARE,
        StageId.GRAPHS,
        StageId.GRAPHS,
        StageId.GRAPHS,
        StageId.PAIRS,
        StageId.TRAIN,
        StageId.RETRIEVE,
        StageId.EVALUATE,
        StageId.AGGREGATE,
    ]
    assert [command.variant for command in commands if command.stage is StageId.TRAIN] == ["wo_graph"]
    assert [command.variant for command in commands if command.stage is StageId.PAIRS] == [None]
    train_command = next(command for command in commands if command.stage is StageId.TRAIN)
    output_dir = Path(train_command.argv[train_command.argv.index("--output_dir") + 1])
    assert output_dir == (
        tmp_path / "runs" / "quick_rgcn_ablation" / "ablations" / TRAINABLE_METHOD / "wo_graph"
    )
    evaluate_command = next(command for command in commands if command.stage is StageId.EVALUATE)
    reference_graph = evaluate_command.argv[evaluate_command.argv.index("--graphs") + 1]
    assert reference_graph == manifest["artifacts"]["graphs"]["test"]
    aggregate = next(command for command in commands if command.stage is StageId.AGGREGATE)
    assert "--ablation_index" in aggregate.argv
    assert "--output_ablation" in aggregate.argv
    assert _repeated_arg_values(aggregate.argv, "--ablation_selection") == [
        f"{TRAINABLE_METHOD}=full_rgcn",
        f"{TRAINABLE_METHOD}=wo_graph",
    ]


def test_ablation_only_plan_requires_main_baseline_metrics_before_scheduling(tmp_path) -> None:
    config_path = _write_ablation_experiment_config(tmp_path, variants=["wo_graph"])
    manifest = initialize_experiment(
        "quick_rgcn_ablation",
        config=json.loads(config_path.read_text(encoding="utf-8")),
        run_root=tmp_path / "runs",
        profile="quick",
        methods=[TRAINABLE_METHOD],
    )
    baseline_metrics = manifest["artifacts"]["metrics"][TRAINABLE_METHOD]

    with pytest.raises(ValueError, match="Ablation-only execution requires ordinary baseline metrics") as exc_info:
        build_stage_plan(
            manifest,
            methods=[TRAINABLE_METHOD],
            variants=["wo_graph"],
            ablations_only=True,
            stages=["prepare"],
        )
    assert baseline_metrics in str(exc_info.value)


def test_ablation_only_plan_rejects_method_without_registered_suite(tmp_path) -> None:
    config_path = _write_ablation_experiment_config(tmp_path, variants=["wo_graph"])
    manifest = initialize_experiment(
        "quick_rgcn_ablation",
        config=json.loads(config_path.read_text(encoding="utf-8")),
        run_root=tmp_path / "runs",
        profile="quick",
        methods=["bm25", TRAINABLE_METHOD],
    )

    with pytest.raises(ValueError, match="No executable ablation variants.*bm25"):
        build_stage_plan(
            manifest,
            methods=["bm25"],
            ablations_only=True,
        )


def test_variant_resume_from_retrieve_uses_existing_checkpoint_without_hidden_train(tmp_path) -> None:
    config_path = _write_ablation_experiment_config(tmp_path, variants=["wo_graph"])
    manifest = initialize_experiment(
        "quick_rgcn_ablation",
        config=json.loads(config_path.read_text(encoding="utf-8")),
        run_root=tmp_path / "runs",
        profile="quick",
        methods=[TRAINABLE_METHOD],
    )
    _write_main_rgcn_metrics_placeholder(manifest)
    checkpoint = Path(
        manifest["artifacts"]["ablations"][TRAINABLE_METHOD]["wo_graph"]["artifacts"]["checkpoint"]
    )
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"checkpoint")

    commands = build_stage_plan(
        manifest,
        methods=[TRAINABLE_METHOD],
        variants=["wo_graph"],
        ablations_only=True,
        from_stage="retrieve",
    )

    assert [command.stage for command in commands] == [StageId.RETRIEVE, StageId.EVALUATE, StageId.AGGREGATE]
    assert all(command.stage is not StageId.TRAIN for command in commands)


def test_variant_resume_from_retrieve_reports_missing_checkpoint_without_hidden_train(tmp_path) -> None:
    config_path = _write_ablation_experiment_config(tmp_path, variants=["wo_graph"])
    manifest = initialize_experiment(
        "quick_rgcn_ablation",
        config=json.loads(config_path.read_text(encoding="utf-8")),
        run_root=tmp_path / "runs",
        profile="quick",
        methods=[TRAINABLE_METHOD],
    )
    checkpoint = manifest["artifacts"]["ablations"][TRAINABLE_METHOD]["wo_graph"]["artifacts"]["checkpoint"]

    with pytest.raises(ValueError, match=f"Missing trainable checkpoints.*{Path(checkpoint).name}"):
        build_stage_plan(
            manifest,
            methods=[TRAINABLE_METHOD],
            variants=["wo_graph"],
            ablations_only=True,
            from_stage="retrieve",
        )


def test_variant_plan_rendering_and_status_are_variant_qualified(tmp_path) -> None:
    config_path = _write_ablation_experiment_config(tmp_path, variants=["wo_graph"])
    manifest = initialize_experiment(
        "quick_rgcn_ablation",
        config=json.loads(config_path.read_text(encoding="utf-8")),
        run_root=tmp_path / "runs",
        profile="quick",
        methods=[TRAINABLE_METHOD],
    )
    _write_main_rgcn_metrics_placeholder(manifest)

    commands = build_stage_plan(
        manifest,
        methods=[TRAINABLE_METHOD],
        variants=["wo_graph"],
        ablations_only=True,
    )
    rendered = format_commands(commands)
    status = inspect_experiment_status(manifest)

    assert "train method=dense_rgcn_graph_retriever variant=wo_graph" in rendered
    assert _variant_state(status, "train", "full_rgcn") == "alias"
    assert _variant_state(status, "train", "wo_graph") == "missing"
    assert _variant_state(status, "pairs", "wo_graph") == "alias"


def test_experiment_cli_lists_and_filters_ablation_variants(tmp_path, capsys) -> None:
    assert experiment_script.main(["ablations", "list", "--method", TRAINABLE_METHOD]) == 0
    discovery_output = capsys.readouterr().out

    assert "full_rgcn" in discovery_output
    assert "wo_hard_negatives" in discovery_output
    assert "random_edges" not in discovery_output

    config_path = _write_ablation_experiment_config(tmp_path, variants=["wo_graph", "wo_hard_negatives"])
    run_root = tmp_path / "runs"
    assert experiment_script.main(
        [
            "init",
            "quick_rgcn_ablation",
            "--config",
            str(config_path),
            "--profile",
            "quick",
            "--methods",
            TRAINABLE_METHOD,
            "--run-root",
            str(run_root),
        ]
    ) == 0
    capsys.readouterr()
    _write_main_rgcn_metrics_placeholder(
        json.loads((run_root / "quick_rgcn_ablation" / "manifest.json").read_text(encoding="utf-8"))
    )

    assert experiment_script.main(
        [
            "plan",
            "quick_rgcn_ablation",
            "--methods",
            TRAINABLE_METHOD,
            "--variant",
            "wo_graph",
            "--ablations-only",
            "--run-root",
            str(run_root),
        ]
    ) == 0
    plan_output = capsys.readouterr().out

    assert "variant=wo_graph" in plan_output
    assert "variant=wo_hard_negatives" not in plan_output


def test_hard_negative_variant_pair_command_uses_local_config_and_pair_output(tmp_path) -> None:
    config_path = _write_ablation_experiment_config(tmp_path, variants=["wo_hard_negatives"])
    manifest = initialize_experiment(
        "quick_rgcn_ablation",
        config=json.loads(config_path.read_text(encoding="utf-8")),
        run_root=tmp_path / "runs",
        profile="quick",
        methods=[TRAINABLE_METHOD],
    )
    _write_main_rgcn_metrics_placeholder(manifest)

    commands = build_stage_plan(
        manifest,
        methods=[TRAINABLE_METHOD],
        variants=["wo_hard_negatives"],
        ablations_only=True,
    )

    pair_command = next(command for command in commands if command.stage is StageId.PAIRS)
    variant_root = tmp_path / "runs" / "quick_rgcn_ablation" / "ablations" / TRAINABLE_METHOD / "wo_hard_negatives"
    assert pair_command.variant == "wo_hard_negatives"
    assert (variant_root / "effective_training_config.json").as_posix() in pair_command.argv
    assert (variant_root / "train.pairs.json").as_posix() in pair_command.argv


def test_schema_version_one_manifest_remains_readable_but_cannot_enable_ablation_in_place(tmp_path) -> None:
    config_path = _write_ablation_experiment_config(tmp_path, variants=["wo_graph"])
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["enable_ablation"] = False
    manifest = initialize_experiment(
        "legacy_rgcn",
        config=config,
        run_root=tmp_path / "runs",
        profile="quick",
        methods=[TRAINABLE_METHOD],
    )

    manifest["schema_version"] = 1
    manifest["effective_config"].pop("enable_ablation")
    manifest["effective_config"].pop("ablation_variants")
    Path(manifest["paths"]["manifest"]).write_text(json.dumps(manifest), encoding="utf-8")
    assert initialize_experiment(
        "legacy_rgcn",
        config=config,
        run_root=tmp_path / "runs",
        profile="quick",
        methods=[TRAINABLE_METHOD],
    )["schema_version"] == 1

    config["enable_ablation"] = True
    with pytest.raises(ValueError, match="cannot enable ablation in place"):
        initialize_experiment(
            "legacy_rgcn",
            config=config,
            run_root=tmp_path / "runs",
            profile="quick",
            methods=[TRAINABLE_METHOD],
        )


def _write_ablation_experiment_config(tmp_path: Path, *, variants: list[str]) -> Path:
    raw_path = Path("tests/fixtures/hotpotqa_smoke.json")
    training_config_path = tmp_path / "configs" / "training" / TRAINABLE_METHOD / "base.json"
    config_path = tmp_path / "configs" / "experiments" / "hotpotqa_evidence_retrieval.json"
    _write_rgcn_training_config(training_config_path)
    _write_trainable_experiment_config(config_path, raw_path, training_config_path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["enable_ablation"] = True
    payload["ablation_variants"] = {TRAINABLE_METHOD: variants}
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    return config_path


def _variant_state(status: list[dict[str, str]], stage: str, variant: str) -> str:
    return next(row["state"] for row in status if row["stage"] == stage and row.get("variant") == variant)


def _write_main_rgcn_metrics_placeholder(manifest: dict[str, object]) -> None:
    artifacts = manifest["artifacts"]
    assert isinstance(artifacts, dict)
    metrics = artifacts["metrics"]
    assert isinstance(metrics, dict)
    path = Path(metrics[TRAINABLE_METHOD])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("review baseline metrics placeholder", encoding="utf-8")


def _repeated_arg_values(argv: list[str], name: str) -> list[str]:
    return [argv[index + 1] for index, value in enumerate(argv) if value == name]

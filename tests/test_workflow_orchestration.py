from __future__ import annotations

import subprocess
import sys
import logging
from pathlib import Path
from typing import cast

import pytest

from graph_memory.contracts.tasks import MemoryTaskInput
from graph_memory.io import read_json
from graph_memory.io import write_json
from graph_memory.retrieval.methods.memory_stream.contracts import ImportanceArtifact
from graph_memory.retrieval.methods.memory_stream.artifact import importance_content_digest
from scripts.workflow.manifest import initialize_experiment, load_experiment_config
from scripts.workflow.planner import build_stage_plan, earliest_invalidated_stage
from scripts.workflow.registry import (
    discover_ablation_variants,
    get_ablation_suite,
    get_workflow,
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
from scripts.workflow.workflows import RGCN_WORKFLOW
from scripts.workflow.status import inspect_experiment_status
import scripts.experiment as experiment_script

TRAINABLE_METHOD = "dense_rgcn_graph_retriever"


def _ablation_config(*, variants: list[str] | None = None) -> dict[str, object]:
    config = load_experiment_config()
    config["enable_ablation"] = True
    if variants is not None:
        config["ablation_variants"] = {TRAINABLE_METHOD: variants}
    return config


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
        "tuned_stateless_retrieval",
        "graph_rerank",
        "rgcn_trainable_retrieval",
        "dense_finetune_retrieval",
    }
    assert ArtifactRole.EFFECTIVE_METHOD_CONFIG.value == "effective_method_config"
    assert {dimension.value for dimension in ChangeDimension} == {
        "pair_sampling",
        "model_structure",
        "model_graph_view",
    }
    assert {state.value for state in ArtifactState} == {"missing", "complete", "stale", "alias"}


def test_workflow_types_import_without_stdlib_strenum() -> None:
    code = """
import builtins

original_import = builtins.__import__

def import_without_strenum(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "enum" and fromlist and "StrEnum" in fromlist:
        raise ImportError("cannot import name 'StrEnum' from 'enum'")
    return original_import(name, globals, locals, fromlist, level)

builtins.__import__ = import_without_strenum

from scripts.workflow.types import StageId

assert StageId("prepare") is StageId.PREPARE
assert str(StageId.PREPARE) == "prepare"
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_rgcn_ablation_suite_exposes_current_patch_paths() -> None:
    validate_workflow_registry()
    rows = discover_ablation_variants(TRAINABLE_METHOD)
    suite = get_ablation_suite(TRAINABLE_METHOD)

    assert [row["variant"] for row in rows] == [variant.value for variant in RgcnAblationVariant]
    assert suite is not None
    assert "random_edges" not in {row["variant"] for row in rows}
    wo_graph = next(variant for variant in suite.variants if variant.identifier is RgcnAblationVariant.WO_GRAPH)
    assert wo_graph.training_config_override == {
        "train": {"model": {"ablation": "wo_graph", "num_layers": 0}}
    }


def test_rgcn_workflow_invalidation_is_declared_by_dimension() -> None:
    suite = get_ablation_suite(TRAINABLE_METHOD)
    assert suite is not None
    assert earliest_invalidated_stage(
        get_workflow(TRAINABLE_METHOD),
        next(variant for variant in suite.variants if variant.identifier.value == "full_rgcn"),
    ) is None
    assert RGCN_WORKFLOW.steps[2].stage is StageId.PAIRS
    assert RGCN_WORKFLOW.steps[2].invalidated_by == frozenset({ChangeDimension.PAIR_SAMPLING})
    assert RGCN_WORKFLOW.steps[3].stage is StageId.TRAIN
    assert RGCN_WORKFLOW.steps[3].invalidated_by == frozenset(
        {
            ChangeDimension.PAIR_SAMPLING,
            ChangeDimension.MODEL_STRUCTURE,
            ChangeDimension.MODEL_GRAPH_VIEW,
        }
    )


def test_ablation_manifest_writes_variant_stage_configs_and_unversioned_metric_index(tmp_path: Path) -> None:
    manifest = initialize_experiment(
        "ablation",
        config=_ablation_config(variants=["wo_graph", "wo_hard_negatives"]),
        run_root=tmp_path,
        profile="smoke",
        methods=[TRAINABLE_METHOD],
        force=True,
    )

    records = manifest["artifacts"]["ablations"][TRAINABLE_METHOD]
    assert set(records) == {"full_rgcn", "wo_graph", "wo_hard_negatives"}
    assert records["full_rgcn"]["baseline_alias"] is True
    assert records["wo_graph"]["invalidated_from"] == "train"
    assert records["wo_hard_negatives"]["invalidated_from"] == "pairs"

    for variant, record in records.items():
        assert "stage_configs" in record
        for stage, path in record["stage_configs"].items():
            assert Path(path).is_file(), (variant, stage, path)

    index = read_json(manifest["paths"]["ablation_metrics_index"])
    assert "schema_version" not in index
    assert {entry["variant"] for entry in index["metrics"]} == set(records)


def test_ablation_plan_uses_variant_stage_config_commands(tmp_path: Path) -> None:
    manifest = initialize_experiment(
        "ablation-plan",
        config=_ablation_config(variants=["wo_hard_negatives"]),
        run_root=tmp_path,
        profile="smoke",
        methods=[TRAINABLE_METHOD],
        force=True,
    )
    baseline_metrics = Path(manifest["artifacts"]["metrics"][TRAINABLE_METHOD])
    baseline_metrics.parent.mkdir(parents=True, exist_ok=True)
    baseline_metrics.write_text("metric,value\nRecall@5,1\n", encoding="utf-8")
    commands = build_stage_plan(
        manifest,
        stages=["pairs", "train", "retrieve", "evaluate"],
        methods=[TRAINABLE_METHOD],
        variants=["wo_hard_negatives"],
        ablations_only=True,
    )

    variant_commands = [command for command in commands if command.variant == "wo_hard_negatives"]
    assert [command.stage for command in variant_commands] == [
        StageId.PAIRS,
        StageId.TRAIN,
        StageId.RETRIEVE,
        StageId.EVALUATE,
    ]
    assert all(command.argv[2] == "--config" for command in variant_commands)
    assert all("ablations/dense_rgcn_graph_retriever/wo_hard_negatives" in command.argv[3] for command in variant_commands)


def test_ablation_only_requires_main_baseline_metrics(tmp_path: Path) -> None:
    manifest = initialize_experiment(
        "ablation-only",
        config=_ablation_config(variants=["wo_graph"]),
        run_root=tmp_path,
        profile="smoke",
        methods=[TRAINABLE_METHOD],
        force=True,
    )

    with pytest.raises(ValueError, match="baseline metrics"):
        build_stage_plan(
            manifest,
            stages=["pairs"],
            methods=[TRAINABLE_METHOD],
            variants=["wo_graph"],
            ablations_only=True,
        )


def test_legacy_manifest_is_rejected_instead_of_read_with_fallback(tmp_path: Path) -> None:
    run_dir = tmp_path / "legacy"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text('{"schema_version": 1}', encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported fields|requires field"):
        initialize_experiment(
            "legacy",
            config=load_experiment_config(),
            run_root=tmp_path,
            profile="smoke",
            methods=[TRAINABLE_METHOD],
        )


def test_experiment_cli_lists_and_filters_ablation_variants(capsys) -> None:
    assert experiment_script.main(["ablations", "list", "--method", TRAINABLE_METHOD]) == 0
    output = capsys.readouterr().out

    assert "wo_graph" in output
    assert "full_rgcn" in output


def _memory_stream_task(task_id: str, query: str, memory_items: list[dict[str, object]]) -> MemoryTaskInput:
    return cast(
        MemoryTaskInput,
        cast(
            object,
            {
                "task_id": task_id,
                "query": query,
                "memory_items": memory_items,
            },
        ),
    )


def _memory_stream_artifact(task_input: MemoryTaskInput) -> ImportanceArtifact:
    return cast(
        ImportanceArtifact,
        cast(
            object,
            {
                "schema_version": 1,
                "method": "memory_stream",
                "tasks": [
                    {
                        "task_id": task_input["task_id"],
                        "content_digest": importance_content_digest(task_input),
                        "scores": {"m0": 10},
                    }
                ],
            },
        ),
    )


def test_memory_stream_manifest_caps_test_split_and_stage_config_importance_path(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config = load_experiment_config()
    config["methods"] = ["memory_stream"]
    importance_path = tmp_path / "dev.first_1000.importance.json"
    task_input = _memory_stream_task(
        "hotpot_ms_1",
        "Which river runs through Paris?",
        [
            {
                "id": "m0",
                "node_type": "document_sentence",
                "text": "The Eiffel Tower is in Paris.",
                "source": "Eiffel Tower",
                "sentence_id": 0,
                "position": 0,
            }
        ],
    )
    write_json(importance_path, _memory_stream_artifact(task_input))
    config["memory_stream_importance_path"] = str(importance_path)

    with caplog.at_level(logging.WARNING):
        manifest = initialize_experiment(
            "memory-stream-cap",
            config=config,
            run_root=tmp_path,
            profile="full",
            methods=["memory_stream"],
            force=True,
        )

    assert manifest["effective_config"]["splits"]["test"]["max_examples"] == 1
    assert any("Memory Stream" in record.message and "capped" in record.message for record in caplog.records)

    retrieve_path = Path(manifest["stage_configs"]["retrieve"]["memory_stream"])
    retrieve_config = read_json(retrieve_path)
    assert retrieve_config["io"]["importance"] == str(importance_path)
    assert retrieve_config["job"]["capped_test_count"] == 1
    assert retrieve_config["job"]["scoring"] == {
        "relevance_weight": 1.0,
        "recency_weight": 0.0,
        "importance_weight": 0.01,
        "recency_decay": 0.99,
    }

    workflow = get_workflow("memory_stream")
    assert workflow.identifier is WorkflowId.TUNED_STATELESS_RETRIEVAL
    assert [step.stage for step in workflow.steps] == [
        StageId.PREPARE,
        StageId.GRAPHS,
        StageId.TUNE,
        StageId.RETRIEVE,
        StageId.EVALUATE,
        StageId.AGGREGATE,
    ]

    tune_commands = build_stage_plan(
        manifest,
        stages=["tune"],
        methods=["memory_stream"],
    )
    assert len(tune_commands) == 1
    tune_command = tune_commands[0]
    assert tune_command.argv[1] == "scripts/tune_memory_stream.py"
    assert tune_command.argv[tune_command.argv.index("--importance") + 1] == str(
        importance_path
    )
    assert tune_command.argv[tune_command.argv.index("--grid_config") + 1] == (
        "configs/search_spaces/memory_stream.json"
    )

    selected_config_path = Path(
        manifest["artifacts"]["tuned"]["memory_stream"]
    )
    write_json(
        selected_config_path,
        {
            "relevance_weight": 1.0,
            "recency_weight": 0.0,
            "importance_weight": 0.01,
            "recency_decay": 0.99,
        },
    )
    write_json(
        selected_config_path.with_name(
            f"{selected_config_path.stem}.run_summary.json"
        ),
        {
            "script": "tune_memory_stream.py",
            "status": "success",
            "inputs": {
                "tasks": manifest["artifacts"]["inputs"]["dev"]["input"],
                "labels": manifest["artifacts"]["inputs"]["dev"]["labels"],
                "graphs": manifest["artifacts"]["graphs"]["dev"],
                "importance": str(importance_path),
                "grid_config": "configs/search_spaces/memory_stream.json",
            },
            "outputs": {"selected_config": str(selected_config_path)},
            "effective_config": {
                "top_k": manifest["effective_config"]["top_k"],
                "grid_config": "configs/search_spaces/memory_stream.json",
            },
        },
    )
    tune_status = next(
        row
        for row in inspect_experiment_status(manifest)
        if row["stage"] == "tune" and row.get("method") == "memory_stream"
    )
    assert tune_status["state"] == "complete"

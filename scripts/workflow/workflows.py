from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from graph_memory.retrieval_registry import get_method_spec, get_methods_requiring_dense_encoder
from graph_memory.training_config import device_from_training_config
from scripts.workflow.types import ArtifactRole, ChangeDimension, StageCommand, StageId, WorkflowId, WorkflowSpec, WorkflowStepSpec


_PREPARE = WorkflowStepSpec(
    stage=StageId.PREPARE,
    inputs=(),
    outputs=(ArtifactRole.INPUTS, ArtifactRole.LABELS),
    command_adapter="scripts/prepare_hotpotqa.py",
)
_GRAPHS = WorkflowStepSpec(
    stage=StageId.GRAPHS,
    inputs=(ArtifactRole.INPUTS,),
    outputs=(ArtifactRole.GRAPHS,),
    command_adapter="scripts/build_graphs.py",
)
_RETRIEVE = WorkflowStepSpec(
    stage=StageId.RETRIEVE,
    inputs=(ArtifactRole.INPUTS,),
    outputs=(ArtifactRole.PREDICTIONS,),
    command_adapter="scripts/run_retrieval.py",
)
_EVALUATE = WorkflowStepSpec(
    stage=StageId.EVALUATE,
    inputs=(ArtifactRole.PREDICTIONS, ArtifactRole.LABELS, ArtifactRole.GRAPHS),
    outputs=(ArtifactRole.METRICS,),
    command_adapter="scripts/evaluate_retrieval.py",
)
_AGGREGATE = WorkflowStepSpec(
    stage=StageId.AGGREGATE,
    inputs=(ArtifactRole.METRICS,),
    outputs=(ArtifactRole.MAIN_TABLE,),
    command_adapter="scripts/aggregate_tables.py",
)


STATELESS_RETRIEVAL_WORKFLOW = WorkflowSpec(
    identifier=WorkflowId.STATELESS_RETRIEVAL,
    steps=(_PREPARE, _GRAPHS, _RETRIEVE, _EVALUATE, _AGGREGATE),
)

GRAPH_RERANK_WORKFLOW = WorkflowSpec(
    identifier=WorkflowId.GRAPH_RERANK,
    steps=(
        _PREPARE,
        _GRAPHS,
        WorkflowStepSpec(
            stage=StageId.TUNE,
            inputs=(ArtifactRole.INPUTS, ArtifactRole.LABELS, ArtifactRole.GRAPHS),
            outputs=(ArtifactRole.TUNED_CONFIG,),
            command_adapter="scripts/tune_graph_rerank.py",
        ),
        _RETRIEVE,
        _EVALUATE,
        _AGGREGATE,
    ),
)

_TRAIN_INVALIDATIONS = frozenset(
    {
        ChangeDimension.PAIR_SAMPLING,
        ChangeDimension.MODEL_STRUCTURE,
        ChangeDimension.MODEL_GRAPH_VIEW,
    }
)

RGCN_WORKFLOW = WorkflowSpec(
    identifier=WorkflowId.RGCN_TRAINABLE_RETRIEVAL,
    steps=(
        _PREPARE,
        _GRAPHS,
        WorkflowStepSpec(
            stage=StageId.PAIRS,
            inputs=(ArtifactRole.INPUTS, ArtifactRole.LABELS, ArtifactRole.GRAPHS),
            outputs=(ArtifactRole.TRAIN_PAIRS,),
            invalidated_by=frozenset({ChangeDimension.PAIR_SAMPLING}),
            command_adapter="scripts/build_train_pairs.py",
        ),
        WorkflowStepSpec(
            stage=StageId.TRAIN,
            inputs=(ArtifactRole.INPUTS, ArtifactRole.LABELS, ArtifactRole.GRAPHS, ArtifactRole.TRAIN_PAIRS),
            outputs=(ArtifactRole.CHECKPOINT,),
            invalidated_by=_TRAIN_INVALIDATIONS,
            command_adapter="scripts/train_graph_retriever.py",
        ),
        WorkflowStepSpec(
            stage=StageId.RETRIEVE,
            inputs=(ArtifactRole.INPUTS, ArtifactRole.GRAPHS, ArtifactRole.CHECKPOINT),
            outputs=(ArtifactRole.PREDICTIONS,),
            invalidated_by=_TRAIN_INVALIDATIONS,
            command_adapter="scripts/run_retrieval.py",
        ),
        WorkflowStepSpec(
            stage=StageId.EVALUATE,
            inputs=(ArtifactRole.PREDICTIONS, ArtifactRole.LABELS, ArtifactRole.GRAPHS),
            outputs=(ArtifactRole.METRICS,),
            invalidated_by=_TRAIN_INVALIDATIONS,
            command_adapter="scripts/evaluate_retrieval.py",
        ),
        _AGGREGATE,
    ),
)


def build_prepare_commands(manifest: dict[str, Any]) -> list[StageCommand]:
    commands: list[StageCommand] = []
    config = manifest["effective_config"]
    for split in ("train", "dev", "test"):
        split_config = config["splits"][split]
        raw_source = split_config["source"]
        artifacts = manifest["artifacts"]["inputs"][split]
        commands.append(
            StageCommand(
                stage=StageId.PREPARE,
                split=split,
                argv=[
                    sys.executable,
                    "scripts/prepare_hotpotqa.py",
                    "--input",
                    str(config["raw"][raw_source]),
                    "--output_input",
                    artifacts["input"],
                    "--output_labels",
                    artifacts["labels"],
                    "--output_combined",
                    artifacts["combined"],
                    "--max_examples",
                    str(split_config["max_examples"]),
                    "--seed",
                    str(split_config["seed"]),
                    "--offset",
                    str(split_config["offset"]),
                ],
            )
        )
    return commands


def build_graph_commands(manifest: dict[str, Any]) -> list[StageCommand]:
    commands: list[StageCommand] = []
    graph_config = manifest["effective_config"]["graph"]
    for split in ("train", "dev", "test"):
        argv = [
            sys.executable,
            "scripts/build_graphs.py",
            "--input",
            manifest["artifacts"]["inputs"][split]["input"],
            "--output",
            manifest["artifacts"]["graphs"][split],
            "--max_query_overlap",
            str(graph_config["max_query_overlap"]),
            "--max_entity_neighbors",
            str(graph_config["max_entity_neighbors"]),
            "--max_bridge_edges",
            str(graph_config["max_bridge_edges"]),
        ]
        if graph_config.get("use_spacy"):
            argv.append("--use_spacy")
        commands.append(StageCommand(stage=StageId.GRAPHS, split=split, argv=argv))
    return commands


def build_pair_commands(manifest: dict[str, Any], methods: Sequence[str]) -> list[StageCommand]:
    return [
        StageCommand(
            stage=StageId.PAIRS,
            method=method,
            argv=[
                sys.executable,
                "scripts/build_train_pairs.py",
                "--tasks",
                manifest["artifacts"]["inputs"]["train"]["input"],
                "--labels",
                manifest["artifacts"]["inputs"]["train"]["labels"],
                "--graphs",
                manifest["artifacts"]["graphs"]["train"],
                "--output",
                manifest["artifacts"]["learned"][method]["train_pairs"],
                "--config",
                manifest["artifacts"]["learned"][method]["effective_training_config"],
            ],
        )
        for method in methods
    ]


def build_train_commands(manifest: dict[str, Any], methods: Sequence[str]) -> list[StageCommand]:
    commands: list[StageCommand] = []
    for method in methods:
        learned = manifest["artifacts"]["learned"][method]
        commands.append(
            StageCommand(
                stage=StageId.TRAIN,
                method=method,
                argv=[
                    sys.executable,
                    "scripts/train_graph_retriever.py",
                    "--train_tasks",
                    manifest["artifacts"]["inputs"]["train"]["input"],
                    "--train_labels",
                    manifest["artifacts"]["inputs"]["train"]["labels"],
                    "--train_graphs",
                    manifest["artifacts"]["graphs"]["train"],
                    "--train_pairs",
                    learned["train_pairs"],
                    "--dev_tasks",
                    manifest["artifacts"]["inputs"]["dev"]["input"],
                    "--dev_labels",
                    manifest["artifacts"]["inputs"]["dev"]["labels"],
                    "--dev_graphs",
                    manifest["artifacts"]["graphs"]["dev"],
                    "--output_dir",
                    learned["training_output_dir"],
                    "--config",
                    learned["effective_training_config"],
                ],
            )
        )
    return commands


def build_tune_commands(manifest: dict[str, Any], methods: Sequence[str]) -> list[StageCommand]:
    commands: list[StageCommand] = []
    for method in methods:
        argv = [
            sys.executable,
            "scripts/tune_graph_rerank.py",
            "--method",
            method,
            "--tasks",
            manifest["artifacts"]["inputs"]["dev"]["input"],
            "--labels",
            manifest["artifacts"]["inputs"]["dev"]["labels"],
            "--graphs",
            manifest["artifacts"]["graphs"]["dev"],
            "--output_config",
            manifest["artifacts"]["tuned"][method],
            "--top_k",
            str(manifest["effective_config"]["top_k"]),
            "--grid_config",
            str(manifest["effective_config"]["search_spaces"]["graph_rerank"]),
        ]
        _append_dense_args(argv, manifest)
        commands.append(StageCommand(stage=StageId.TUNE, method=method, argv=argv))
    return commands


def build_retrieve_commands(manifest: dict[str, Any], methods: Sequence[str]) -> list[StageCommand]:
    commands: list[StageCommand] = []
    dense_methods = set(get_methods_requiring_dense_encoder())
    for method in methods:
        spec = get_method_spec(method)
        argv = [
            sys.executable,
            "scripts/run_retrieval.py",
            "--method",
            method,
            "--tasks",
            manifest["artifacts"]["inputs"]["test"]["input"],
            "--output",
            manifest["artifacts"]["predictions"][method],
            "--top_k",
            str(manifest["effective_config"]["top_k"]),
        ]
        if spec.requires_graphs:
            argv.extend(["--graphs", manifest["artifacts"]["graphs"]["test"]])
        if spec.requires_graph_config:
            argv.extend(["--graph_config", manifest["artifacts"]["tuned"][method]])
        if spec.requires_checkpoint:
            learned = manifest["artifacts"]["learned"][method]
            argv.extend(
                [
                    "--checkpoint",
                    learned["best_checkpoint"],
                    "--device",
                    device_from_training_config(manifest["effective_config"]["training"][method]),
                ]
            )
        if method in dense_methods and not spec.requires_checkpoint:
            _append_dense_args(argv, manifest)
        commands.append(StageCommand(stage=StageId.RETRIEVE, method=method, argv=argv))
    return commands


def build_evaluate_commands(manifest: dict[str, Any], methods: Sequence[str]) -> list[StageCommand]:
    return [
        StageCommand(
            stage=StageId.EVALUATE,
            method=method,
            argv=[
                sys.executable,
                "scripts/evaluate_retrieval.py",
                "--pred",
                manifest["artifacts"]["predictions"][method],
                "--labels",
                manifest["artifacts"]["inputs"]["test"]["labels"],
                "--graphs",
                manifest["artifacts"]["graphs"]["test"],
                "--output",
                manifest["artifacts"]["metrics"][method],
                "--failure_cases_output",
                manifest["artifacts"]["failure_cases"][method],
                "--failure_case_limit",
                "50",
            ],
        )
        for method in methods
    ]


def build_aggregate_command(
    manifest: dict[str, Any],
    *,
    ablation_selections: Sequence[tuple[str, str]] = (),
) -> StageCommand:
    argv = [
        sys.executable,
        "scripts/aggregate_tables.py",
        "--input_dir",
        str(Path(manifest["paths"]["run_dir"]) / "metrics"),
        "--output_main",
        manifest["artifacts"]["tables"]["main"],
        "--output_path",
        manifest["artifacts"]["tables"]["path"],
        "--output_efficiency",
        manifest["artifacts"]["tables"]["efficiency"],
    ]
    index_path = manifest.get("paths", {}).get("ablation_metrics_index")
    table_path = manifest.get("artifacts", {}).get("tables", {}).get("ablation")
    if index_path is not None and table_path is not None and ablation_selections:
        argv.extend(["--ablation_index", index_path, "--output_ablation", table_path])
        for method, variant in ablation_selections:
            argv.extend(["--ablation_selection", f"{method}={variant}"])
    return StageCommand(stage=StageId.AGGREGATE, argv=argv)


def _append_dense_args(argv: list[str], manifest: dict[str, Any]) -> None:
    config = manifest["effective_config"]
    argv.extend(
        [
            "--encoder_model",
            str(config["dense_encoder"]),
            "--query_prefix",
            str(config["query_prefix"]),
            "--passage_prefix",
            str(config["passage_prefix"]),
        ]
    )

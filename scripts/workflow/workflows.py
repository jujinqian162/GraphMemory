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
            command_adapter="scripts/train_method.py",
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

DENSE_FT_WORKFLOW = WorkflowSpec(
    identifier=WorkflowId.DENSE_FINETUNE_RETRIEVAL,
    steps=(
        _PREPARE,
        _GRAPHS,
        WorkflowStepSpec(
            stage=StageId.PAIRS,
            inputs=(ArtifactRole.INPUTS, ArtifactRole.LABELS, ArtifactRole.GRAPHS),
            outputs=(ArtifactRole.TRAIN_PAIRS,),
            command_adapter="scripts/build_train_pairs.py",
        ),
        WorkflowStepSpec(
            stage=StageId.TRAIN,
            inputs=(ArtifactRole.INPUTS, ArtifactRole.LABELS, ArtifactRole.TRAIN_PAIRS),
            outputs=(ArtifactRole.CHECKPOINT,),
            command_adapter="scripts/train_method.py",
        ),
        WorkflowStepSpec(
            stage=StageId.RETRIEVE,
            inputs=(ArtifactRole.INPUTS, ArtifactRole.CHECKPOINT),
            outputs=(ArtifactRole.PREDICTIONS,),
            command_adapter="scripts/run_retrieval.py",
        ),
        _EVALUATE,
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
    commands: list[StageCommand] = []
    for method in methods:
        projection = _stage_config_projection(manifest, StageId.PAIRS, method)
        if projection is not None:
            io = projection["io"]
            argv = [
                sys.executable,
                "scripts/build_train_pairs.py",
                "--tasks",
                str(io["tasks"]),
                "--labels",
                str(io["labels"]),
                "--graphs",
                str(io["graphs"]),
                "--output",
                str(io["output"]),
            ]
            if io.get("config") is not None:
                argv.extend(["--config", str(io["config"])])
        else:
            argv = [
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
            ]
        commands.append(StageCommand(stage=StageId.PAIRS, method=method, argv=argv))
    return commands


def build_train_commands(manifest: dict[str, Any], methods: Sequence[str]) -> list[StageCommand]:
    commands: list[StageCommand] = []
    for method in methods:
        projection = _stage_config_projection(manifest, StageId.TRAIN, method)
        if projection is not None:
            io = projection["io"]
            argv = [
                sys.executable,
                "scripts/train_method.py",
                "--method",
                method,
                "--train_tasks",
                str(io["train_tasks"]),
                "--train_pairs",
                str(io["train_pairs"]),
                "--dev_tasks",
                str(io["dev_tasks"]),
                "--dev_labels",
                str(io["dev_labels"]),
                "--output_dir",
                str(io["output_dir"]),
            ]
            if io.get("train_labels") is not None:
                argv.extend(["--train_labels", str(io["train_labels"])])
            if io.get("train_graphs") is not None:
                argv.extend(["--train_graphs", str(io["train_graphs"])])
            if io.get("dev_graphs") is not None:
                argv.extend(["--dev_graphs", str(io["dev_graphs"])])
            if io.get("model_dir") is not None:
                argv.extend(["--model_dir", str(io["model_dir"])])
            if io.get("config") is not None:
                argv.extend(["--config", str(io["config"])])
        else:
            learned = manifest["artifacts"]["learned"][method]
            argv = [
                sys.executable,
                "scripts/train_method.py",
                "--method",
                method,
                "--train_tasks",
                manifest["artifacts"]["inputs"]["train"]["input"],
                "--train_labels",
                manifest["artifacts"]["inputs"]["train"]["labels"],
                "--train_pairs",
                learned["train_pairs"],
                "--dev_tasks",
                manifest["artifacts"]["inputs"]["dev"]["input"],
                "--dev_labels",
                manifest["artifacts"]["inputs"]["dev"]["labels"],
                "--output_dir",
                learned["training_output_dir"],
                "--config",
                learned["effective_training_config"],
            ]
            if method == "dense_ft":
                argv.extend(["--model_dir", learned["best_checkpoint"]])
            else:
                argv.extend(
                    [
                        "--train_graphs",
                        manifest["artifacts"]["graphs"]["train"],
                        "--dev_graphs",
                        manifest["artifacts"]["graphs"]["dev"],
                    ]
                )
        commands.append(
            StageCommand(
                stage=StageId.TRAIN,
                method=method,
                argv=argv,
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
        projection = _stage_config_projection(manifest, StageId.RETRIEVE, method)
        if projection is not None:
            argv = _retrieve_argv_from_projection(projection)
        else:
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
    commands: list[StageCommand] = []
    for method in methods:
        projection = _stage_config_projection(manifest, StageId.EVALUATE, method)
        if projection is not None:
            io = projection["io"]
            argv = [
                sys.executable,
                "scripts/evaluate_retrieval.py",
                "--pred",
                str(io["predictions"]),
                "--labels",
                str(io["labels"]),
                "--graphs",
                str(io["graphs"]),
                "--output",
                str(io["output"]),
            ]
            if io.get("failure_cases_output") is not None:
                argv.extend(["--failure_cases_output", str(io["failure_cases_output"])])
            argv.extend(["--failure_case_limit", str(projection["failure_case_limit"])])
        else:
            argv = [
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
            ]
        commands.append(StageCommand(stage=StageId.EVALUATE, method=method, argv=argv))
    return commands


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


def _stage_config_projection(manifest: dict[str, Any], stage: StageId, method: str) -> dict[str, Any] | None:
    value = manifest.get("stage_configs", {}).get(stage.value, {}).get(method)
    return value if isinstance(value, dict) else None


def _retrieve_argv_from_projection(projection: dict[str, Any]) -> list[str]:
    io = projection["io"]
    job = projection["job"]
    argv = [
        sys.executable,
        "scripts/run_retrieval.py",
        "--method",
        str(job["method"]),
        "--tasks",
        str(io["tasks"]),
        "--output",
        str(io["output"]),
        "--top_k",
        str(job["top_k"]),
    ]
    if io.get("graphs") is not None:
        argv.extend(["--graphs", str(io["graphs"])])
    if io.get("graph_config") is not None:
        argv.extend(["--graph_config", str(io["graph_config"])])
    if job.get("checkpoint") is not None:
        argv.extend(["--checkpoint", str(job["checkpoint"]), "--device", str(job["device"])])

    encoder = _retrieval_projection_encoder(job)
    if encoder is not None:
        argv.extend(
            [
                "--encoder_model",
                str(encoder["model_name"]),
                "--query_prefix",
                str(encoder["query_prefix"]),
                "--passage_prefix",
                str(encoder["passage_prefix"]),
            ]
        )
    return argv


def _retrieval_projection_encoder(job: dict[str, Any]) -> dict[str, Any] | None:
    encoder = job.get("encoder")
    if isinstance(encoder, dict):
        return encoder
    seed = job.get("seed")
    if isinstance(seed, dict) and isinstance(seed.get("encoder"), dict):
        return seed["encoder"]
    return None

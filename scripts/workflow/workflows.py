from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from graph_memory.registry import Registry
from graph_memory.registry.methods import TuningKind
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

TUNED_STATELESS_RETRIEVAL_WORKFLOW = WorkflowSpec(
    identifier=WorkflowId.TUNED_STATELESS_RETRIEVAL,
    steps=(
        _PREPARE,
        _GRAPHS,
        WorkflowStepSpec(
            stage=StageId.TUNE,
            inputs=(ArtifactRole.INPUTS, ArtifactRole.LABELS, ArtifactRole.GRAPHS),
            outputs=(ArtifactRole.TUNED_CONFIG,),
            command_adapter="scripts/tune_memory_stream.py",
        ),
        _RETRIEVE,
        _EVALUATE,
        _AGGREGATE,
    ),
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
    return [
        _stage_config_command(
            manifest,
            stage=StageId.PAIRS,
            method=method,
            script="scripts/build_train_pairs.py",
        )
        for method in methods
    ]


def build_train_commands(manifest: dict[str, Any], methods: Sequence[str]) -> list[StageCommand]:
    return [
        _stage_config_command(
            manifest,
            stage=StageId.TRAIN,
            method=method,
            script="scripts/train_method.py",
        )
        for method in methods
    ]


def build_tune_commands(manifest: dict[str, Any], methods: Sequence[str]) -> list[StageCommand]:
    commands: list[StageCommand] = []
    for method in methods:
        definition = Registry.methods.get(method)
        if definition.tuning is TuningKind.GRAPH_RERANK:
            argv = _graph_rerank_tune_argv(manifest, method)
        elif definition.tuning is TuningKind.MEMORY_STREAM:
            argv = _memory_stream_tune_argv(manifest, method)
        else:
            raise ValueError(f"Method does not register a tuning adapter: {method}")
        _append_dense_args(argv, manifest)
        commands.append(StageCommand(stage=StageId.TUNE, method=method, argv=argv))
    return commands


def _graph_rerank_tune_argv(
    manifest: dict[str, Any],
    method: str,
) -> list[str]:
    return [
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


def _memory_stream_tune_argv(
    manifest: dict[str, Any],
    method: str,
) -> list[str]:
    return [
        sys.executable,
        "scripts/tune_memory_stream.py",
        "--tasks",
        manifest["artifacts"]["inputs"]["dev"]["input"],
        "--labels",
        manifest["artifacts"]["inputs"]["dev"]["labels"],
        "--graphs",
        manifest["artifacts"]["graphs"]["dev"],
        "--importance",
        str(manifest["effective_config"]["memory_stream_importance_path"]),
        "--output_config",
        manifest["artifacts"]["tuned"][method],
        "--top_k",
        str(manifest["effective_config"]["top_k"]),
        "--grid_config",
        str(manifest["effective_config"]["search_spaces"]["memory_stream"]),
    ]


def build_retrieve_commands(manifest: dict[str, Any], methods: Sequence[str]) -> list[StageCommand]:
    return [
        _stage_config_command(
            manifest,
            stage=StageId.RETRIEVE,
            method=method,
            script="scripts/run_retrieval.py",
        )
        for method in methods
    ]


def build_evaluate_commands(manifest: dict[str, Any], methods: Sequence[str]) -> list[StageCommand]:
    return [
        _stage_config_command(
            manifest,
            stage=StageId.EVALUATE,
            method=method,
            script="scripts/evaluate_retrieval.py",
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


def _stage_config_command(
    manifest: dict[str, Any],
    *,
    stage: StageId,
    method: str,
    script: str,
) -> StageCommand:
    stage_configs = manifest.get("stage_configs")
    if not isinstance(stage_configs, dict):
        raise ValueError("Manifest requires stage_configs.")
    stage_mapping = stage_configs.get(stage.value)
    if not isinstance(stage_mapping, dict):
        raise ValueError(f"Manifest requires stage_configs.{stage.value}.")
    config_path = stage_mapping.get(method)
    if not isinstance(config_path, str) or not config_path:
        raise ValueError(f"Manifest requires stage config for stage={stage.value} method={method}.")
    return StageCommand(
        stage=stage,
        method=method,
        argv=[sys.executable, script, "--config", config_path],
    )

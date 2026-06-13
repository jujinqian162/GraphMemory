from __future__ import annotations

import subprocess
from collections.abc import Iterable, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from graph_memory.registry import Registry
from scripts.workflow.registry import get_workflow, validate_workflow_registry
from scripts.workflow.types import StageCommand, StageId, VariantSpec, WorkflowSpec
from scripts.workflow.workflows import (
    build_aggregate_command,
    build_evaluate_commands,
    build_graph_commands,
    build_importance_commands,
    build_pair_commands,
    build_prepare_commands,
    build_retrieve_commands,
    build_train_commands,
    build_tune_commands,
)


def earliest_invalidated_stage(workflow: WorkflowSpec, variant: VariantSpec) -> StageId | None:
    """Return the first workflow step invalidated by a variant."""

    if variant.baseline_alias:
        return None
    for step in workflow.steps:
        if step.invalidated_by & variant.changed_dimensions:
            return step.stage
    dimensions = ", ".join(sorted(dimension.value for dimension in variant.changed_dimensions))
    raise ValueError(
        f"Workflow={workflow.identifier.value} has no invalidation boundary for "
        f"variant={variant.identifier.value} dimensions=[{dimensions}]"
    )


def build_stage_plan(
    manifest: dict[str, Any],
    *,
    stages: Sequence[str] | None = None,
    methods: Sequence[str] | None = None,
    from_stage: str | None = None,
    to_stage: str | None = None,
    variants: Sequence[str] | None = None,
    ablations_only: bool = False,
) -> list[StageCommand]:
    """Build ordinary and variant-qualified low-level commands."""

    validate_workflow_registry()
    enabled = bool(manifest.get("effective_config", {}).get("enable_ablation", False))
    selected_methods = list(methods) if methods is not None else list(manifest["selected_methods"])
    _validate_methods(selected_methods)
    if not enabled:
        if variants or ablations_only:
            raise ValueError("Ablation filters require an ablation-enabled manifest.")
        return _ordinary_stage_plan(
            manifest,
            stages=stages,
            methods=selected_methods,
            from_stage=from_stage,
            to_stage=to_stage,
        )

    selected_stages = _select_stages(
        stages,
        from_stage=from_stage,
        to_stage=to_stage,
        methods=selected_methods,
    )
    selected_units = _selected_variant_units(manifest, selected_methods, variants)
    _validate_variant_dependencies(selected_stages, selected_units)
    if ablations_only:
        _validate_ablation_only_dependencies(manifest, selected_methods, selected_units)

    commands: list[StageCommand] = []
    if not ablations_only:
        commands.extend(
            _without_aggregate(
                _ordinary_stage_plan(
                    manifest,
                    stages=stages,
                    methods=selected_methods,
                    from_stage=from_stage,
                    to_stage=to_stage,
                )
            )
        )
    else:
        if StageId.PREPARE.value in selected_stages:
            commands.extend(build_prepare_commands(manifest))
        if StageId.GRAPHS.value in selected_stages:
            commands.extend(build_graph_commands(manifest))
        shared_pair_methods = sorted(
            {
                method
                for method, record in selected_units
                if record["invalidated_from"] == StageId.TRAIN.value
            }
        )
        if StageId.PAIRS.value in selected_stages and shared_pair_methods:
            commands.extend(build_pair_commands(manifest, shared_pair_methods))

    for method, record in selected_units:
        commands.extend(_variant_commands(manifest, method, record, selected_stages))
    if StageId.AGGREGATE.value in selected_stages:
        commands.append(
            build_aggregate_command(
                manifest,
                ablation_selections=_aggregate_ablation_selections(manifest, selected_units),
            )
        )
    return commands


def _ordinary_stage_plan(
    manifest: dict[str, Any],
    *,
    stages: Sequence[str] | None,
    methods: Sequence[str],
    from_stage: str | None,
    to_stage: str | None,
) -> list[StageCommand]:
    selected_stages = _select_stages(stages, from_stage=from_stage, to_stage=to_stage, methods=methods)
    _validate_stage_dependencies(manifest, selected_stages, methods)
    commands: list[StageCommand] = []
    if StageId.PREPARE.value in selected_stages:
        commands.extend(build_prepare_commands(manifest))
    if StageId.GRAPHS.value in selected_stages:
        commands.extend(build_graph_commands(manifest))
    if StageId.IMPORTANCE.value in selected_stages:
        commands.extend(build_importance_commands(manifest, _methods_with_stage(methods, StageId.IMPORTANCE)))
    if StageId.PAIRS.value in selected_stages:
        commands.extend(build_pair_commands(manifest, _methods_with_stage(methods, StageId.PAIRS)))
    if StageId.TUNE.value in selected_stages:
        commands.extend(build_tune_commands(manifest, _methods_with_stage(methods, StageId.TUNE)))
    if StageId.TRAIN.value in selected_stages:
        commands.extend(build_train_commands(manifest, _methods_with_stage(methods, StageId.TRAIN)))
    if StageId.RETRIEVE.value in selected_stages:
        commands.extend(build_retrieve_commands(manifest, methods))
    if StageId.EVALUATE.value in selected_stages:
        commands.extend(build_evaluate_commands(manifest, methods))
    if StageId.AGGREGATE.value in selected_stages:
        commands.append(build_aggregate_command(manifest))
    return commands


def _selected_variant_units(
    manifest: dict[str, Any],
    selected_methods: Sequence[str],
    variants: Sequence[str] | None,
) -> list[tuple[str, dict[str, Any]]]:
    requested = set(variants or ())
    available: set[str] = set()
    selected: list[tuple[str, dict[str, Any]]] = []
    for method in selected_methods:
        records = manifest.get("artifacts", {}).get("ablations", {}).get(method, {})
        for variant, record in records.items():
            available.add(variant)
            if record["baseline_alias"]:
                continue
            if requested and variant not in requested:
                continue
            selected.append((method, record))
    unknown = sorted(requested - available)
    if unknown:
        allowed = ", ".join(sorted(available))
        raise ValueError(f"Unknown selected variants={unknown}; allowed values: {allowed}")
    return selected


def _validate_variant_dependencies(
    selected_stages: Sequence[str],
    units: Sequence[tuple[str, dict[str, Any]]],
) -> None:
    if StageId.TRAIN.value in selected_stages and StageId.PAIRS.value not in selected_stages:
        missing_pairs = [
            record["artifacts"]["train_pairs"]
            for _, record in units
            if not Path(record["artifacts"]["train_pairs"]).exists()
        ]
        if missing_pairs:
            raise ValueError(f"Missing train pairs for train stage: {', '.join(missing_pairs)}")
    if StageId.RETRIEVE.value in selected_stages and StageId.TRAIN.value not in selected_stages:
        missing_checkpoints = [
            record["artifacts"]["checkpoint"]
            for _, record in units
            if not Path(record["artifacts"]["checkpoint"]).exists()
        ]
        if missing_checkpoints:
            raise ValueError(f"Missing trainable checkpoints for retrieve stage: {', '.join(missing_checkpoints)}")


def _validate_ablation_only_dependencies(
    manifest: dict[str, Any],
    selected_methods: Sequence[str],
    units: Sequence[tuple[str, dict[str, Any]]],
) -> None:
    if not units:
        methods = ", ".join(selected_methods)
        raise ValueError(
            "No executable ablation variants are registered for the selected methods: "
            f"{methods}. Select a method with a registered non-baseline ablation variant."
        )
    missing_baselines = [
        record["artifacts"]["metrics"]
        for method in sorted({method for method, _ in units})
        for record in manifest["artifacts"]["ablations"][method].values()
        if record["baseline_alias"] and not Path(record["artifacts"]["metrics"]).exists()
    ]
    if missing_baselines:
        raise ValueError(
            "Ablation-only execution requires ordinary baseline metrics because baseline aliases are included "
            "in the ablation table. Run the main experiment without --ablations-only first. "
            f"Missing: {', '.join(missing_baselines)}"
        )


def _aggregate_ablation_selections(
    manifest: dict[str, Any],
    units: Sequence[tuple[str, dict[str, Any]]],
) -> list[tuple[str, str]]:
    requested = {(method, str(record["variant"])) for method, record in units}
    methods = {method for method, _ in units}
    return [
        (method, variant)
        for method in sorted(methods)
        for variant, record in manifest["artifacts"]["ablations"][method].items()
        if record["baseline_alias"] or (method, variant) in requested
    ]


def _variant_commands(
    manifest: dict[str, Any],
    method: str,
    record: dict[str, Any],
    selected_stages: Sequence[str],
) -> list[StageCommand]:
    variant_manifest = _materialize_variant_manifest(manifest, method, record)
    variant = str(record["variant"])
    boundary = StageId(record["invalidated_from"])
    stage_index = {stage: index for index, stage in enumerate(StageId)}
    commands: list[StageCommand] = []
    if StageId.PAIRS.value in selected_stages and stage_index[boundary] <= stage_index[StageId.PAIRS]:
        commands.extend(_qualify(build_pair_commands(variant_manifest, [method]), variant))
    if StageId.TRAIN.value in selected_stages and stage_index[boundary] <= stage_index[StageId.TRAIN]:
        commands.extend(_qualify(build_train_commands(variant_manifest, [method]), variant))
    if StageId.RETRIEVE.value in selected_stages:
        commands.extend(_qualify(build_retrieve_commands(variant_manifest, [method]), variant))
    if StageId.EVALUATE.value in selected_stages:
        commands.extend(_qualify(build_evaluate_commands(variant_manifest, [method]), variant))
    return commands


def _materialize_variant_manifest(manifest: dict[str, Any], method: str, record: dict[str, Any]) -> dict[str, Any]:
    resolved = deepcopy(manifest)
    artifacts = record["artifacts"]
    learned = resolved["artifacts"]["learned"][method]
    learned.update(
        {
            "train_pairs": artifacts["train_pairs"],
            "train_pair_summary": artifacts["train_pair_summary"],
            "train_pair_run_summary": artifacts["train_pair_run_summary"],
            "effective_method_config": artifacts["effective_method_config"],
            "training_output_dir": str(Path(artifacts["checkpoint"]).parents[1]),
            "train_metrics": artifacts["train_metrics"],
            "train_run_summary": artifacts["train_run_summary"],
            "best_checkpoint": artifacts["checkpoint"],
        }
    )
    resolved["artifacts"]["predictions"][method] = artifacts["predictions"]
    resolved["artifacts"]["metrics"][method] = artifacts["metrics"]
    resolved["artifacts"]["failure_cases"][method] = artifacts["failure_cases"]
    stage_configs = record.get("stage_configs")
    if not isinstance(stage_configs, dict):
        raise ValueError(f"Variant requires stage_configs: method={method} variant={record['variant']}")
    for stage, config_path in stage_configs.items():
        resolved["stage_configs"].setdefault(stage, {})[method] = config_path
    return resolved


def required_stages_for_methods(methods: Sequence[str]) -> list[str]:
    selected_methods = list(methods)
    _validate_methods(selected_methods)
    required = {
        step.stage.value
        for method in selected_methods
        for step in get_workflow(method).steps
    }
    return [stage.value for stage in StageId if stage.value in required]


def _select_stages(
    stages: Sequence[str] | None,
    *,
    from_stage: str | None,
    to_stage: str | None,
    methods: Sequence[str] | None = None,
) -> list[str]:
    selected: list[str]
    if stages is not None:
        if from_stage is not None or to_stage is not None:
            raise ValueError("Use either explicit stages or a stage range, not both.")
        selected = list(stages)
    else:
        workflow = required_stages_for_methods(methods) if methods is not None else [stage.value for stage in StageId]
        if from_stage is None and to_stage is None:
            selected = workflow
        else:
            start_stage = from_stage or workflow[0]
            end_stage = to_stage or workflow[-1]
            start_index = _workflow_stage_index(workflow, start_stage)
            end_index = _workflow_stage_index(workflow, end_stage)
            if start_index > end_index:
                raise ValueError(f"Stage range start={start_stage} comes after end={end_stage}.")
            selected = workflow[start_index : end_index + 1]
    unknown = [stage for stage in selected if stage not in {member.value for member in StageId}]
    if unknown:
        allowed = ", ".join(stage.value for stage in StageId)
        raise ValueError(f"Unsupported stage: {', '.join(unknown)}; allowed values: {allowed}")
    return selected


def _workflow_stage_index(workflow: Sequence[str], stage: str) -> int:
    allowed = {member.value for member in StageId}
    if stage not in allowed:
        raise ValueError(f"Unsupported stage: {stage}; allowed values: {', '.join(sorted(allowed))}")
    try:
        return list(workflow).index(stage)
    except ValueError as error:
        raise ValueError(f"Stage {stage} is not in the available workflow stages: {', '.join(workflow)}") from error


def _validate_methods(methods: Iterable[str]) -> None:
    supported_methods = {method.value for method in Registry.methods.list_ids()}
    unsupported = [method for method in methods if method not in supported_methods]
    if unsupported:
        raise ValueError(f"Unsupported method: {', '.join(unsupported)}")


def _validate_stage_dependencies(
    manifest: dict[str, Any],
    selected_stages: Sequence[str],
    selected_methods: Sequence[str],
) -> None:
    if StageId.RETRIEVE.value in selected_stages and StageId.TUNE.value not in selected_stages:
        missing_tuned = [
            method
            for method in _methods_with_stage(selected_methods, StageId.TUNE)
            if not Path(manifest["artifacts"]["tuned"][method]).exists()
        ]
        if missing_tuned:
            missing_paths = ", ".join(manifest["artifacts"]["tuned"][method] for method in missing_tuned)
            raise ValueError(
                "Graph rerank retrieval requires tuned graph config. "
                "Add the tune stage or run tune first. "
                f"Missing: {missing_paths}"
            )

    trainable_methods = _methods_with_stage(selected_methods, StageId.TRAIN)
    if StageId.TRAIN.value in selected_stages and StageId.PAIRS.value not in selected_stages:
        missing_pairs = [
            method
            for method in trainable_methods
            if not Path(manifest["artifacts"]["learned"][method]["train_pairs"]).exists()
        ]
        if missing_pairs:
            missing_paths = ", ".join(manifest["artifacts"]["learned"][method]["train_pairs"] for method in missing_pairs)
            raise ValueError(
                "Trainable training requires train pairs. "
                "Add the pairs stage or run pairs first. "
                f"Missing: {missing_paths}"
            )

    if StageId.RETRIEVE.value in selected_stages and StageId.TRAIN.value not in selected_stages:
        missing_checkpoints = [
            method
            for method in trainable_methods
            if not Path(manifest["artifacts"]["learned"][method]["best_checkpoint"]).exists()
        ]
        if missing_checkpoints:
            missing_paths = ", ".join(
                manifest["artifacts"]["learned"][method]["best_checkpoint"] for method in missing_checkpoints
            )
            raise ValueError(
                "Trainable retrieval requires a trained checkpoint. "
                "Add the train stage or run train first. "
                f"Missing: {missing_paths}"
            )


def _methods_with_stage(methods: Sequence[str], stage: StageId) -> list[str]:
    return [method for method in methods if _method_has_stage(method, stage)]


def _method_has_stage(method: str, stage: StageId) -> bool:
    return any(step.stage is stage for step in get_workflow(method).steps)


def _qualify(commands: Sequence[StageCommand], variant: str) -> list[StageCommand]:
    return [
        StageCommand(
            stage=command.stage,
            argv=list(command.argv),
            method=command.method,
            split=command.split,
            variant=variant,
        )
        for command in commands
    ]


def _without_aggregate(commands: Sequence[StageCommand]) -> list[StageCommand]:
    return [command for command in commands if command.stage is not StageId.AGGREGATE]


def run_stage_plan(commands: Sequence[StageCommand]) -> None:
    for command in commands:
        subprocess.run(command.argv, check=True)


def format_commands(commands: Sequence[StageCommand], *, color: bool = False) -> str:
    return "\n\n".join(_format_command_block(index, command, color=color) for index, command in enumerate(commands, 1))


def _format_command_block(index: int, command: StageCommand, *, color: bool) -> str:
    qualifier = ""
    if command.method is not None:
        qualifier = f" method={command.method}"
    elif command.split is not None:
        qualifier = f" split={command.split}"
    if command.variant is not None:
        qualifier = f"{qualifier} variant={command.variant}"
    lines = [
        f"[{index}] {command.stage}{qualifier}",
        f"script: {_command_script(command.argv)}",
        "command:",
    ]
    lines.extend(_format_argv_lines(command.argv, color=color))
    return "\n".join(lines)


def _command_script(argv: Sequence[str]) -> str:
    for value in argv:
        if value.endswith(".py"):
            return value
    return argv[0] if argv else ""


def _format_argv_lines(argv: Sequence[str], *, color: bool) -> list[str]:
    lines: list[str] = []
    index = 0
    while index < len(argv):
        value = argv[index]
        if value.startswith("--"):
            option = _color_option(value, color=color)
            if index + 1 < len(argv) and not argv[index + 1].startswith("--"):
                lines.append(f"  {option} {argv[index + 1]}")
                index += 2
            else:
                lines.append(f"  {option}")
                index += 1
        else:
            lines.append(f"  {value}")
            index += 1
    return lines


def _color_option(value: str, *, color: bool) -> str:
    if not color:
        return value
    return f"\033[36m{value}\033[0m"

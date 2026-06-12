from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from graph_memory.registry import Registry
from graph_memory.registry.methods import ArtifactKind, GraphInputSource
from graph_memory.io import read_json, write_json
from graph_memory.observability import now_iso
from scripts.workflow.planner import _materialize_variant_manifest, _method_has_stage
from scripts.workflow.types import ArtifactRole, ArtifactState, StageId


def inspect_experiment_status(manifest: dict[str, Any]) -> list[dict[str, str]]:
    """Inspect ordinary artifacts and expanded ablation namespaces."""

    rows: list[dict[str, str]] = []
    for split in ("train", "dev", "test"):
        rows.append(_prepare_status(manifest, split))
        rows.append(_graph_status(manifest, split))
    for method in manifest["selected_methods"]:
        if _method_has_stage(method, StageId.PAIRS):
            rows.append(_pair_status(manifest, method))
        if _method_has_stage(method, StageId.TRAIN):
            rows.append(_train_status(manifest, method))
        if _method_has_stage(method, StageId.TUNE):
            rows.append(_tune_status(manifest, method))
        rows.append(_retrieval_status(manifest, method))
        rows.append(_evaluate_status(manifest, method))
    rows.append(_aggregate_status(manifest))

    for method, variants in manifest.get("artifacts", {}).get("ablations", {}).items():
        for variant, record in variants.items():
            if record["baseline_alias"]:
                rows.extend(
                    _alias_row(stage, method=method, variant=variant, path=record["artifacts"][role.value])
                    for stage, role in (
                        (StageId.PAIRS, ArtifactRole.TRAIN_PAIRS),
                        (StageId.TRAIN, ArtifactRole.CHECKPOINT),
                        (StageId.RETRIEVE, ArtifactRole.PREDICTIONS),
                        (StageId.EVALUATE, ArtifactRole.METRICS),
                    )
                )
                continue
            alias_roles = {alias["role"] for alias in record["aliases"]}
            rows.append(_artifact_or_alias_row(StageId.PAIRS, ArtifactRole.TRAIN_PAIRS, method, variant, record, alias_roles))
            rows.append(_artifact_or_alias_row(StageId.TRAIN, ArtifactRole.CHECKPOINT, method, variant, record, alias_roles))
            variant_manifest = _materialize_variant_manifest(manifest, method, record)
            retrieval = _retrieval_status(variant_manifest, method)
            retrieval["variant"] = variant
            rows.append(retrieval)
            rows.append(_artifact_or_alias_row(StageId.EVALUATE, ArtifactRole.METRICS, method, variant, record, alias_roles))
    return rows


def update_manifest_status(manifest: dict[str, Any]) -> dict[str, Any]:
    rows = inspect_experiment_status(manifest)
    manifest["stage_status"] = {_status_key(row): row for row in rows}
    manifest["updated_at"] = now_iso()
    write_json(manifest["paths"]["manifest"], manifest)
    return manifest


def format_status(rows: Sequence[dict[str, str]]) -> str:
    lines = []
    for row in rows:
        qualifier = row.get("method") or row.get("split") or ""
        variant = f" variant={row['variant']}" if row.get("variant") else ""
        lines.append(f"{row['stage']} {qualifier}{variant} {row['state']}".strip())
    return "\n".join(lines)


def _artifact_or_alias_row(
    stage: StageId,
    role: ArtifactRole,
    method: str,
    variant: str,
    record: dict[str, Any],
    alias_roles: set[str],
) -> dict[str, str]:
    path = record["artifacts"][role.value]
    if role.value in alias_roles:
        return _alias_row(stage, method=method, variant=variant, path=path)
    row = _artifact_status(stage=stage.value, method=method, path=path)
    row["variant"] = variant
    return row


def _prepare_status(manifest: dict[str, Any], split: str) -> dict[str, str]:
    artifacts = manifest["artifacts"]["inputs"][split]
    path = artifacts["input"]
    output_path = Path(path)
    split_config = manifest["effective_config"]["splits"][split]
    raw_path = manifest["effective_config"]["raw"][split_config["source"]]
    return _summary_status(
        stage=StageId.PREPARE.value,
        path=path,
        summary_path=output_path.with_name(f"{output_path.stem}.run_summary.json"),
        script="prepare_hotpotqa.py",
        expected_inputs={"raw": raw_path},
        expected_outputs={
            "inputs": artifacts["input"],
            "labels": artifacts["labels"],
            "combined": artifacts["combined"],
        },
        expected_config={
            "max_examples": split_config["max_examples"],
            "seed": split_config["seed"],
            "offset": split_config["offset"],
        },
        split=split,
    )


def _graph_status(manifest: dict[str, Any], split: str) -> dict[str, str]:
    path = manifest["artifacts"]["graphs"][split]
    output_path = Path(path)
    return _summary_status(
        stage=StageId.GRAPHS.value,
        path=path,
        summary_path=output_path.with_name(f"{output_path.stem}.run_summary.json"),
        script="build_graphs.py",
        expected_inputs={"tasks": manifest["artifacts"]["inputs"][split]["input"]},
        expected_outputs={"graphs": path},
        expected_config=manifest["effective_config"]["graph"],
        split=split,
    )


def _pair_status(manifest: dict[str, Any], method: str) -> dict[str, str]:
    learned = manifest["artifacts"]["learned"][method]
    method_config = manifest["effective_config"]["resolved_method_configs"][method]
    return _summary_status(
        stage=StageId.PAIRS.value,
        path=learned["train_pairs"],
        summary_path=Path(learned["train_pair_run_summary"]),
        script="build_train_pairs.py",
        expected_inputs={
            "tasks": manifest["artifacts"]["inputs"]["train"]["input"],
            "labels": manifest["artifacts"]["inputs"]["train"]["labels"],
            "graphs": manifest["artifacts"]["graphs"]["train"],
        },
        expected_outputs={
            "pairs": learned["train_pairs"],
            "summary": learned["train_pair_summary"],
        },
        expected_config=dict(method_config.get("pairs", {})),
        method=method,
    )


def _train_status(manifest: dict[str, Any], method: str) -> dict[str, str]:
    learned = manifest["artifacts"]["learned"][method]
    definition = Registry.methods.get(method)
    expected_inputs = {
        "train_tasks": manifest["artifacts"]["inputs"]["train"]["input"],
        "train_labels": manifest["artifacts"]["inputs"]["train"]["labels"],
        "train_pairs": learned["train_pairs"],
        "dev_tasks": manifest["artifacts"]["inputs"]["dev"]["input"],
        "dev_labels": manifest["artifacts"]["inputs"]["dev"]["labels"],
    }
    if definition.dependencies.graphs is GraphInputSource.GRAPH_ARTIFACT:
        expected_inputs["train_graphs"] = manifest["artifacts"]["graphs"]["train"]
        expected_inputs["dev_graphs"] = manifest["artifacts"]["graphs"]["dev"]
    train_artifact = definition.train_artifact
    if train_artifact is None:
        raise ValueError(f"Trainable workflow requires a train artifact: {method}")
    return _summary_status(
        stage=StageId.TRAIN.value,
        path=learned["best_checkpoint"],
        summary_path=Path(learned["train_run_summary"]),
        script="train_method.py",
        expected_inputs=expected_inputs,
        expected_outputs={
            "best_checkpoint": learned["best_checkpoint"],
            "metrics": learned["train_metrics"],
        },
        expected_config={"method": method},
        artifact_kind=train_artifact.kind,
        method=method,
    )


def _tune_status(manifest: dict[str, Any], method: str) -> dict[str, str]:
    path = manifest["artifacts"]["tuned"][method]
    output_path = Path(path)
    expected_config: dict[str, object] = {
        "method": method,
        "top_k": manifest["effective_config"]["top_k"],
        "grid_config": str(manifest["effective_config"]["search_spaces"]["graph_rerank"]),
    }
    return _summary_status(
        stage=StageId.TUNE.value,
        path=path,
        summary_path=output_path.with_name(f"{output_path.stem}.run_summary.json"),
        script="tune_graph_rerank.py",
        expected_inputs={
            "tasks": manifest["artifacts"]["inputs"]["dev"]["input"],
            "labels": manifest["artifacts"]["inputs"]["dev"]["labels"],
            "graphs": manifest["artifacts"]["graphs"]["dev"],
            "grid_config": str(manifest["effective_config"]["search_spaces"]["graph_rerank"]),
        },
        expected_outputs={"selected_config": path},
        expected_config=expected_config,
        method=method,
    )


def _evaluate_status(manifest: dict[str, Any], method: str) -> dict[str, str]:
    path = manifest["artifacts"]["metrics"][method]
    output_path = Path(path)
    return _summary_status(
        stage=StageId.EVALUATE.value,
        path=path,
        summary_path=output_path.with_name(f"{output_path.stem}.run_summary.json"),
        script="evaluate_retrieval.py",
        expected_inputs={
            "predictions": manifest["artifacts"]["predictions"][method],
            "labels": manifest["artifacts"]["inputs"]["test"]["labels"],
            "graphs": manifest["artifacts"]["graphs"]["test"],
        },
        expected_outputs={
            "metrics": path,
            "failure_cases": manifest["artifacts"]["failure_cases"][method],
        },
        expected_config={"failure_case_limit": 50},
        method=method,
    )


def _aggregate_status(manifest: dict[str, Any]) -> dict[str, str]:
    table = manifest["artifacts"]["tables"]["main"]
    table_path = Path(table)
    expected_outputs = {
        "main": manifest["artifacts"]["tables"]["main"],
        "path": manifest["artifacts"]["tables"]["path"],
        "efficiency": manifest["artifacts"]["tables"]["efficiency"],
    }
    if "ablation" in manifest["artifacts"]["tables"]:
        expected_outputs["ablation"] = manifest["artifacts"]["tables"]["ablation"]
    return _summary_status(
        stage=StageId.AGGREGATE.value,
        path=table,
        summary_path=table_path.with_name("aggregate_tables.run_summary.json"),
        script="aggregate_tables.py",
        expected_inputs={"input_dir": (Path(manifest["paths"]["run_dir"]) / "metrics").as_posix()},
        expected_outputs=expected_outputs,
        expected_config={},
    )


def _artifact_status(
    *,
    stage: str,
    path: str,
    method: str | None = None,
    split: str | None = None,
    artifact_kind: ArtifactKind | None = None,
) -> dict[str, str]:
    artifact_path = Path(path)
    exists = artifact_path.exists()
    if exists and artifact_kind is ArtifactKind.FILE:
        exists = artifact_path.is_file()
    if exists and artifact_kind is ArtifactKind.DIRECTORY:
        exists = artifact_path.is_dir()
    state = ArtifactState.COMPLETE.value if exists else ArtifactState.MISSING.value
    row = {"stage": stage, "state": state, "path": path}
    if method is not None:
        row["method"] = method
    if split is not None:
        row["split"] = split
    return row


def _summary_status(
    *,
    stage: str,
    path: str,
    summary_path: Path,
    script: str,
    expected_inputs: dict[str, object],
    expected_outputs: dict[str, object],
    expected_config: dict[str, object],
    method: str | None = None,
    split: str | None = None,
    artifact_kind: ArtifactKind | None = None,
) -> dict[str, str]:
    row = _artifact_status(
        stage=stage,
        method=method,
        split=split,
        path=path,
        artifact_kind=artifact_kind,
    )
    if row["state"] == ArtifactState.MISSING.value:
        return row
    if not summary_path.exists():
        row["state"] = ArtifactState.STALE.value
        return row
    summary = read_json(summary_path)
    if not isinstance(summary, dict):
        row["state"] = ArtifactState.STALE.value
        return row
    if summary.get("status") != "success" or summary.get("script") != script:
        row["state"] = ArtifactState.STALE.value
        return row
    if not _summary_section_matches(summary.get("inputs"), expected_inputs, path_values=True):
        row["state"] = ArtifactState.STALE.value
        return row
    if not _summary_section_matches(summary.get("outputs"), expected_outputs, path_values=True):
        row["state"] = ArtifactState.STALE.value
        return row
    if not _summary_section_matches(summary.get("effective_config"), expected_config, path_values=False):
        row["state"] = ArtifactState.STALE.value
    return row


def _retrieval_status(manifest: dict[str, Any], method: str) -> dict[str, str]:
    path = Path(manifest["artifacts"]["predictions"][method])
    row = {
        "stage": StageId.RETRIEVE.value,
        "method": method,
        "path": path.as_posix(),
        "state": ArtifactState.MISSING.value,
    }
    if not path.exists():
        return row
    summary_path = path.with_name(f"{path.stem}.run_summary.json")
    if not summary_path.exists():
        row["state"] = ArtifactState.STALE.value
        return row
    summary = read_json(summary_path)
    if not isinstance(summary, dict):
        row["state"] = ArtifactState.STALE.value
        return row
    expected_tasks = manifest["artifacts"]["inputs"]["test"]["input"]
    expected_top_k = manifest["effective_config"]["top_k"]
    if (
        summary.get("status") == "success"
        and _same_path(summary.get("inputs", {}).get("tasks"), expected_tasks)
        and _same_path(summary.get("outputs", {}).get("predictions"), path)
        and summary.get("effective_config", {}).get("method") == method
        and summary.get("effective_config", {}).get("top_k") == expected_top_k
    ):
        row["state"] = ArtifactState.COMPLETE.value
    else:
        row["state"] = ArtifactState.STALE.value
    return row


def _alias_row(stage: StageId, *, method: str, variant: str, path: str) -> dict[str, str]:
    return {
        "stage": stage.value,
        "method": method,
        "variant": variant,
        "state": ArtifactState.ALIAS.value,
        "path": path,
    }


def _status_key(row: dict[str, str]) -> str:
    if row.get("variant"):
        return f"{row['stage']}:{row['method']}:{row['variant']}"
    if row.get("method"):
        return f"{row['stage']}:{row['method']}"
    if row.get("split"):
        return f"{row['stage']}:{row['split']}"
    return row["stage"]


def _same_path(left: object, right: str | Path) -> bool:
    return isinstance(left, str) and Path(left) == Path(right)


def _summary_section_matches(section: object, expected: dict[str, object], *, path_values: bool) -> bool:
    if not isinstance(section, dict):
        return not expected
    for key, expected_value in expected.items():
        if key not in section:
            return False
        actual_value = section[key]
        if path_values and isinstance(expected_value, (str, Path)):
            if not _same_path(actual_value, expected_value):
                return False
        elif actual_value != expected_value:
            return False
    return True

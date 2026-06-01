from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from graph_memory.io import read_json, write_json
from graph_memory.observability import now_iso
from scripts.workflow.planner import _materialize_variant_manifest, _method_has_stage
from scripts.workflow.types import ArtifactRole, ArtifactState, StageId


def inspect_experiment_status(manifest: dict[str, Any]) -> list[dict[str, str]]:
    """Inspect ordinary artifacts and expanded ablation namespaces."""

    rows: list[dict[str, str]] = []
    for split in ("train", "dev", "test"):
        rows.append(_artifact_status(stage=StageId.PREPARE.value, split=split, path=manifest["artifacts"]["inputs"][split]["input"]))
        rows.append(_artifact_status(stage=StageId.GRAPHS.value, split=split, path=manifest["artifacts"]["graphs"][split]))
    for method in manifest["selected_methods"]:
        if _method_has_stage(method, StageId.PAIRS):
            rows.append(_artifact_status(stage=StageId.PAIRS.value, method=method, path=manifest["artifacts"]["learned"][method]["train_pairs"]))
        if _method_has_stage(method, StageId.TRAIN):
            rows.append(_artifact_status(stage=StageId.TRAIN.value, method=method, path=manifest["artifacts"]["learned"][method]["best_checkpoint"]))
        if _method_has_stage(method, StageId.TUNE):
            rows.append(_artifact_status(stage=StageId.TUNE.value, method=method, path=manifest["artifacts"]["tuned"][method]))
        rows.append(_retrieval_status(manifest, method))
        rows.append(_artifact_status(stage=StageId.EVALUATE.value, method=method, path=manifest["artifacts"]["metrics"][method]))
    rows.append(_artifact_status(stage=StageId.AGGREGATE.value, path=manifest["artifacts"]["tables"]["main"]))

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


def _artifact_status(
    *,
    stage: str,
    path: str,
    method: str | None = None,
    split: str | None = None,
) -> dict[str, str]:
    state = ArtifactState.COMPLETE.value if Path(path).exists() else ArtifactState.MISSING.value
    row = {"stage": stage, "state": state, "path": path}
    if method is not None:
        row["method"] = method
    if split is not None:
        row["split"] = split
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
        row["state"] = ArtifactState.COMPLETE.value
        return row
    summary = read_json(summary_path)
    if not isinstance(summary, dict):
        row["state"] = ArtifactState.STALE.value
        return row
    expected_tasks = manifest["artifacts"]["inputs"]["test"]["input"]
    expected_top_k = manifest["effective_config"]["top_k"]
    if (
        summary.get("status") == "success"
        and summary.get("inputs", {}).get("tasks") == expected_tasks
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

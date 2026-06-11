from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from scripts.workflow.types import (
    ArtifactAlias,
    ArtifactRole,
    StageId,
    VariantArtifactNamespace,
    VariantSpec,
)
from scripts.workflow.registry import checkpoint_artifact_name


_ROLE_STAGE = {
    ArtifactRole.TRAIN_PAIRS: StageId.PAIRS,
    ArtifactRole.TRAIN_PAIR_SUMMARY: StageId.PAIRS,
    ArtifactRole.TRAIN_PAIR_RUN_SUMMARY: StageId.PAIRS,
    ArtifactRole.EFFECTIVE_TRAINING_CONFIG: StageId.TRAIN,
    ArtifactRole.TRAIN_METRICS: StageId.TRAIN,
    ArtifactRole.TRAIN_RUN_SUMMARY: StageId.TRAIN,
    ArtifactRole.CHECKPOINT: StageId.TRAIN,
    ArtifactRole.PREDICTIONS: StageId.RETRIEVE,
    ArtifactRole.METRICS: StageId.EVALUATE,
    ArtifactRole.FAILURE_CASES: StageId.EVALUATE,
}


def build_main_method_artifacts(run_dir: str | Path, method: str) -> dict[ArtifactRole, str]:
    """Build stable ordinary-run paths for one trainable method."""

    root = Path(run_dir)
    learned = root / "learned" / method
    return {
        ArtifactRole.TRAIN_PAIRS: (learned / "train.pairs.json").as_posix(),
        ArtifactRole.TRAIN_PAIR_SUMMARY: (learned / "train.pairs.summary.json").as_posix(),
        ArtifactRole.TRAIN_PAIR_RUN_SUMMARY: (learned / "train.pairs.run_summary.json").as_posix(),
        ArtifactRole.EFFECTIVE_TRAINING_CONFIG: (learned / "effective_training_config.json").as_posix(),
        ArtifactRole.TRAIN_METRICS: (learned / "train_metrics.jsonl").as_posix(),
        ArtifactRole.TRAIN_RUN_SUMMARY: (learned / "train_run_summary.json").as_posix(),
        ArtifactRole.CHECKPOINT: (learned / "checkpoints" / checkpoint_artifact_name(method)).as_posix(),
        ArtifactRole.PREDICTIONS: (root / "predictions" / f"test.{method}.ranked.json").as_posix(),
        ArtifactRole.METRICS: (root / "metrics" / f"test.{method}.metrics.csv").as_posix(),
        ArtifactRole.FAILURE_CASES: (root / "debug" / f"failure_cases_{method}.jsonl").as_posix(),
    }


def build_variant_artifact_namespace(
    run_dir: str | Path,
    method: str,
    variant: VariantSpec,
    main_artifacts: Mapping[ArtifactRole, str],
) -> VariantArtifactNamespace:
    """Allocate local outputs at and below a variant's invalidation boundary."""

    from scripts.workflow.planner import earliest_invalidated_stage
    from scripts.workflow.registry import get_workflow

    invalidated_from = earliest_invalidated_stage(get_workflow(method), variant)
    local_paths = _variant_local_paths(run_dir, method, variant.identifier.value)
    if invalidated_from is None:
        baseline_aliases = tuple(
            ArtifactAlias(role=role, source=source, target=local_paths[role])
            for role, source in main_artifacts.items()
        )
        return VariantArtifactNamespace(
            method=method,
            variant=variant.identifier.value,
            invalidated_from=None,
            paths=dict(main_artifacts),
            local_paths=local_paths,
            aliases=baseline_aliases,
        )

    stage_index = {stage: index for index, stage in enumerate(StageId)}
    boundary = stage_index[invalidated_from]
    paths: dict[ArtifactRole, str] = {}
    aliases: list[ArtifactAlias] = []
    for role, source in main_artifacts.items():
        role_stage = _ROLE_STAGE[role]
        if stage_index[role_stage] >= boundary:
            paths[role] = local_paths[role]
        else:
            paths[role] = source
            aliases.append(ArtifactAlias(role=role, source=source, target=local_paths[role]))
    return VariantArtifactNamespace(
        method=method,
        variant=variant.identifier.value,
        invalidated_from=invalidated_from,
        paths=paths,
        local_paths=local_paths,
        aliases=tuple(aliases),
    )


def _variant_local_paths(run_dir: str | Path, method: str, variant: str) -> dict[ArtifactRole, str]:
    root = Path(run_dir) / "ablations" / method / variant
    return {
        ArtifactRole.TRAIN_PAIRS: (root / "train.pairs.json").as_posix(),
        ArtifactRole.TRAIN_PAIR_SUMMARY: (root / "train.pairs.summary.json").as_posix(),
        ArtifactRole.TRAIN_PAIR_RUN_SUMMARY: (root / "train.pairs.run_summary.json").as_posix(),
        ArtifactRole.EFFECTIVE_TRAINING_CONFIG: (root / "effective_training_config.json").as_posix(),
        ArtifactRole.TRAIN_METRICS: (root / "train_metrics.jsonl").as_posix(),
        ArtifactRole.TRAIN_RUN_SUMMARY: (root / "train_run_summary.json").as_posix(),
        ArtifactRole.CHECKPOINT: (root / "checkpoints" / "best.pt").as_posix(),
        ArtifactRole.PREDICTIONS: (root / "predictions" / "test.ranked.json").as_posix(),
        ArtifactRole.METRICS: (root / "metrics" / "test.metrics.csv").as_posix(),
        ArtifactRole.FAILURE_CASES: (root / "debug" / "failure_cases.jsonl").as_posix(),
    }

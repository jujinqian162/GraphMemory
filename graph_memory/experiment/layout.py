from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from graph_memory.experiment.config import (
    AliasArtifactRef,
    ArtifactRef,
    PublicStageName,
    SplitName,
)
from graph_memory.registry.retrieval import RetrievalMethodId

RunMode = Literal["single", "multirun"]


@dataclass(frozen=True)
class SingleRunIdentity:
    mode: Literal["single"] = "single"


@dataclass(frozen=True)
class MultirunIdentity:
    job_num: int
    override_dirname: str
    mode: Literal["multirun"] = "multirun"

    def __post_init__(self) -> None:
        if self.job_num < 0:
            raise ValueError("multirun identity requires a non-negative job_num")
        if not self.override_dirname:
            raise ValueError("multirun identity requires override_dirname")


RunIdentity = SingleRunIdentity | MultirunIdentity


@dataclass(frozen=True)
class RunLayout:
    repository_root: Path
    name: str
    identity: RunIdentity = SingleRunIdentity()

    def __post_init__(self) -> None:
        root = self.repository_root.resolve()
        object.__setattr__(self, "repository_root", root)
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", self.name):
            raise ValueError(
                "run name must start with an alphanumeric character and contain only letters, digits, '.', '_' or '-'"
            )

    @property
    def mode(self) -> RunMode:
        return self.identity.mode

    @property
    def runs_root(self) -> Path:
        return self.repository_root / "runs"

    @property
    def named_root(self) -> Path:
        return self.runs_root / self.name

    @property
    def run_dir(self) -> Path:
        if isinstance(self.identity, SingleRunIdentity):
            return self.named_root
        suffix = _safe_override_dirname(self.identity.override_dirname)
        return self.named_root / f"{self.identity.job_num}_{suffix}"

    @property
    def resolved_config(self) -> Path:
        return self.run_dir / "config" / "resolved.yaml"

    @property
    def overrides(self) -> Path:
        return self.run_dir / "config" / "overrides.yaml"

    @property
    def run_state(self) -> Path:
        return self.run_dir / "run_state.yaml"

    @property
    def ablation_metrics_index(self) -> Path:
        return self.run_dir / "ablations" / "metrics_index.yaml"

    def stage_config(
        self,
        stage: PublicStageName,
        *,
        method: RetrievalMethodId | None = None,
        split: SplitName | None = None,
        variant: str | None = None,
    ) -> Path:
        qualifier = _unit_name(method=method, split=split)
        if variant is None:
            return self.run_dir / "config" / "stages" / stage / f"{qualifier}.yaml"
        method_name = _method_name(method)
        if method_name is None:
            raise ValueError("variant stage config requires a method")
        return (
            self.run_dir
            / "config"
            / "stages"
            / "ablations"
            / method_name
            / variant
            / f"{stage}.yaml"
        )

    def inputs(self, split: SplitName) -> dict[str, Path]:
        root = self.run_dir / "inputs"
        return {
            "input": root / f"{split}.input.json",
            "labels": root / f"{split}.labels.json",
            "combined": root / f"{split}.combined.json",
        }

    def graph(self, split: SplitName) -> Path:
        return self.run_dir / "graphs" / f"{split}.graphs.json"

    def tuned(self, method: RetrievalMethodId) -> Path:
        return self.run_dir / "tuned" / f"{_method_name(method)}.dev_selected.json"

    def tuned_candidates(self, method: RetrievalMethodId) -> Path:
        return self.run_dir / "tuned" / f"{_method_name(method)}.dev_candidates.json"

    def learned_root(
        self,
        method: RetrievalMethodId,
        *,
        variant: str | None = None,
    ) -> Path:
        method_name = _required_method_name(method)
        if variant is None:
            return self.run_dir / "learned" / method_name
        return self.run_dir / "ablations" / method_name / variant / "learned"

    def train_pairs(
        self, method: RetrievalMethodId, *, variant: str | None = None
    ) -> Path:
        return self.learned_root(method, variant=variant) / "train.pairs.json"

    def train_pair_summary(
        self, method: RetrievalMethodId, *, variant: str | None = None
    ) -> Path:
        return self.learned_root(method, variant=variant) / "train.pairs.summary.json"

    def training_metrics(
        self, method: RetrievalMethodId, *, variant: str | None = None
    ) -> Path:
        return self.learned_root(method, variant=variant) / "train_metrics.jsonl"

    def checkpoint(
        self,
        method: RetrievalMethodId,
        *,
        kind: Literal["file", "directory"],
        variant: str | None = None,
    ) -> Path:
        root = self.learned_root(method, variant=variant) / "checkpoints"
        return root / ("best.pt" if kind == "file" else "best_model")

    def prediction(
        self, method: RetrievalMethodId, *, variant: str | None = None
    ) -> Path:
        method_name = _required_method_name(method)
        if variant is None:
            return self.run_dir / "predictions" / f"test.{method_name}.ranked.json"
        return (
            self.run_dir
            / "ablations"
            / method_name
            / variant
            / "predictions"
            / "test.ranked.json"
        )

    def metric(self, method: RetrievalMethodId, *, variant: str | None = None) -> Path:
        method_name = _required_method_name(method)
        if variant is None:
            return self.run_dir / "metrics" / f"test.{method_name}.metrics.csv"
        return (
            self.run_dir
            / "ablations"
            / method_name
            / variant
            / "metrics"
            / "test.metrics.csv"
        )

    def failure_cases(
        self, method: RetrievalMethodId, *, variant: str | None = None
    ) -> Path:
        method_name = _required_method_name(method)
        if variant is None:
            return self.run_dir / "debug" / f"failure_cases_{method_name}.jsonl"
        return (
            self.run_dir
            / "ablations"
            / method_name
            / variant
            / "debug"
            / "failure_cases.jsonl"
        )

    def table(self, kind: Literal["main", "path", "efficiency", "ablation"]) -> Path:
        filename = {
            "main": "main_results.csv",
            "path": "path_results.csv",
            "efficiency": "efficiency_results.csv",
            "ablation": "ablation_results.csv",
        }[kind]
        return self.run_dir / "tables" / filename

    def summary_for(self, primary: Path, *, stage: PublicStageName) -> Path:
        if primary.suffix:
            return primary.with_name(f"{primary.stem}.run_summary.yaml")
        return primary.parent / f"{stage}.run_summary.yaml"

    def artifact(
        self,
        *,
        role: str,
        path: Path,
        kind: Literal["file", "directory"] = "file",
    ) -> ArtifactRef:
        resolved = path.resolve()
        self._require_contained(resolved)
        return ArtifactRef(role=role, path=resolved, kind=kind)

    def alias_artifact(
        self,
        *,
        role: str,
        path: Path,
        alias_of: Path,
        kind: Literal["file", "directory"] = "file",
    ) -> AliasArtifactRef:
        resolved = path.resolve()
        source = alias_of.resolve()
        self._require_contained(resolved)
        self._require_contained(source)
        return AliasArtifactRef(
            role=role,
            path=resolved,
            kind=kind,
            alias_of=source,
        )

    def _require_contained(self, path: Path) -> None:
        try:
            path.relative_to(self.run_dir.resolve())
        except ValueError as error:
            raise ValueError(f"run artifact escapes run directory: {path}") from error


def _method_name(method: RetrievalMethodId | None) -> str | None:
    if method is None:
        return None
    return None if method is None else method.value


def _required_method_name(method: RetrievalMethodId) -> str:
    value = _method_name(method)
    if value is None or not value:
        raise ValueError("method name is required")
    return value


def _unit_name(*, method: RetrievalMethodId | None, split: SplitName | None) -> str:
    method_name = _method_name(method)
    if method_name and split:
        raise ValueError("stage unit cannot have both method and split")
    return method_name or split or "aggregate"


def _safe_override_dirname(value: str) -> str:
    sanitized = re.sub(r'[<>:"/\\|?*]', "_", value).strip(" .")
    if not sanitized:
        raise ValueError("override_dirname has no usable path characters")
    return sanitized


__all__ = [
    "MultirunIdentity",
    "RunIdentity",
    "RunLayout",
    "RunMode",
    "SingleRunIdentity",
]

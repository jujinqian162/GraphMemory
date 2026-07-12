from __future__ import annotations

import shutil
from pathlib import Path

from graph_memory.experiment.config import ClosedModel
from graph_memory.experiment.layout import RunLayout


class ResetCommandConfig(ClosedModel):
    name: str
    job: str | None = None


def reset_named_run(
    name: str,
    *,
    repository_root: Path,
    job: str | None = None,
) -> Path:
    layout = RunLayout(repository_root, name)
    if layout.named_root.is_symlink():
        raise ValueError(f"reset refuses symbolic-link run roots: {layout.named_root}")
    runs_root = layout.runs_root.resolve()
    if job is not None and (Path(job).name != job or job in {".", ".."}):
        raise ValueError(f"invalid multirun job selector: {job!r}")
    source = layout.named_root if job is None else layout.named_root / job
    if source.is_symlink():
        raise ValueError(f"reset refuses symbolic-link run roots: {source}")
    target = source.resolve()
    expected_parent = runs_root if job is None else layout.named_root.resolve()
    if target.parent != expected_parent or target == expected_parent:
        raise ValueError(f"reset target is not a direct run child: {target}")
    if source.exists():
        shutil.rmtree(source)
    return target

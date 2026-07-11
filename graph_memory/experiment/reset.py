from __future__ import annotations

import shutil
from pathlib import Path

from graph_memory.experiment.config import ClosedModel
from graph_memory.experiment.layout import RunLayout


class ResetCommandConfig(ClosedModel):
    name: str


def reset_named_run(name: str, *, repository_root: Path) -> Path:
    layout = RunLayout(repository_root, name)
    if layout.named_root.is_symlink():
        raise ValueError(f"reset refuses symbolic-link run roots: {layout.named_root}")
    runs_root = layout.runs_root.resolve()
    target = layout.named_root.resolve()
    if target.parent != runs_root or target == runs_root:
        raise ValueError(f"reset target is not a direct named child of runs/: {target}")
    if layout.named_root.exists():
        shutil.rmtree(layout.named_root)
    return target


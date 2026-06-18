from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from abstraction.domain.common.capability_names import StageKind
from abstraction.domain.common.identifiers import StageId


@dataclass(frozen=True)
class ScriptArgument:
    name: str
    value: str


@dataclass(frozen=True)
class ScriptCommand:
    stage_id: StageId
    stage_kind: StageKind
    script_name: str
    arguments: Sequence[ScriptArgument]
    reads_artifacts: Sequence[str]
    writes_artifacts: Sequence[str]


@dataclass(frozen=True)
class ScriptCommandResult:
    command: ScriptCommand
    produced_artifacts: Sequence[str]
    selected_branch: str


@dataclass(frozen=True)
class WorkflowCommandPlan:
    commands: Sequence[ScriptCommand]
    command_dependencies: Mapping[StageId, Sequence[StageId]]


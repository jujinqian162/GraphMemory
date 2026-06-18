from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from abstraction.domain.scripts.commands import ScriptCommand
from abstraction.domain.workflow.manifest import ExperimentRunIntent
from abstraction.domain.workflow.stages import StageGraph


@dataclass(frozen=True)
class ScriptCliArguments:
    command: ScriptCommand
    intent: ExperimentRunIntent
    stage_graph: StageGraph
    flags: Mapping[str, str]


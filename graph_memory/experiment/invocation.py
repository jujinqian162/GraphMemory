from __future__ import annotations

import sys
from pathlib import Path

from pydantic import model_validator

from graph_memory.experiment.config import (
    AliasArtifactRef,
    ArtifactBinding,
    ClosedModel,
    PublicStageName,
    SplitName,
)
from graph_memory.experiment.stage_models import StageConfig
from graph_memory.registry.retrieval import RetrievalMethodId


class StageInvocation(ClosedModel):
    identifier: str
    stage: PublicStageName
    script: Path
    config_path: Path
    summary_path: Path
    config: StageConfig
    inputs: tuple[ArtifactBinding, ...]
    outputs: tuple[ArtifactBinding, ...]
    dependencies: tuple[str, ...]
    method: RetrievalMethodId | None = None
    split: SplitName | None = None
    variant: str | None = None

    @model_validator(mode="after")
    def validate_paths(self) -> StageInvocation:
        artifacts = (*self.inputs, *self.outputs)
        paths = (
            self.script,
            self.config_path,
            self.summary_path,
            *(artifact.path for artifact in artifacts),
            *(
                artifact.alias_of
                for artifact in artifacts
                if isinstance(artifact, AliasArtifactRef)
            ),
        )
        relative = next((path for path in paths if not path.is_absolute()), None)
        if relative is not None:
            raise ValueError(f"stage execution paths must be absolute: {relative}")
        return self

    @property
    def argv(self) -> tuple[str, ...]:
        return (
            sys.executable,
            str(self.script),
            "--config",
            str(self.config_path),
        )

    @property
    def primary_output(self) -> ArtifactBinding:
        if not self.outputs:
            raise ValueError(f"stage invocation has no outputs: {self.identifier}")
        return self.outputs[0]


__all__ = ["StageInvocation"]

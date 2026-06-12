from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, TypeVar

from graph_memory.config.patches import ConfigPatch
from graph_memory.contracts.common import JsonValue
from graph_memory.registry.ids import StageId

ConfigT = TypeVar("ConfigT")


@dataclass(frozen=True)
class StageConfigSpec(Generic[ConfigT]):
    stage: StageId
    config_type: type[ConfigT] | object
    parser_factory: Callable[[], argparse.ArgumentParser]
    config_path: Callable[[argparse.Namespace], Path | None]
    cli_patch: Callable[[argparse.Namespace], ConfigPatch]
    registry_patch: Callable[[argparse.Namespace, Mapping[str, JsonValue]], ConfigPatch]
    profile_name: Callable[[argparse.Namespace, Mapping[str, JsonValue]], str | None] | None = None


__all__ = ["StageConfigSpec"]

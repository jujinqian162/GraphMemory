from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, TypeVar

from graph_memory.config.codec import JsonConfigCodec
from graph_memory.config.converter import ConfigConverter
from graph_memory.config.patches import deep_merge_patch
from graph_memory.contracts.common import JsonValue
from graph_memory.registry.specs import StageConfigSpec

ConfigT = TypeVar("ConfigT")

DEFAULT_PROFILE_KEY = "default_profile"
PROFILE_KEY = "profiles"


class ConfigLoader:
    def __init__(self, codec: JsonConfigCodec | None = None, converter: ConfigConverter | None = None) -> None:
        self.codec = codec or JsonConfigCodec()
        self.converter = converter or ConfigConverter()

    def load(self, spec: StageConfigSpec[ConfigT], argv: Sequence[str] | None) -> ConfigT:
        parser = spec.parser_factory()
        namespace = parser.parse_args(argv)
        _set_provided_options(namespace, parser, argv)
        raw = spec.normalize_raw_config(namespace, self._load_raw_config(spec, namespace))
        merged = self._resolve_layers(spec, namespace, raw)
        return self.converter.structure(merged, spec.config_type)

    def to_json(self, config: object) -> JsonValue:
        return self.converter.unstructure(config)

    def write_resolved(self, path: str | Path, config: object) -> None:
        resolved = self.to_json(config)
        if not isinstance(resolved, dict):
            raise ValueError(f"Resolved config must be a JSON object: {type(config).__name__}")
        self.codec.write(path, resolved)

    def _load_raw_config(self, spec: StageConfigSpec[Any], namespace: Any) -> dict[str, Any]:
        path = spec.config_path(namespace)
        if path is None:
            return {}
        return self.codec.read(path)

    def _resolve_layers(
        self,
        spec: StageConfigSpec[Any],
        namespace: Any,
        raw: Mapping[str, JsonValue],
    ) -> dict[str, Any]:
        base = {
            key: value
            for key, value in raw.items()
            if key not in {PROFILE_KEY, DEFAULT_PROFILE_KEY}
        }
        profile_patch = self._profile_patch(spec, namespace, raw)
        registry_patch = spec.registry_patch(namespace, raw)
        cli_patch = spec.cli_patch(namespace)

        merged = deep_merge_patch(base, profile_patch)
        merged = deep_merge_patch(merged, registry_patch)
        return deep_merge_patch(merged, cli_patch)

    def _profile_patch(
        self,
        spec: StageConfigSpec[Any],
        namespace: Any, # HUMAN REVIEW POINT: namespace 是啥，为什么有Any这种糟糕的对象
        raw: Mapping[str, JsonValue],
    ) -> Mapping[str, Any]:
        profile_name = spec.profile_name(namespace, raw)
        if profile_name is None:
            return {}
        profiles = raw.get(PROFILE_KEY, {})
        if not isinstance(profiles, Mapping):
            raise ValueError("Config profiles must be an object.")
        if profile_name not in profiles:
            raise ValueError(f"Unknown config profile: {profile_name}")
        profile = profiles[profile_name]
        if not isinstance(profile, Mapping):
            raise ValueError(f"Config profile must be an object: {profile_name}")
        return dict(profile)


def _set_provided_options(
    namespace: argparse.Namespace,
    parser: argparse.ArgumentParser,
    argv: Sequence[str] | None,
) -> None:
    tokens = tuple(sys.argv[1:] if argv is None else argv)
    option_to_dest = {
        option: action.dest
        for action in parser._actions
        for option in action.option_strings
    }
    provided: set[str] = set()
    for token in tokens:
        if token == "--":
            break
        option = token.split("=", 1)[0]
        dest = option_to_dest.get(option)
        if dest is not None:
            provided.add(dest)
    setattr(namespace, "_provided_options", frozenset(provided))


CONFIG_LOADER = ConfigLoader()

__all__ = ["CONFIG_LOADER", "ConfigLoader"]

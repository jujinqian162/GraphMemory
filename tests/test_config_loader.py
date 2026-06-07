from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pytest

from graph_memory.config import CONFIG_LOADER, ConfigLoader
from graph_memory.config.patches import deep_merge_patch
from graph_memory.contracts.common import JsonValue
from graph_memory.registry.ids import StageId, StrEnum
from graph_memory.registry.specs import StageConfigSpec


class DemoMode(StrEnum):
    FAST = "fast"
    SLOW = "slow"


@dataclass(frozen=True)
class DemoNestedConfig:
    limit: int
    tags: tuple[str, ...]


@dataclass(frozen=True)
class DemoConfig:
    schema_version: int
    name: str
    path: Path
    stage: StageId
    mode: DemoMode
    nested: DemoNestedConfig
    optional_path: Path | None = None


def test_load_applies_profile_registry_patch_and_cli_patch_in_order(tmp_path: Path) -> None:
    config_path = tmp_path / "demo.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "default_profile": "cloud",
                "name": "base",
                "path": "base.txt",
                "stage": "retrieve",
                "mode": "fast",
                "nested": {"limit": 1, "tags": ["base"]},
                "profiles": {
                    "cloud": {
                        "name": "profile",
                        "path": "profile.txt",
                        "nested": {"limit": 2, "tags": ["profile"]},
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    config = ConfigLoader().load(
        _demo_spec(),
        [
            "--config",
            str(config_path),
            "--variant",
            "tight",
            "--name",
            "cli",
            "--limit",
            "4",
        ],
    )

    assert config == DemoConfig(
        schema_version=1,
        name="cli",
        path=Path("profile.txt"),
        stage=StageId.RETRIEVE,
        mode=DemoMode.FAST,
        nested=DemoNestedConfig(limit=4, tags=("profile",)),
    )


def test_load_skips_profiles_when_spec_has_no_profile_selector(tmp_path: Path) -> None:
    config_path = tmp_path / "demo.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "default_profile": "cloud",
                "name": "base",
                "path": "base.txt",
                "stage": "retrieve",
                "mode": "fast",
                "nested": {"limit": 1, "tags": ["base"]},
                "profiles": {
                    "cloud": {
                        "name": "profile",
                        "path": "profile.txt",
                        "nested": {"limit": 2, "tags": ["profile"]},
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    config = ConfigLoader().load(
        _demo_spec_without_profile_selector(),
        [
            "--config",
            str(config_path),
        ],
    )

    assert config == DemoConfig(
        schema_version=1,
        name="base",
        path=Path("base.txt"),
        stage=StageId.RETRIEVE,
        mode=DemoMode.FAST,
        nested=DemoNestedConfig(limit=1, tags=("base",)),
    )


def test_loader_write_resolved_unstructures_paths_enums_and_tuples(tmp_path: Path) -> None:
    output_path = tmp_path / "resolved.json"
    config = DemoConfig(
        schema_version=2,
        name="resolved",
        path=Path("artifact.json"),
        stage=StageId.TRAIN,
        mode=DemoMode.SLOW,
        nested=DemoNestedConfig(limit=9, tags=("a", "b")),
        optional_path=Path("maybe.json"),
    )

    CONFIG_LOADER.write_resolved(output_path, config)

    assert json.loads(output_path.read_text(encoding="utf-8")) == {
        "mode": "slow",
        "name": "resolved",
        "nested": {"limit": 9, "tags": ["a", "b"]},
        "optional_path": "maybe.json",
        "path": "artifact.json",
        "schema_version": 2,
        "stage": "train",
    }


def test_loader_rejects_unknown_fields(tmp_path: Path) -> None:
    config_path = tmp_path / "demo.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "name": "base",
                "path": "base.txt",
                "stage": "retrieve",
                "mode": "fast",
                "nested": {"limit": 1, "tags": []},
                "unsupported": True,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsupported fields"):
        ConfigLoader().load(_demo_spec(), ["--config", str(config_path)])


def test_deep_merge_patch_does_not_mutate_inputs() -> None:
    base = {"outer": {"left": 1}, "kept": True}
    patch = {"outer": {"right": 2}}

    merged = deep_merge_patch(base, patch)

    assert merged == {"outer": {"left": 1, "right": 2}, "kept": True}
    assert base == {"outer": {"left": 1}, "kept": True}
    assert patch == {"outer": {"right": 2}}


def test_config_api_stays_small() -> None:
    import graph_memory.config as config_api

    assert hasattr(config_api, "ConfigLoader")
    assert hasattr(config_api, "CONFIG_LOADER")
    assert not hasattr(config_api, "load_cli_config")
    assert not hasattr(config_api, "load_profiled_file")
    assert not hasattr(config_api, "ConfigSource")


def _demo_spec() -> StageConfigSpec[DemoConfig]:
    return StageConfigSpec(
        stage=StageId.RETRIEVE,
        config_type=DemoConfig,
        parser_factory=_demo_parser,
        config_path=lambda namespace: Path(namespace.config),
        profile_name=_profile_name,
        cli_patch=_cli_patch,
        registry_patch=_registry_patch,
    )


def _demo_spec_without_profile_selector() -> StageConfigSpec[DemoConfig]:
    return StageConfigSpec(
        stage=StageId.RETRIEVE,
        config_type=DemoConfig,
        parser_factory=_demo_parser,
        config_path=lambda namespace: Path(namespace.config),
        cli_patch=_cli_patch,
        registry_patch=_registry_patch,
    )


def _demo_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--profile", default=None)
    parser.add_argument("--variant", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--limit", type=int, default=None)
    return parser


def _profile_name(namespace: argparse.Namespace, raw: Mapping[str, JsonValue]) -> str | None:
    configured = raw.get("default_profile")
    if namespace.profile is not None:
        return str(namespace.profile)
    return str(configured) if configured is not None else None


def _cli_patch(namespace: argparse.Namespace) -> dict[str, Any]:
    patch: dict[str, Any] = {}
    if namespace.name is not None:
        patch["name"] = namespace.name
    if namespace.limit is not None:
        patch["nested"] = {"limit": namespace.limit}
    return patch


def _registry_patch(namespace: argparse.Namespace, raw: Mapping[str, JsonValue]) -> dict[str, Any]:
    if namespace.variant == "tight":
        return {"nested": {"limit": 3}}
    return {}

"""Compatibility facade for experiment orchestration.

New code should import from ``scripts.workflow``. This module remains for
existing callers while the user-facing entry point stays ``scripts/experiment.py``.
"""

from scripts.workflow import (
    build_stage_plan,
    format_commands,
    format_status,
    initialize_experiment,
    inspect_experiment_status,
    list_config_entries,
    list_method_specs,
    list_profile_specs,
    list_recipe_specs,
    list_stage_specs,
    load_experiment_config,
    load_manifest,
    run_stage_plan,
    update_manifest_status,
)
from scripts.workflow.manifest import (
    CONFIG_ROOT,
    DEFAULT_EXPERIMENT_CONFIG,
    DEFAULT_SEARCH_SPACE_CONFIG,
    EXPERIMENT_CONFIG_DIR,
    SEARCH_SPACE_CONFIG_DIR,
    STAGE_DESCRIPTIONS,
    STAGE_ORDER,
    TRAINING_CONFIG_DIR,
    build_effective_config,
    resolve_experiment_config_path,
    resolve_training_config_path,
)
from scripts.workflow.planner import required_stages_for_methods
from scripts.workflow.types import ConfigEntry

__all__ = [
    "CONFIG_ROOT",
    "DEFAULT_EXPERIMENT_CONFIG",
    "DEFAULT_SEARCH_SPACE_CONFIG",
    "EXPERIMENT_CONFIG_DIR",
    "SEARCH_SPACE_CONFIG_DIR",
    "STAGE_DESCRIPTIONS",
    "STAGE_ORDER",
    "TRAINING_CONFIG_DIR",
    "ConfigEntry",
    "build_effective_config",
    "build_stage_plan",
    "format_commands",
    "format_status",
    "initialize_experiment",
    "inspect_experiment_status",
    "list_config_entries",
    "list_method_specs",
    "list_profile_specs",
    "list_recipe_specs",
    "list_stage_specs",
    "load_experiment_config",
    "load_manifest",
    "required_stages_for_methods",
    "resolve_experiment_config_path",
    "resolve_training_config_path",
    "run_stage_plan",
    "update_manifest_status",
]

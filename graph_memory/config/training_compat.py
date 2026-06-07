from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, TypedDict

from graph_memory.infrastructure.io import merge_config, read_json
from graph_memory.models.graph_retriever.config.records import TrainableTrainingConfig
from graph_memory.training_pairs.config import NegativeSamplingConfig
from graph_memory.validation import validate_negative_sampling_config, validate_trainable_training_config


JsonConfig = dict[str, Any]


class EncoderConfig(TypedDict):
    model: str
    query_prefix: str
    passage_prefix: str


class ModelConfigValues(TypedDict):
    hidden_dim: int
    num_layers: int
    dropout: float
    ablation: str


def load_trainable_training_config(
    path: str | Path,
    *,
    profile: str | None = None,
    required_sections: Sequence[str] = ("encoder", "model", "optimization", "pair_sampling"),
) -> JsonConfig:
    config = read_json(path)
    if not isinstance(config, dict):
        raise ValueError(f"Training config must be a JSON object: {path}")
    return resolve_trainable_training_config(config, profile=profile, required_sections=required_sections)


def resolve_trainable_training_config(
    config: JsonConfig,
    *,
    profile: str | None = None,
    required_sections: Sequence[str] = ("encoder", "model", "optimization", "pair_sampling"),
) -> JsonConfig:
    if "defaults" in config:
        resolved = _resolve_profiled_config(config, profile=profile)
    elif "profiles" in config:
        resolved = _resolve_schema_v2_config(config, profile=profile)
    else:
        resolved = dict(config)
        if profile is not None:
            resolved["profile"] = profile
        resolved.setdefault("schema_version", 1)

    _validate_required_sections(resolved, required_sections)
    if "pair_sampling" in resolved:
        _ = negative_sampling_config_from_training_config(resolved)
    if "optimization" in resolved:
        _ = trainable_training_config_from_training_config(resolved)
    return resolved


def negative_sampling_config_from_training_config(config: JsonConfig) -> NegativeSamplingConfig:
    pair_sampling = _required_section(config, "pair_sampling")
    sampling_config = NegativeSamplingConfig(
        random_seed=_int_value(pair_sampling, "random_seed"),
        easy_random_per_positive=_int_value(pair_sampling, "easy_random_per_positive"),
        hard_bm25_per_positive=_int_value(pair_sampling, "hard_bm25_per_positive"),
        hard_dense_per_positive=_int_value(pair_sampling, "hard_dense_per_positive"),
        hard_graph_neighbor_per_positive=_int_value(pair_sampling, "hard_graph_neighbor_per_positive"),
        hard_pool_size=_int_value(pair_sampling, "hard_pool_size"),
    )
    validate_negative_sampling_config(sampling_config)
    return sampling_config


def trainable_training_config_from_training_config(config: JsonConfig) -> TrainableTrainingConfig:
    optimization = _required_section(config, "optimization")
    training_config = TrainableTrainingConfig(
        optimizer_name=str(optimization.get("optimizer", optimization.get("optimizer_name", "AdamW"))),
        learning_rate=_float_value(optimization, "learning_rate"),
        batch_size=_int_value(optimization, "batch_size"),
        max_grad_norm=_float_value(optimization, "max_grad_norm"),
        random_seed=_int_value(optimization, "random_seed"),
        pos_weight_enabled=_bool_value(
            optimization,
            "pos_weight" if "pos_weight" in optimization else "pos_weight_enabled",
        ),
        epochs=_int_value(optimization, "epochs"),
    )
    validate_trainable_training_config(training_config)
    return training_config


def encoder_config_from_training_config(config: JsonConfig) -> EncoderConfig:
    encoder = _required_section(config, "encoder")
    return {
        "model": _string_value(encoder, "model"),
        "query_prefix": _string_value(encoder, "query_prefix"),
        "passage_prefix": _string_value(encoder, "passage_prefix"),
    }


def model_config_values_from_training_config(config: JsonConfig) -> ModelConfigValues:
    model = _required_section(config, "model")
    return {
        "hidden_dim": _int_value(model, "hidden_dim"),
        "num_layers": _int_value(model, "num_layers"),
        "dropout": _float_value(model, "dropout"),
        "ablation": _string_value(model, "ablation"),
    }


def device_from_training_config(config: JsonConfig, *, default: str = "cpu") -> str:
    optimization = config.get("optimization")
    if not isinstance(optimization, dict):
        return default
    device = optimization.get("device", default)
    if not isinstance(device, str) or not device:
        raise ValueError("Training config optimization.device must be a non-empty string.")
    return device


def _resolve_profiled_config(config: JsonConfig, *, profile: str | None) -> JsonConfig:
    method = config.get("method")
    if not isinstance(method, str) or not method:
        raise ValueError("Training config requires a non-empty method.")
    defaults = _required_section(config, "defaults")
    profile_name = profile or str(config.get("default_profile", "quick"))
    profiles = config.get("profiles", {})
    if not isinstance(profiles, dict):
        raise ValueError("Training config profiles must be an object.")
    if profile_name not in profiles:
        raise ValueError(f"Unknown training config profile: {profile_name}")
    profile_config = profiles[profile_name]
    if not isinstance(profile_config, dict):
        raise ValueError(f"Training config profile must be an object: {profile_name}")
    resolved = merge_config(defaults, profile_config)
    return {
        "schema_version": config.get("schema_version", 1),
        "method": method,
        "profile": profile_name,
        **resolved,
    }


def _resolve_schema_v2_config(config: JsonConfig, *, profile: str | None) -> JsonConfig:
    method = config.get("method")
    if not isinstance(method, str) or not method:
        raise ValueError("Training config requires a non-empty method.")
    profile_name = profile or str(config.get("default_profile", "quick"))
    profiles = config.get("profiles", {})
    if not isinstance(profiles, dict):
        raise ValueError("Training config profiles must be an object.")
    if profile_name not in profiles:
        raise ValueError(f"Unknown training config profile: {profile_name}")
    profile_config = profiles[profile_name]
    if not isinstance(profile_config, dict):
        raise ValueError(f"Training config profile must be an object: {profile_name}")
    base = {
        key: value
        for key, value in config.items()
        if key not in {"default_profile", "profiles"}
    }
    resolved = merge_config(base, profile_config)
    return _compat_training_config_from_schema_v2(resolved, profile_name=profile_name)


def _compat_training_config_from_schema_v2(config: JsonConfig, *, profile_name: str) -> JsonConfig:
    return {
        "schema_version": config.get("schema_version", 2),
        "method": config["method"],
        "profile": profile_name,
        "encoder": _compat_encoder_from_schema_v2(_required_section(config, "encoder")),
        "model": dict(_required_section(config, "model")),
        "optimization": _compat_optimization_from_schema_v2(_required_section(config, "trainer")),
        "pair_sampling": dict(_required_section(config, "pairs")),
        "reporting": dict(config.get("reporting", {})),
        "selection": dict(config.get("selection", {})),
    }


def _compat_encoder_from_schema_v2(encoder: JsonConfig) -> JsonConfig:
    value = dict(encoder)
    model = value.pop("model_name", None)
    if model is not None:
        value["model"] = model
    return value


def _compat_optimization_from_schema_v2(trainer: JsonConfig) -> JsonConfig:
    value = dict(trainer)
    optimizer = value.pop("optimizer_name", None)
    if optimizer is not None:
        value["optimizer"] = optimizer
    pos_weight = value.pop("pos_weight_enabled", None)
    if pos_weight is not None:
        value["pos_weight"] = pos_weight
    return value


def _validate_required_sections(config: JsonConfig, required_sections: Sequence[str]) -> None:
    method = config.get("method")
    if not isinstance(method, str) or not method:
        raise ValueError("Training config requires a non-empty method.")
    for section in required_sections:
        _required_section(config, section)


def _required_section(config: JsonConfig, key: str) -> JsonConfig:
    section = config.get(key)
    if not isinstance(section, dict):
        raise ValueError(f"Training config requires object section: {key}")
    return section


def _int_value(config: JsonConfig, key: str) -> int:
    value = config.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"Training config field must be an integer: {key}")
    return value


def _float_value(config: JsonConfig, key: str) -> float:
    value = config.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"Training config field must be numeric: {key}")
    return float(value)


def _bool_value(config: JsonConfig, key: str) -> bool:
    value = config.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"Training config field must be boolean: {key}")
    return value


def _string_value(config: JsonConfig, key: str) -> str:
    value = config.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Training config field must be a non-empty string: {key}")
    return value

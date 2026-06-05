from __future__ import annotations

import types
from collections.abc import Mapping
from dataclasses import MISSING, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Literal, get_args, get_origin, get_type_hints


class ConfigConverter:
    def structure(self, value: Any, target_type: type[Any] | object) -> Any:
        return _structure_value(value, target_type)

    def unstructure(self, value: Any) -> Any:
        return _unstructure_value(value)


def _structure_value(value: Any, target_type: type[Any] | object) -> Any:
    if target_type is Any or target_type is object:
        return value
    if value is None:
        if _allows_none(target_type):
            return None
        raise ValueError(f"Expected {target_type}, got null.")

    origin = get_origin(target_type)
    args = get_args(target_type)

    if origin is Literal:
        for allowed in args:
            if value == allowed:
                return allowed
        raise ValueError(f"Expected one of {args!r}, got {value!r}.")
    if origin in (types.UnionType, None) and isinstance(target_type, types.UnionType):
        return _structure_union(value, args)
    if origin is not None and str(origin) == "typing.Union":
        return _structure_union(value, args)
    if origin is tuple:
        return _structure_tuple(value, args)
    if origin is list:
        (item_type,) = args or (Any,)
        if not isinstance(value, list):
            raise ValueError(f"Expected list for {target_type}, got {type(value).__name__}.")
        return [_structure_value(item, item_type) for item in value]
    if origin is dict:
        key_type, value_type = args or (Any, Any)
        if not isinstance(value, Mapping):
            raise ValueError(f"Expected object for {target_type}, got {type(value).__name__}.")
        return {
            _structure_value(key, key_type): _structure_value(item, value_type)
            for key, item in value.items()
        }

    if target_type is Path:
        if not isinstance(value, str | Path):
            raise ValueError(f"Expected path string, got {type(value).__name__}.")
        return Path(value)
    if isinstance(target_type, type) and issubclass(target_type, Enum):
        return target_type(value)
    if isinstance(target_type, type) and is_dataclass(target_type):
        return _structure_dataclass(value, target_type)
    if target_type in (str, int, float, bool):
        if not isinstance(value, target_type):
            raise ValueError(f"Expected {target_type.__name__}, got {type(value).__name__}.")
        return value
    return value


def _structure_union(value: Any, variants: tuple[object, ...]) -> Any:
    non_none_variants = tuple(variant for variant in variants if variant is not type(None))
    if value is None and len(non_none_variants) != len(variants):
        return None
    dataclass_variants = tuple(
        variant for variant in non_none_variants if isinstance(variant, type) and is_dataclass(variant)
    )
    if dataclass_variants and isinstance(value, Mapping):
        matched = _match_discriminated_dataclass(value, dataclass_variants)
        if matched is not None:
            return _structure_value(value, matched)
    errors: list[Exception] = []
    for variant in non_none_variants:
        try:
            return _structure_value(value, variant)
        except (TypeError, ValueError) as error:
            errors.append(error)
    detail = "; ".join(str(error) for error in errors)
    raise ValueError(f"Value does not match any union variant: {detail}")


def _match_discriminated_dataclass(value: Mapping[str, Any], variants: tuple[type[Any], ...]) -> type[Any] | None:
    method = value.get("method")
    if method is None:
        return None
    for variant in variants:
        annotations = get_type_hints(variant)
        method_annotation = annotations.get("method")
        if get_origin(method_annotation) is Literal and method in get_args(method_annotation):
            return variant
    return None


def _structure_tuple(value: Any, args: tuple[object, ...]) -> tuple[Any, ...]:
    if not isinstance(value, list | tuple):
        raise ValueError(f"Expected array for tuple, got {type(value).__name__}.")
    if len(args) == 2 and args[1] is Ellipsis:
        return tuple(_structure_value(item, args[0]) for item in value)
    if len(value) != len(args):
        raise ValueError(f"Expected tuple of length {len(args)}, got {len(value)}.")
    return tuple(_structure_value(item, item_type) for item, item_type in zip(value, args))


def _structure_dataclass(value: Any, target_type: type[Any]) -> Any:
    if not isinstance(value, Mapping):
        raise ValueError(f"Expected object for {target_type.__name__}, got {type(value).__name__}.")
    field_map = {field.name: field for field in fields(target_type)}
    type_hints = get_type_hints(target_type)
    unknown_fields = sorted(set(value) - set(field_map))
    if unknown_fields:
        raise ValueError(f"{target_type.__name__} contains unsupported fields: {unknown_fields}.")

    init_values: dict[str, Any] = {}
    for name, field in field_map.items():
        if name in value:
            init_values[name] = _structure_value(value[name], type_hints[name])
        elif field.default is MISSING and field.default_factory is MISSING:
            raise ValueError(f"{target_type.__name__} requires field: {name}")
    return target_type(**init_values)


def _unstructure_value(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _unstructure_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, tuple | list):
        return [_unstructure_value(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _unstructure_value(item) for key, item in value.items()}
    return value


def _allows_none(target_type: object) -> bool:
    if target_type is None or target_type is type(None):
        return True
    args = get_args(target_type)
    return any(arg is type(None) for arg in args)


__all__ = ["ConfigConverter"]

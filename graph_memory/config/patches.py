from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any, TypeAlias

ConfigPatch: TypeAlias = Mapping[str, Any]


def deep_merge_patch(base: Mapping[str, Any], patch: Mapping[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = deepcopy(dict(base))
    for key, value in patch.items():
        existing = merged.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            merged[key] = deep_merge_patch(existing, value)
        else:
            merged[key] = deepcopy(value)
    return merged


__all__ = ["ConfigPatch", "deep_merge_patch"]

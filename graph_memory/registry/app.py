from __future__ import annotations

from dataclasses import dataclass

from graph_memory.registry.methods import MethodRegistry, build_method_registry


@dataclass(frozen=True)
class AppRegistry:
    methods: MethodRegistry


Registry = AppRegistry(methods=build_method_registry())

__all__ = ["AppRegistry", "Registry"]

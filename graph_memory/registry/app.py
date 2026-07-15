from __future__ import annotations

from dataclasses import dataclass

from graph_memory.registry.methods import MethodRegistry, build_method_registry
from graph_memory.registry.retrieval import RetrievalRegistry
from graph_memory.registry.retrieval_builders import build_retrieval_registry


@dataclass(frozen=True)
class AppRegistry:
    methods: MethodRegistry
    retrieval: RetrievalRegistry


_METHODS = build_method_registry()
Registry = AppRegistry(
    methods=_METHODS,
    retrieval=build_retrieval_registry(_METHODS),
)

__all__ = ["AppRegistry", "Registry"]

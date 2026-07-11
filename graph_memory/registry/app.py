from __future__ import annotations

from dataclasses import dataclass

from graph_memory.registry.methods import MethodRegistry, build_method_registry
from graph_memory.registry.retrieval import RetrievalRegistry
from graph_memory.registry.retrieval_builders import build_retrieval_registry


@dataclass(frozen=True)
class AppRegistry:
    methods: MethodRegistry
    retrieval: RetrievalRegistry


Registry = AppRegistry(
    methods=build_method_registry(),
    retrieval=build_retrieval_registry(),
)

__all__ = ["AppRegistry", "Registry"]

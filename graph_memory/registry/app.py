from __future__ import annotations

from dataclasses import dataclass

from graph_memory.registry.methods import MethodRegistry, build_method_registry
from graph_memory.registry.retrieval import RetrievalRegistry
from graph_memory.registry.retrieval_builders import build_retrieval_registry
from graph_memory.registry.stage_configs import StageConfigRegistry, build_stage_config_registry
from graph_memory.registry.training import TrainingRegistry, build_training_registry


@dataclass(frozen=True)
class AppRegistry:
    configs: StageConfigRegistry
    methods: MethodRegistry
    retrieval: RetrievalRegistry
    training: TrainingRegistry


Registry = AppRegistry(
    configs=build_stage_config_registry(),
    methods=build_method_registry(),
    retrieval=build_retrieval_registry(),
    training=build_training_registry(),
)

__all__ = ["AppRegistry", "Registry"]

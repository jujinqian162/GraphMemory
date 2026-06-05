from __future__ import annotations

from dataclasses import dataclass

from graph_memory.contracts.graphs import MemoryGraph
from graph_memory.contracts.tasks import MemoryTaskInput, MemoryTaskLabels
from graph_memory.contracts.training_pairs import TrainPairRecord
from graph_memory.models.graph_retriever.contracts import TextEmbeddingProvider
from graph_memory.models.graph_retriever.training import TrainableTrainingResult
from graph_memory.registry import Registry
from graph_memory.registry.stage_configs import TrainStageConfig
from graph_memory.registry.training import TrainDependencies
from graph_memory.retrieval.signals import SeedSignalProvider


@dataclass(frozen=True)
class TrainStageResult:
    result: TrainableTrainingResult


def run_train_stage(
    config: TrainStageConfig,
    *,
    train_task_inputs: list[MemoryTaskInput],
    train_graphs: list[MemoryGraph],
    train_pairs: list[TrainPairRecord],
    dev_task_inputs: list[MemoryTaskInput],
    dev_labels: list[MemoryTaskLabels],
    dev_graphs: list[MemoryGraph],
    text_embedding_provider: TextEmbeddingProvider,
    seed_signal_provider: SeedSignalProvider,
) -> TrainStageResult:
    trainer = Registry.training.build(
        config.job,
        TrainDependencies(
            text_embedding_provider=text_embedding_provider,
            seed_signal_provider=seed_signal_provider,
        ),
    )
    return TrainStageResult(
        result=trainer.train(
            train_task_inputs=train_task_inputs,
            train_graphs=train_graphs,
            train_pairs=train_pairs,
            dev_task_inputs=dev_task_inputs,
            dev_labels=dev_labels,
            dev_graphs=dev_graphs,
        )
    )


__all__ = ["TrainStageResult", "run_train_stage"]

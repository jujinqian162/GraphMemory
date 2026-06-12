from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, TypeAlias, cast

from graph_memory.contracts.graphs import MemoryGraph
from graph_memory.contracts.tasks import MemoryTaskInput, MemoryTaskLabels
from graph_memory.contracts.training_pairs import TrainPairRecord
from graph_memory.registry.conversions import rgcn_training_config_from_trainer_settings
from graph_memory.registry.method_configs import (
    DenseFinetuneMethodSettings,
    RgcnMethodSettings,
    TrainJobSettings,
)
from graph_memory.models.dense_finetune.training import (
    DenseFinetuneRunConfig,
    DenseFinetuneTrainingResult,
    train_dense_finetune,
)

if TYPE_CHECKING:
    from graph_memory.models.graph_retriever.contracts import TextEmbeddingProvider
    from graph_memory.models.graph_retriever.training import RgcnTrainingResult
    from graph_memory.retrieval.signals import SeedSignalProvider


@dataclass(frozen=True)
class TrainDependencies:
    text_embedding_provider: "TextEmbeddingProvider"
    seed_signal_provider: "SeedSignalProvider"


@dataclass(frozen=True)
class RgcnTrainPayload:
    train_task_inputs: list[MemoryTaskInput]
    train_graphs: list[MemoryGraph]
    train_pairs: list[TrainPairRecord]
    dev_task_inputs: list[MemoryTaskInput]
    dev_labels: list[MemoryTaskLabels]
    dev_graphs: list[MemoryGraph]
    train_labels: list[MemoryTaskLabels] | None = None
    dependencies: TrainDependencies | None = None


@dataclass(frozen=True)
class DenseFinetuneTrainPayload:
    train_task_inputs: list[MemoryTaskInput]
    train_labels: list[MemoryTaskLabels]
    train_pairs: list[TrainPairRecord]
    dev_task_inputs: list[MemoryTaskInput]
    dev_labels: list[MemoryTaskLabels]
    output_dir: Path
    model_dir: Path


TrainPayload: TypeAlias = RgcnTrainPayload | DenseFinetuneTrainPayload


class TrainMethodTrainer(Protocol):
    def train(self, payload: TrainPayload) -> object:
        ...


@dataclass(frozen=True)
class TrainingBuilderSpec:
    settings_type: type[object]
    build: Callable[[TrainJobSettings], TrainMethodTrainer]


@dataclass(frozen=True)
class TrainingRegistry:
    builders: dict[type[object], TrainingBuilderSpec]

    def build(self, settings: TrainJobSettings) -> TrainMethodTrainer:
        try:
            spec = self.builders[type(settings)]
        except KeyError as error:
            raise ValueError(f"Unsupported train settings type: {type(settings).__name__}") from error
        return spec.build(settings)


@dataclass(frozen=True)
class RgcnGraphRetrieverTrainer:
    settings: RgcnMethodSettings

    def train(self, payload: TrainPayload) -> "RgcnTrainingResult":
        from graph_memory.models.graph_retriever.config.defaults import default_model_config
        from graph_memory.models.graph_retriever.training import train_graph_retriever

        if not isinstance(payload, RgcnTrainPayload):
            raise TypeError(f"R-GCN trainer expected RgcnTrainPayload, got {type(payload).__name__}.")
        deps = payload.dependencies or _build_rgcn_dependencies(self.settings)
        model_config = default_model_config(
            encoder_model=self.settings.encoder.model_name,
            encoder_dim=deps.text_embedding_provider.embedding_dim,
            query_prefix=self.settings.encoder.query_prefix,
            passage_prefix=self.settings.encoder.passage_prefix,
            encoder_batch_size=self.settings.encoder.batch_size,
            hidden_dim=self.settings.model.hidden_dim,
            num_layers=self.settings.model.num_layers,
            dropout=self.settings.model.dropout,
            ablation_name=self.settings.model.ablation,
        )
        return train_graph_retriever(
            train_task_inputs=payload.train_task_inputs,
            train_graphs=payload.train_graphs,
            train_pairs=payload.train_pairs,
            train_labels=payload.train_labels,
            dev_task_inputs=payload.dev_task_inputs,
            dev_labels=payload.dev_labels,
            dev_graphs=payload.dev_graphs,
            model_config=model_config,
            training_config=rgcn_training_config_from_trainer_settings(self.settings.trainer),
            text_embedding_provider=deps.text_embedding_provider,
            seed_signal_provider=deps.seed_signal_provider,
            device=self.settings.trainer.device,
        )


@dataclass(frozen=True)
class DenseFinetuneMethodTrainer:
    settings: DenseFinetuneMethodSettings

    def train(self, payload: TrainPayload) -> DenseFinetuneTrainingResult:
        if not isinstance(payload, DenseFinetuneTrainPayload):
            raise TypeError(f"Dense-ft trainer expected DenseFinetuneTrainPayload, got {type(payload).__name__}.")
        return train_dense_finetune(
            config=DenseFinetuneRunConfig(
                base_model=self.settings.encoder.model_name,
                query_prefix=self.settings.encoder.query_prefix,
                passage_prefix=self.settings.encoder.passage_prefix,
                batch_size=self.settings.encoder.batch_size,
                data=self.settings.data,
                trainer=self.settings.trainer,
                selection=self.settings.selection,
            ),
            train_task_inputs=payload.train_task_inputs,
            train_pairs=payload.train_pairs,
            dev_task_inputs=payload.dev_task_inputs,
            dev_labels=payload.dev_labels,
            output_dir=payload.output_dir,
            model_dir=payload.model_dir,
        )


def _build_rgcn_dependencies(settings: RgcnMethodSettings) -> TrainDependencies:
    from graph_memory.models.graph_retriever.text_embeddings import DenseGraphFeatureProvider

    text_embedding_provider = DenseGraphFeatureProvider(
        model_name=settings.encoder.model_name,
        query_prefix=settings.encoder.query_prefix,
        passage_prefix=settings.encoder.passage_prefix,
        batch_size=settings.encoder.batch_size,
    )
    return TrainDependencies(
        text_embedding_provider=text_embedding_provider,
        seed_signal_provider=text_embedding_provider,
    )


def build_training_registry() -> TrainingRegistry:
    return TrainingRegistry(
        builders={
            RgcnMethodSettings: TrainingBuilderSpec(
                RgcnMethodSettings,
                lambda settings: RgcnGraphRetrieverTrainer(cast(RgcnMethodSettings, settings)),
            ),
            DenseFinetuneMethodSettings: TrainingBuilderSpec(
                DenseFinetuneMethodSettings,
                lambda settings: DenseFinetuneMethodTrainer(cast(DenseFinetuneMethodSettings, settings)),
            ),
        }
    )


__all__ = [
    "TrainDependencies",
    "TrainingRegistry",
    "build_training_registry",
]

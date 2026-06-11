from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol, TypeAlias, cast

from graph_memory.contracts.graphs import MemoryGraph
from graph_memory.contracts.tasks import MemoryTaskInput, MemoryTaskLabels
from graph_memory.contracts.training_pairs import TrainPairRecord
from graph_memory.registry.conversions import trainable_training_config_from_trainer_settings
from graph_memory.registry.retrieval import DenseEncoderSettings, RetrievalMethodId
from graph_memory.training_pairs.config import NegativeSamplingConfig

if TYPE_CHECKING:
    from graph_memory.models.graph_retriever.contracts import TextEmbeddingProvider
    from graph_memory.models.graph_retriever.training import TrainableTrainingResult
    from graph_memory.retrieval.signals import SeedSignalProvider


@dataclass(frozen=True)
class RgcnModelSettings:
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.1
    ablation: str = "full_rgcn"


@dataclass(frozen=True)
class RgcnTrainerSettings:
    optimizer_name: str = "AdamW"
    learning_rate: float = 1e-4
    batch_size: int = 1
    max_grad_norm: float = 1.0
    random_seed: int = 13
    pos_weight_enabled: bool = False
    epochs: int = 1
    device: str = "cpu"


@dataclass(frozen=True)
class RgcnPairSamplingSettings:
    random_seed: int = NegativeSamplingConfig().random_seed
    easy_random_per_positive: int = NegativeSamplingConfig().easy_random_per_positive
    hard_bm25_per_positive: int = NegativeSamplingConfig().hard_bm25_per_positive
    hard_dense_per_positive: int = NegativeSamplingConfig().hard_dense_per_positive
    hard_graph_neighbor_per_positive: int = NegativeSamplingConfig().hard_graph_neighbor_per_positive
    hard_pool_size: int = NegativeSamplingConfig().hard_pool_size


@dataclass(frozen=True)
class TrainingReportingSettings:
    render_training_curves: bool = True


@dataclass(frozen=True)
class ModelSelectionSettings:
    best_metric: str = "dev_composite"
    higher_is_better: bool = True


@dataclass(frozen=True)
class RgcnMethodSettings:
    encoder: DenseEncoderSettings
    model: RgcnModelSettings
    trainer: RgcnTrainerSettings
    pairs: RgcnPairSamplingSettings = field(default_factory=RgcnPairSamplingSettings)
    reporting: TrainingReportingSettings = field(default_factory=TrainingReportingSettings)
    selection: ModelSelectionSettings = field(default_factory=ModelSelectionSettings)
    method: Literal[RetrievalMethodId.DENSE_RGCN_GRAPH_RETRIEVER] = RetrievalMethodId.DENSE_RGCN_GRAPH_RETRIEVER


TrainJobSettings: TypeAlias = RgcnMethodSettings


@dataclass(frozen=True)
class TrainDependencies:
    text_embedding_provider: "TextEmbeddingProvider"
    seed_signal_provider: "SeedSignalProvider"


class TrainMethodTrainer(Protocol):
    def train(
        self,
        *,
        train_task_inputs: list[MemoryTaskInput],
        train_graphs: list[MemoryGraph],
        train_pairs: list[TrainPairRecord],
        train_labels: list[MemoryTaskLabels] | None = None,
        dev_task_inputs: list[MemoryTaskInput],
        dev_labels: list[MemoryTaskLabels],
        dev_graphs: list[MemoryGraph],
    ) -> "TrainableTrainingResult":
        ...


@dataclass(frozen=True)
class TrainingBuilderSpec:
    settings_type: type[object]
    build: Callable[[TrainJobSettings, TrainDependencies], TrainMethodTrainer]


@dataclass(frozen=True)
class TrainingRegistry:
    builders: dict[type[object], TrainingBuilderSpec]

    def build(self, settings: TrainJobSettings, deps: TrainDependencies) -> TrainMethodTrainer:
        try:
            spec = self.builders[type(settings)]
        except KeyError as error:
            raise ValueError(f"Unsupported train settings type: {type(settings).__name__}") from error
        return spec.build(settings, deps)


@dataclass(frozen=True)
class RgcnGraphRetrieverTrainer:
    settings: RgcnMethodSettings
    deps: TrainDependencies

    def train(
        self,
        *,
        train_task_inputs: list[MemoryTaskInput],
        train_graphs: list[MemoryGraph],
        train_pairs: list[TrainPairRecord],
        train_labels: list[MemoryTaskLabels] | None = None,
        dev_task_inputs: list[MemoryTaskInput],
        dev_labels: list[MemoryTaskLabels],
        dev_graphs: list[MemoryGraph],
    ) -> "TrainableTrainingResult":
        from graph_memory.models.graph_retriever.config.defaults import default_model_config
        from graph_memory.models.graph_retriever.training import train_graph_retriever

        model_config = default_model_config(
            encoder_model=self.settings.encoder.model_name,
            encoder_dim=self.deps.text_embedding_provider.embedding_dim,
            query_prefix=self.settings.encoder.query_prefix,
            passage_prefix=self.settings.encoder.passage_prefix,
            hidden_dim=self.settings.model.hidden_dim,
            num_layers=self.settings.model.num_layers,
            dropout=self.settings.model.dropout,
            ablation_name=self.settings.model.ablation,
        )
        return train_graph_retriever(
            train_task_inputs=train_task_inputs,
            train_graphs=train_graphs,
            train_pairs=train_pairs,
            train_labels=train_labels,
            dev_task_inputs=dev_task_inputs,
            dev_labels=dev_labels,
            dev_graphs=dev_graphs,
            model_config=model_config,
            training_config=trainable_training_config_from_trainer_settings(self.settings.trainer),
            text_embedding_provider=self.deps.text_embedding_provider,
            seed_signal_provider=self.deps.seed_signal_provider,
            device=self.settings.trainer.device,
        )


def build_training_registry() -> TrainingRegistry:
    return TrainingRegistry(
        builders={
            RgcnMethodSettings: TrainingBuilderSpec(
                RgcnMethodSettings,
                lambda settings, deps: RgcnGraphRetrieverTrainer(cast(RgcnMethodSettings, settings), deps),
            )
        }
    )


__all__ = [
    "ModelSelectionSettings",
    "RgcnGraphRetrieverTrainer",
    "RgcnMethodSettings",
    "RgcnModelSettings",
    "RgcnPairSamplingSettings",
    "RgcnTrainerSettings",
    "TrainDependencies",
    "TrainJobSettings",
    "TrainMethodTrainer",
    "TrainingBuilderSpec",
    "TrainingRegistry",
    "TrainingReportingSettings",
    "build_training_registry",
]

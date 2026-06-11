from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, TypeAlias, cast

from graph_memory.contracts.graphs import MemoryGraph
from graph_memory.contracts.tasks import MemoryTaskInput, MemoryTaskLabels
from graph_memory.contracts.training_pairs import TrainPairRecord
from graph_memory.registry.conversions import trainable_training_config_from_trainer_settings
from graph_memory.registry.retrieval import DenseEncoderSettings, RetrievalMethodId
from graph_memory.training_pairs.config import NegativeSamplingConfig
from graph_memory.models.dense_finetune.training import (
    DenseFinetuneRunConfig,
    DenseFinetuneSelectionSettings,
    DenseFinetuneTrainerSettings,
    DenseFinetuneTrainingResult,
    train_dense_finetune,
)
from graph_memory.models.dense_finetune.contracts import DenseFinetuneDataSettings

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


@dataclass(frozen=True)
class DenseFinetuneMethodSettings:
    encoder: DenseEncoderSettings
    data: DenseFinetuneDataSettings = field(default_factory=DenseFinetuneDataSettings)
    trainer: DenseFinetuneTrainerSettings = field(default_factory=DenseFinetuneTrainerSettings)
    selection: DenseFinetuneSelectionSettings = field(default_factory=DenseFinetuneSelectionSettings)
    method: Literal[RetrievalMethodId.DENSE_FT] = RetrievalMethodId.DENSE_FT


TrainJobSettings: TypeAlias = RgcnMethodSettings | DenseFinetuneMethodSettings

_TRAINING_CONFIG_REQUIRED_SECTIONS: dict[str, tuple[str, ...]] = {
    RetrievalMethodId.DENSE_RGCN_GRAPH_RETRIEVER.value: (
        "encoder",
        "model",
        "optimization",
        "pair_sampling",
    ),
    RetrievalMethodId.DENSE_FT.value: (
        "encoder",
        "pair_sampling",
        "data",
        "trainer",
        "selection",
    ),
}


def training_config_required_sections(method: str) -> tuple[str, ...]:
    try:
        return _TRAINING_CONFIG_REQUIRED_SECTIONS[method]
    except KeyError as error:
        raise ValueError(f"Unsupported training method: {method}") from error


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

    def train(self, payload: TrainPayload) -> "TrainableTrainingResult":
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
            training_config=trainable_training_config_from_trainer_settings(self.settings.trainer),
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
    "ModelSelectionSettings",
    "RgcnMethodSettings",
    "RgcnModelSettings",
    "RgcnPairSamplingSettings",
    "RgcnTrainerSettings",
    "TrainDependencies",
    "TrainJobSettings",
    "TrainingRegistry",
    "TrainingReportingSettings",
    "build_training_registry",
]

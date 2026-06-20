from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from graph_memory.models.dense_finetune.training import (
    DenseFinetuneRunConfig,
    DenseFinetuneTrainingResult,
    train_dense_finetune,
)
from graph_memory.registry.conversions import rgcn_training_config_from_trainer_settings
from graph_memory.registry.method_configs import DenseFinetuneMethodSettings, RgcnMethodSettings
from graph_memory.stages.train_payloads import DenseFinetuneTrainPayload, RgcnTrainPayload, TrainDependencies, TrainPayload

if TYPE_CHECKING:
    from graph_memory.models.graph_retriever.training import RgcnTrainingResult


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
            train_requests=payload.train_requests,
            train_graphs=payload.train_graphs,
            train_pairs=payload.train_pairs,
            train_labels=payload.train_labels,
            dev_requests=payload.dev_requests,
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
            train_requests=payload.train_requests,
            train_pairs=payload.train_pairs,
            dev_requests=payload.dev_requests,
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


__all__ = [
    "DenseFinetuneMethodTrainer",
    "RgcnGraphRetrieverTrainer",
]

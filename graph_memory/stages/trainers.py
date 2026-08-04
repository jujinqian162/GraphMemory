from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from graph_memory.experiment.config import (
    DenseEncoderConfig,
    DenseFinetuneStageConfig,
    RgcnTrainConfig,
)
from graph_memory.models.frozen_embeddings import FrozenTaskEmbeddings
from graph_memory.models.dense_finetune.training import (
    DenseFinetuneRunConfig,
    DenseFinetuneTrainingResult,
    train_dense_finetune,
)
from graph_memory.registry.retrieval import DenseEncoderSettings
from graph_memory.stages.train_payloads import (
    DenseFinetuneTrainPayload,
    RgcnTrainPayload,
    TrainDependencies,
    TrainPayload,
)

if TYPE_CHECKING:
    from graph_memory.models.graph_retriever.training import RgcnTrainingResult


@dataclass(frozen=True)
class RgcnGraphRetrieverTrainer:
    method: str
    encoder: DenseEncoderConfig
    train_config: RgcnTrainConfig
    seed_checkpoint: Path | None = None
    train_embeddings: Mapping[str, FrozenTaskEmbeddings] | None = None
    dev_embeddings: Mapping[str, FrozenTaskEmbeddings] | None = None

    def train(self, payload: TrainPayload) -> RgcnTrainingResult:
        from graph_memory.models.graph_retriever.config.defaults import (
            default_model_config,
        )
        from graph_memory.models.graph_retriever.training import train_graph_retriever

        if not isinstance(payload, RgcnTrainPayload):
            raise TypeError(
                f"R-GCN trainer expected RgcnTrainPayload, got {type(payload).__name__}."
            )
        settings = self.train_config
        encoder_settings = _effective_rgcn_encoder_settings(
            self.encoder, self.seed_checkpoint
        )
        if (self.train_embeddings is None) != (self.dev_embeddings is None):
            raise ValueError("Evidence R-GCN requires both train and dev embeddings.")
        if self.train_embeddings is None:
            train_deps = _build_rgcn_dependencies(
                encoder_settings, device=settings.trainer.device
            )
            dev_deps = train_deps
        else:
            from graph_memory.models.graph_retriever.text_embeddings import (
                PrecomputedGraphFeatureProvider,
            )

            embedding_dim = next(iter(self.train_embeddings.values())).values.shape[1]
            train_provider = PrecomputedGraphFeatureProvider(
                self.train_embeddings, embedding_dim=embedding_dim
            )
            dev_provider = PrecomputedGraphFeatureProvider(
                self.dev_embeddings or {}, embedding_dim=embedding_dim
            )
            train_deps = TrainDependencies(train_provider, train_provider)
            dev_deps = TrainDependencies(dev_provider, dev_provider)
        model_config = default_model_config(
            method_name=self.method,
            encoder_model=encoder_settings.model_name,
            encoder_dim=train_deps.text_embedding_provider.embedding_dim,
            query_prefix=encoder_settings.query_prefix,
            passage_prefix=encoder_settings.passage_prefix,
            encoder_batch_size=encoder_settings.batch_size,
            hidden_dim=settings.model.hidden_dim,
            num_layers=settings.model.num_layers,
            dropout=settings.model.dropout,
            ablation_name=settings.model.ablation,
        )
        return train_graph_retriever(
            train_requests=list(payload.train_requests),
            train_graphs=list(payload.train_graphs),
            train_labels=list(payload.train_labels),
            train_pairs=list(payload.train_pairs),
            dev_requests=list(payload.dev_requests),
            dev_labels=list(payload.dev_labels),
            dev_graphs=list(payload.dev_graphs),
            model_config=model_config,
            training_config=settings.trainer,
            text_embedding_provider=train_deps.text_embedding_provider,
            seed_signal_provider=train_deps.seed_signal_provider,
            dev_text_embedding_provider=dev_deps.text_embedding_provider,
            dev_seed_signal_provider=dev_deps.seed_signal_provider,
            selection_settings=settings.selection,
            device=settings.trainer.device,
        )


@dataclass(frozen=True)
class DenseFinetuneMethodTrainer:
    config: DenseFinetuneStageConfig

    def train(self, payload: TrainPayload) -> DenseFinetuneTrainingResult:
        if not isinstance(payload, DenseFinetuneTrainPayload):
            raise TypeError(
                "Dense-ft trainer expected DenseFinetuneTrainPayload, "
                f"got {type(payload).__name__}."
            )
        settings = self.config.train
        encoder = self.config.encoder
        return train_dense_finetune(
            config=DenseFinetuneRunConfig(
                base_model=encoder.model_name,
                query_prefix=encoder.query_prefix,
                passage_prefix=encoder.passage_prefix,
                batch_size=encoder.batch_size,
                data=settings.data,
                trainer=settings.trainer,
                selection=settings.selection,
            ),
            train_requests=payload.train_requests,
            train_pairs=payload.train_pairs,
            train_group_ids=payload.train_group_ids,
            dev_requests=payload.dev_requests,
            dev_labels=payload.dev_labels,
            dev_query_origins=payload.dev_query_origins,
            output_dir=payload.output_dir,
            model_dir=payload.model_dir,
        )



def _effective_rgcn_encoder_settings(
    settings: DenseEncoderConfig,
    seed_checkpoint: Path | None,
) -> DenseEncoderSettings:
    if seed_checkpoint is None:
        return DenseEncoderSettings(
            model_name=settings.model_name,
            query_prefix=settings.query_prefix,
            passage_prefix=settings.passage_prefix,
            batch_size=settings.batch_size,
        )
    from graph_memory.models.dense_finetune.metadata import load_dense_ft_model_metadata

    metadata = load_dense_ft_model_metadata(seed_checkpoint)
    return DenseEncoderSettings(
        model_name=str(seed_checkpoint),
        query_prefix=metadata.query_prefix,
        passage_prefix=metadata.passage_prefix,
        batch_size=metadata.batch_size,
    )


def _build_rgcn_dependencies(
    encoder_settings: DenseEncoderSettings,
    *,
    device: str,
) -> TrainDependencies:
    from graph_memory.models.graph_retriever.text_embeddings import (
        DenseGraphFeatureProvider,
    )

    provider = DenseGraphFeatureProvider(
        model_name=encoder_settings.model_name,
        query_prefix=encoder_settings.query_prefix,
        passage_prefix=encoder_settings.passage_prefix,
        batch_size=encoder_settings.batch_size,
        device=device,
    )
    return TrainDependencies(
        text_embedding_provider=provider,
        seed_signal_provider=provider,
    )


__all__ = [
    "DenseFinetuneMethodTrainer",
    "RgcnGraphRetrieverTrainer",
]

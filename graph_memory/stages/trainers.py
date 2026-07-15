from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from graph_memory.models.dense_finetune.training import (
    DenseFinetuneRunConfig,
    DenseFinetuneSelectionSettings,
    DenseFinetuneTrainerSettings,
    DenseFinetuneTrainingResult,
    train_dense_finetune,
)
from graph_memory.models.dense_finetune.contracts import DenseFinetuneDataSettings
from graph_memory.registry.conversions import rgcn_training_config_from_trainer_settings
from graph_memory.experiment.config import DenseEncoderConfig
from graph_memory.experiment.stage_models import (
    DenseFinetuneTrainStageConfig,
    RgcnTrainStageConfig,
    ProvenanceRgcnTrainStageConfig,
)
from graph_memory.registry.retrieval import DenseEncoderSettings
from graph_memory.stages.train_payloads import (
    DenseFinetuneTrainPayload,
    RgcnTrainPayload,
    ProvenanceRgcnTrainPayload,
    TrainDependencies,
    TrainPayload,
)

if TYPE_CHECKING:
    from graph_memory.models.graph_retriever.training import RgcnTrainingResult


@dataclass(frozen=True)
class RgcnGraphRetrieverTrainer:
    config: RgcnTrainStageConfig

    def train(self, payload: TrainPayload) -> "RgcnTrainingResult":
        from graph_memory.models.graph_retriever.config.defaults import (
            default_model_config,
        )
        from graph_memory.models.graph_retriever.training import train_graph_retriever
        from graph_memory.models.graph_retriever.selection import (
            RgcnSelectionSettings,
        )

        if not isinstance(payload, RgcnTrainPayload):
            raise TypeError(
                f"R-GCN trainer expected RgcnTrainPayload, got {type(payload).__name__}."
            )
        settings = self.config.train
        encoder_settings = _effective_rgcn_encoder_settings(
            self.config.encoder, payload.seed_checkpoint
        )
        deps = payload.dependencies or _build_rgcn_dependencies(
            encoder_settings, device=settings.trainer.device
        )
        model_config = default_model_config(
            method_name=self.config.method,
            encoder_model=encoder_settings.model_name,
            encoder_dim=deps.text_embedding_provider.embedding_dim,
            query_prefix=encoder_settings.query_prefix,
            passage_prefix=encoder_settings.passage_prefix,
            encoder_batch_size=encoder_settings.batch_size,
            hidden_dim=settings.model.hidden_dim,
            num_layers=settings.model.num_layers,
            dropout=settings.model.dropout,
            ablation_name=settings.model.ablation,
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
            training_config=rgcn_training_config_from_trainer_settings(
                settings.trainer
            ),
            text_embedding_provider=deps.text_embedding_provider,
            seed_signal_provider=deps.seed_signal_provider,
            selection_settings=RgcnSelectionSettings(**settings.selection.model_dump()),
            device=settings.trainer.device,
        )


@dataclass(frozen=True)
class DenseFinetuneMethodTrainer:
    config: DenseFinetuneTrainStageConfig

    def train(self, payload: TrainPayload) -> DenseFinetuneTrainingResult:
        if not isinstance(payload, DenseFinetuneTrainPayload):
            raise TypeError(
                f"Dense-ft trainer expected DenseFinetuneTrainPayload, got {type(payload).__name__}."
            )
        settings = self.config.train
        encoder = self.config.encoder
        return train_dense_finetune(
            config=DenseFinetuneRunConfig(
                base_model=encoder.model_name,
                query_prefix=encoder.query_prefix,
                passage_prefix=encoder.passage_prefix,
                batch_size=encoder.batch_size,
                data=DenseFinetuneDataSettings(**settings.data.model_dump()),
                trainer=DenseFinetuneTrainerSettings(**settings.trainer.model_dump()),
                selection=DenseFinetuneSelectionSettings(
                    **settings.selection.model_dump()
                ),
            ),
            train_requests=payload.train_requests,
            train_pairs=payload.train_pairs,
            dev_requests=payload.dev_requests,
            dev_labels=payload.dev_labels,
            output_dir=payload.output_dir,
            model_dir=payload.model_dir,
        )


@dataclass(frozen=True)
class ProvenanceRgcnMethodTrainer:
    config: ProvenanceRgcnTrainStageConfig

    def train(self, payload: TrainPayload):
        import numpy as np

        from graph_memory.embeddings import load_sentence_transformer
        from graph_memory.models.provenance_rgcn import (
            ProvenanceRgcnModelConfig,
            ProvenanceRgcnTrainingConfig,
            train_provenance_rgcn,
        )

        if not isinstance(payload, ProvenanceRgcnTrainPayload):
            raise TypeError(
                "Provenance R-GCN trainer expected ProvenanceRgcnTrainPayload, "
                f"got {type(payload).__name__}."
            )
        encoder = payload.encoder or load_sentence_transformer(
            self.config.encoder.model_name,
            device=self.config.train.trainer.device,
        )
        probe = np.asarray(
            encoder.encode(
                [self.config.encoder.query_prefix + "dimension probe"],
                batch_size=1,
                normalize_embeddings=True,
            )
        )
        if probe.ndim != 2 or probe.shape[0] != 1:
            raise ValueError("Unable to infer provenance encoder dimension.")
        model = self.config.train.model
        trainer = self.config.train.trainer
        return train_provenance_rgcn(
            train_requests=payload.train_requests,
            train_labels=payload.train_labels,
            dev_requests=payload.dev_requests,
            dev_labels=payload.dev_labels,
            model_config=ProvenanceRgcnModelConfig(
                encoder_model=self.config.encoder.model_name,
                encoder_dim=int(probe.shape[1]),
                query_prefix=self.config.encoder.query_prefix,
                passage_prefix=self.config.encoder.passage_prefix,
                encoder_batch_size=self.config.encoder.batch_size,
                hidden_dim=model.hidden_dim,
                node_type_dim=model.node_type_dim,
                num_layers=model.num_layers,
                dropout=model.dropout,
            ),
            training_config=ProvenanceRgcnTrainingConfig(
                learning_rate=trainer.learning_rate,
                batch_size=trainer.batch_size,
                epochs=trainer.epochs,
                max_grad_norm=trainer.max_grad_norm,
                random_seed=trainer.random_seed,
                candidate_loss_weight=trainer.candidate_loss_weight,
                edge_loss_weight=trainer.edge_loss_weight,
            ),
            train_pairs=payload.train_pairs,
            encoder=encoder,
            device=trainer.device,
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

    text_embedding_provider = DenseGraphFeatureProvider(
        model_name=encoder_settings.model_name,
        query_prefix=encoder_settings.query_prefix,
        passage_prefix=encoder_settings.passage_prefix,
        batch_size=encoder_settings.batch_size,
        device=device,
    )
    return TrainDependencies(
        text_embedding_provider=text_embedding_provider,
        seed_signal_provider=text_embedding_provider,
    )


__all__ = [
    "DenseFinetuneMethodTrainer",
    "RgcnGraphRetrieverTrainer",
    "ProvenanceRgcnMethodTrainer",
]

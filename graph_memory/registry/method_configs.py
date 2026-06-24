from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Literal, TypeAlias, get_type_hints

from graph_memory.config.converter import ConfigConverter
from graph_memory.contracts.common import JsonValue
from graph_memory.models.dense_finetune.contracts import DenseFinetuneDataSettings
from graph_memory.models.dense_finetune.training import (
    DenseFinetuneSelectionSettings,
    DenseFinetuneTrainerSettings,
)
from graph_memory.registry.retrieval import DenseEncoderSettings, RetrievalMethodId


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
    random_seed: int = 13
    easy_random_per_positive: int = 2
    hard_bm25_per_positive: int = 2
    hard_dense_per_positive: int = 0
    hard_graph_neighbor_per_positive: int = 1
    hard_pool_size: int = 30


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
    method: Literal[
        RetrievalMethodId.DENSE_RGCN_GRAPH_RETRIEVER,
        RetrievalMethodId.DENSE_FT_RGCN_GRAPH_RETRIEVER,
    ] = RetrievalMethodId.DENSE_RGCN_GRAPH_RETRIEVER


@dataclass(frozen=True)
class DenseFinetuneMethodSettings:
    encoder: DenseEncoderSettings
    data: DenseFinetuneDataSettings = field(default_factory=DenseFinetuneDataSettings)
    trainer: DenseFinetuneTrainerSettings = field(default_factory=DenseFinetuneTrainerSettings)
    selection: DenseFinetuneSelectionSettings = field(default_factory=DenseFinetuneSelectionSettings)
    method: Literal[RetrievalMethodId.DENSE_FT] = RetrievalMethodId.DENSE_FT


@dataclass(frozen=True)
class RgcnTrainSettings:
    model: RgcnModelSettings
    trainer: RgcnTrainerSettings
    reporting: TrainingReportingSettings
    selection: ModelSelectionSettings


@dataclass(frozen=True)
class DenseFinetuneTrainSettings:
    data: DenseFinetuneDataSettings
    trainer: DenseFinetuneTrainerSettings
    selection: DenseFinetuneSelectionSettings


@dataclass(frozen=True)
class RgcnMethodConfig:
    method: Literal[
        RetrievalMethodId.DENSE_RGCN_GRAPH_RETRIEVER,
        RetrievalMethodId.DENSE_FT_RGCN_GRAPH_RETRIEVER,
    ]
    encoder: DenseEncoderSettings
    pairs: RgcnPairSamplingSettings
    train: RgcnTrainSettings


@dataclass(frozen=True)
class DenseFinetuneMethodConfig:
    method: Literal[RetrievalMethodId.DENSE_FT]
    encoder: DenseEncoderSettings
    pairs: RgcnPairSamplingSettings
    train: DenseFinetuneTrainSettings


TrainableMethodConfig: TypeAlias = RgcnMethodConfig | DenseFinetuneMethodConfig
TrainJobSettings: TypeAlias = RgcnMethodSettings | DenseFinetuneMethodSettings


def validate_complete_method_config_record(value: Mapping[str, JsonValue]) -> None:
    config = ConfigConverter().structure(value, TrainableMethodConfig)
    _require_all_dataclass_fields(value, type(config), path=type(config).__name__)


def _require_all_dataclass_fields(
    value: object,
    target_type: type[Any] | object,
    *,
    path: str,
) -> None:
    if not isinstance(target_type, type) or not is_dataclass(target_type):
        return
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be an object.")
    target_fields = fields(target_type)
    missing = sorted(field.name for field in target_fields if field.name not in value)
    if missing:
        raise ValueError(f"{path} missing required fields: {missing}.")
    type_hints = get_type_hints(target_type)
    for target_field in target_fields:
        _require_all_dataclass_fields(
            value[target_field.name],
            type_hints[target_field.name],
            path=f"{path}.{target_field.name}",
        )


__all__ = [
    "DenseFinetuneMethodConfig",
    "DenseFinetuneMethodSettings",
    "DenseFinetuneTrainSettings",
    "ModelSelectionSettings",
    "RgcnMethodConfig",
    "RgcnMethodSettings",
    "RgcnModelSettings",
    "RgcnPairSamplingSettings",
    "RgcnTrainerSettings",
    "RgcnTrainSettings",
    "TrainJobSettings",
    "TrainableMethodConfig",
    "TrainingReportingSettings",
    "validate_complete_method_config_record",
]

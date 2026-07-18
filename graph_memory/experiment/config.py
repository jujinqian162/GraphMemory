from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Annotated, Literal, TypeAlias, Union, cast

from omegaconf import DictConfig, OmegaConf
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    JsonValue,
    StrictBool,
    model_validator,
)

from graph_memory.models.graph_retriever.selection import RgcnSelectionMetric
from graph_memory.registry.retrieval import RetrievalMethodId


def _scientific_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("value must be an integer without string or boolean coercion")
    return value


def _scientific_float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("value must be numeric without string or boolean coercion")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("value must be finite")
    return number


def _device(value: object) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"(?:cpu|cuda(?::\d+)?)", value):
        raise ValueError("device must be cpu, cuda, or cuda:N")
    return value


ScientificInt = Annotated[int, BeforeValidator(_scientific_int)]
ScientificFloat = Annotated[float, BeforeValidator(_scientific_float)]
PositiveInt = Annotated[ScientificInt, Field(gt=0)]
NonNegativeInt = Annotated[ScientificInt, Field(ge=0)]
PositiveFloat = Annotated[ScientificFloat, Field(gt=0.0)]
NonNegativeFloat = Annotated[ScientificFloat, Field(ge=0.0)]
Device = Annotated[str, BeforeValidator(_device)]

DatasetName: TypeAlias = Literal[
    "hotpotqa",
    "twowiki",
    "twowiki_provenance",
    "musique",
]
SplitName: TypeAlias = Literal["train", "dev", "test"]
EvidenceRgcnVariant: TypeAlias = Literal[
    "full_rgcn",
    "wo_bridge",
    "wo_entity_overlap",
    "wo_sequential",
    "wo_query_overlap",
    "wo_graph",
    "wo_edge_type",
    "wo_edge_weight",
    "wo_seed_score",
    "wo_hard_negatives",
]
ProvenanceRgcnVariant: TypeAlias = Literal[
    "full_rgcn",
    "wo_graph",
    "wo_edge_type",
    "wo_edge_weight",
    "wo_hard_negatives",
]


class ClosedModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
        validate_default=True,
    )


class DatasetSplitBase(ClosedModel):
    source: Path
    offset: NonNegativeInt
    capacity: PositiveInt

    @model_validator(mode="after")
    def validate_offset(self) -> DatasetSplitBase:
        if self.offset >= self.capacity:
            raise ValueError(
                f"split offset={self.offset} must be smaller than capacity={self.capacity}"
            )
        return self


class RawDatasetSplitConfig(DatasetSplitBase):
    kind: Literal["raw"]


DatasetSplitConfig: TypeAlias = RawDatasetSplitConfig


class DatasetSplitsConfig(ClosedModel):
    train: DatasetSplitConfig
    dev: DatasetSplitConfig
    test: DatasetSplitConfig


class DatasetConfig(ClosedModel):
    name: DatasetName
    strict_invalid_examples: StrictBool = False
    splits: DatasetSplitsConfig


class FixedCountPolicy(ClosedModel):
    kind: Literal["fixed"]
    count: PositiveInt


class AllAvailableCountPolicy(ClosedModel):
    kind: Literal["all_available"]


CountPolicy: TypeAlias = Annotated[
    Union[FixedCountPolicy, AllAvailableCountPolicy],
    Field(discriminator="kind"),
]


class ProfileSplitPolicies(ClosedModel):
    train: CountPolicy
    dev: CountPolicy
    test: CountPolicy


class RgcnProfileSettings(ClosedModel):
    hidden_dim: PositiveInt
    num_layers: NonNegativeInt
    batch_size: PositiveInt
    epochs: PositiveInt
    easy_random_per_positive: NonNegativeInt
    hard_bm25_per_positive: NonNegativeInt
    hard_dense_per_positive: NonNegativeInt
    hard_graph_neighbor_per_positive: NonNegativeInt


class DenseFinetuneProfileSettings(ClosedModel):
    train_batch_size: PositiveInt
    eval_batch_size: PositiveInt
    epochs: PositiveInt
    easy_random_per_positive: NonNegativeInt
    hard_bm25_per_positive: NonNegativeInt
    hard_dense_per_positive: NonNegativeInt
    hard_graph_neighbor_per_positive: NonNegativeInt


class TrainableProfileSettings(ClosedModel):
    rgcn: RgcnProfileSettings
    dense_ft: DenseFinetuneProfileSettings


class ProfileConfig(ClosedModel):
    name: str = Field(min_length=1)
    splits: ProfileSplitPolicies
    trainable: TrainableProfileSettings


class DenseEncoderConfig(ClosedModel):
    model_name: str = Field(min_length=1)
    query_prefix: str
    passage_prefix: str
    batch_size: PositiveInt


class Bm25MethodConfig(ClosedModel):
    method: Literal["bm25"]


class DenseMethodConfig(ClosedModel):
    method: Literal["dense"]
    encoder: DenseEncoderConfig


class GraphRAGMethodConfig(ClosedModel):
    method: Literal["graphrag"]
    encoder: DenseEncoderConfig
    seed_top_s: PositiveInt
    max_entity_document_frequency_ratio: Annotated[
        ScientificFloat, Field(gt=0.0, le=1.0)
    ]
    sentence_resolver: Literal["frozen_dense"]
    min_sentence_score_margin: NonNegativeFloat
    min_bridge_confidence: NonNegativeFloat
    max_partners_per_anchor: Literal[1]
    preserve_dense_top_n: NonNegativeInt


class ExecutionProvenanceMethodConfig(ClosedModel):
    method: Literal["execution_provenance_retriever"]
    encoder: DenseEncoderConfig
    seed_top_s: PositiveInt
    beam_width: PositiveInt
    max_hops: PositiveInt
    max_paths_per_seed: Literal[1]
    max_path_expansions: PositiveInt
    min_path_confidence: NonNegativeFloat
    preserve_dense_top_n: NonNegativeInt
    hop_penalty: NonNegativeFloat


class PairSamplingConfig(ClosedModel):
    random_seed: ScientificInt
    easy_random_per_positive: NonNegativeInt
    hard_bm25_per_positive: NonNegativeInt
    hard_dense_per_positive: NonNegativeInt
    hard_graph_neighbor_per_positive: NonNegativeInt
    hard_pool_size: PositiveInt


class RgcnModelConfig(ClosedModel):
    hidden_dim: PositiveInt
    num_layers: NonNegativeInt
    dropout: Annotated[ScientificFloat, Field(ge=0.0, lt=1.0)]
    ablation: str = Field(min_length=1)


class RgcnTrainerConfig(ClosedModel):
    optimizer_name: str = Field(min_length=1)
    learning_rate: PositiveFloat
    batch_size: PositiveInt
    max_grad_norm: PositiveFloat
    random_seed: ScientificInt
    pos_weight_enabled: StrictBool
    epochs: PositiveInt
    device: Device


class ModelSelectionConfig(ClosedModel):
    best_metric: RgcnSelectionMetric
    higher_is_better: StrictBool


class RgcnTrainConfig(ClosedModel):
    model: RgcnModelConfig
    trainer: RgcnTrainerConfig
    selection: ModelSelectionConfig


def _effective_rgcn_parts(
    pairs: PairSamplingConfig,
    train: RgcnTrainConfig,
    variant: EvidenceRgcnVariant,
) -> tuple[PairSamplingConfig, RgcnTrainConfig]:
    if variant == "full_rgcn":
        return pairs, train
    if variant == "wo_hard_negatives":
        return (
            pairs.model_copy(
                update={
                    "hard_bm25_per_positive": 0,
                    "hard_dense_per_positive": 0,
                    "hard_graph_neighbor_per_positive": 0,
                }
            ),
            train,
        )
    model_updates: dict[str, object] = {"ablation": variant}
    if variant == "wo_graph":
        model_updates["num_layers"] = 0
    return pairs, train.model_copy(
        update={"model": train.model.model_copy(update=model_updates)}
    )


class RgcnStageConfig(ClosedModel):
    encoder: DenseEncoderConfig
    pairs: PairSamplingConfig
    train: RgcnTrainConfig

    def for_variant(self, variant: EvidenceRgcnVariant) -> RgcnStageConfig:
        pairs, train = _effective_rgcn_parts(self.pairs, self.train, variant)
        return self.model_copy(update={"pairs": pairs, "train": train})


class RgcnTrainStageConfig(ClosedModel):
    method: Literal[
        "dense_rgcn_graph_retriever",
        "dense_ft_rgcn_graph_retriever",
    ]
    variant: EvidenceRgcnVariant
    encoder: DenseEncoderConfig
    train: RgcnTrainConfig


class RgcnMethodConfig(RgcnStageConfig):
    method: Literal["dense_rgcn_graph_retriever"]
    variant: EvidenceRgcnVariant = "full_rgcn"

    def effective(self) -> RgcnMethodConfig:
        stage = super().for_variant(self.variant)
        return self.model_copy(update={"pairs": stage.pairs, "train": stage.train})

    def train_stage(self) -> RgcnTrainStageConfig:
        effective = self.effective()
        return RgcnTrainStageConfig(
            method=self.method,
            variant=self.variant,
            encoder=effective.encoder,
            train=effective.train,
        )


DenseRgcnMethodConfig = RgcnMethodConfig


class ProvenanceRgcnModelSettings(ClosedModel):
    hidden_dim: PositiveInt
    node_type_dim: PositiveInt
    num_layers: NonNegativeInt
    dropout: Annotated[ScientificFloat, Field(ge=0.0, lt=1.0)]
    ablation: str = Field(min_length=1)


class ProvenanceRgcnTrainerSettings(ClosedModel):
    learning_rate: PositiveFloat
    batch_size: PositiveInt
    epochs: PositiveInt
    max_grad_norm: PositiveFloat
    random_seed: ScientificInt
    device: Device
    candidate_loss_weight: NonNegativeFloat
    edge_loss_weight: NonNegativeFloat


class ProvenanceRgcnTrainSettings(ClosedModel):
    model: ProvenanceRgcnModelSettings
    trainer: ProvenanceRgcnTrainerSettings


class ProvenanceRgcnStageConfig(ClosedModel):
    method: Literal["execution_provenance_rgcn_retriever"]
    variant: ProvenanceRgcnVariant = "full_rgcn"
    encoder: DenseEncoderConfig
    train: ProvenanceRgcnTrainSettings


class ExecutionProvenanceRgcnMethodConfig(ProvenanceRgcnStageConfig):
    pairs: PairSamplingConfig

    def effective(self) -> ExecutionProvenanceRgcnMethodConfig:
        if self.variant == "full_rgcn":
            return self
        if self.variant == "wo_hard_negatives":
            return self.model_copy(
                update={
                    "pairs": self.pairs.model_copy(
                        update={
                            "hard_bm25_per_positive": 0,
                            "hard_dense_per_positive": 0,
                            "hard_graph_neighbor_per_positive": 0,
                        }
                    )
                }
            )
        model_updates: dict[str, object] = {"ablation": self.variant}
        if self.variant == "wo_graph":
            model_updates["num_layers"] = 0
        return self.model_copy(
            update={
                "train": self.train.model_copy(
                    update={"model": self.train.model.model_copy(update=model_updates)}
                )
            }
        )

    def train_stage(self) -> ProvenanceRgcnStageConfig:
        effective = self.effective()
        return ProvenanceRgcnStageConfig(
            method=effective.method,
            variant=effective.variant,
            encoder=effective.encoder,
            train=effective.train,
        )


class DenseFinetuneDataConfig(ClosedModel):
    hard_negatives_per_positive: NonNegativeInt


class DenseFinetuneTrainerConfig(ClosedModel):
    learning_rate: PositiveFloat
    train_batch_size: PositiveInt
    eval_batch_size: PositiveInt
    epochs: PositiveInt
    warmup_steps: NonNegativeInt
    max_grad_norm: PositiveFloat
    random_seed: ScientificInt
    device: Device
    use_amp: StrictBool


class DenseFinetuneSelectionConfig(ClosedModel):
    best_metric: str = Field(min_length=1)
    higher_is_better: StrictBool


class DenseFinetuneTrainConfig(ClosedModel):
    data: DenseFinetuneDataConfig
    trainer: DenseFinetuneTrainerConfig
    selection: DenseFinetuneSelectionConfig


class DenseFinetuneStageConfig(ClosedModel):
    method: Literal["dense_ft"]
    encoder: DenseEncoderConfig
    train: DenseFinetuneTrainConfig


class DenseFinetuneMethodConfig(DenseFinetuneStageConfig):
    pairs: PairSamplingConfig

    def train_stage(self) -> DenseFinetuneStageConfig:
        return DenseFinetuneStageConfig(
            method=self.method,
            encoder=self.encoder,
            train=self.train,
        )


class DenseFtRgcnMethodConfig(ClosedModel):
    method: Literal["dense_ft_rgcn_graph_retriever"]
    variant: EvidenceRgcnVariant = "full_rgcn"
    seed: DenseFinetuneMethodConfig
    rgcn: RgcnStageConfig

    def effective_rgcn(self) -> RgcnStageConfig:
        return self.rgcn.for_variant(self.variant)

    def rgcn_train_stage(self) -> RgcnTrainStageConfig:
        effective = self.effective_rgcn()
        return RgcnTrainStageConfig(
            method=self.method,
            variant=self.variant,
            encoder=effective.encoder,
            train=effective.train,
        )


MethodConfig: TypeAlias = Annotated[
    Union[
        Bm25MethodConfig,
        DenseMethodConfig,
        GraphRAGMethodConfig,
        ExecutionProvenanceMethodConfig,
        RgcnMethodConfig,
        DenseFinetuneMethodConfig,
        DenseFtRgcnMethodConfig,
        ExecutionProvenanceRgcnMethodConfig,
    ],
    Field(discriminator="method"),
]


class TrainableRankingConfig(ClosedModel):
    method: Literal[
        "dense_ft",
        "dense_rgcn_graph_retriever",
        "dense_ft_rgcn_graph_retriever",
        "execution_provenance_rgcn_retriever",
    ]
    variant: str | None = None


RankingMethodConfig: TypeAlias = Annotated[
    Union[
        Bm25MethodConfig,
        DenseMethodConfig,
        GraphRAGMethodConfig,
        ExecutionProvenanceMethodConfig,
        TrainableRankingConfig,
    ],
    Field(discriminator="method"),
]


def ranking_config(method: MethodConfig) -> RankingMethodConfig:
    if isinstance(
        method,
        (
            Bm25MethodConfig,
            DenseMethodConfig,
            GraphRAGMethodConfig,
            ExecutionProvenanceMethodConfig,
        ),
    ):
        return method
    return TrainableRankingConfig(
        method=method.method,
        variant=getattr(method, "variant", None),
    )


class PairBuildConfig(ClosedModel):
    sampling: PairSamplingConfig
    encoder: DenseEncoderConfig
    device: Device


class GraphBuildConfig(ClosedModel):
    max_query_overlap: NonNegativeInt
    max_entity_neighbors: NonNegativeInt
    max_bridge_edges: NonNegativeInt
    use_spacy: StrictBool


class PrepareSplitConfig(ClosedModel):
    dataset: DatasetName
    split: SplitName
    count: PositiveInt
    offset: NonNegativeInt
    seed: ScientificInt
    strict_invalid_examples: StrictBool


class CacheConfig(ClosedModel):
    refresh: StrictBool = False


class BenchmarkConfig(ClosedModel):
    enabled: StrictBool = False
    warmup: NonNegativeInt = 1
    repetitions: PositiveInt = 5


class EvaluationConfig(ClosedModel):
    failure_case_limit: NonNegativeInt = 50


class TrackingConfig(ClosedModel):
    database: Path
    artifact_root: Path
    experiment_name: str = Field(min_length=1)


class ExperimentConfig(ClosedModel):
    name: str = Field(min_length=1)
    dataset: DatasetConfig
    profile: ProfileConfig
    method: MethodConfig
    seed: ScientificInt
    device: Device
    top_k: PositiveInt
    cache: CacheConfig
    benchmark: BenchmarkConfig
    graph: GraphBuildConfig
    evaluation: EvaluationConfig
    tracking: TrackingConfig


class ResolvedRawSplitConfig(ClosedModel):
    kind: Literal["raw"]
    source: Path
    offset: NonNegativeInt
    capacity: PositiveInt
    count: PositiveInt


ResolvedSplitConfig: TypeAlias = ResolvedRawSplitConfig


class ResolvedDatasetConfig(ClosedModel):
    name: DatasetName
    strict_invalid_examples: StrictBool
    splits: dict[SplitName, ResolvedSplitConfig]


class ResolvedTrackingConfig(ClosedModel):
    database: Path
    artifact_root: Path
    experiment_name: str

    @property
    def tracking_uri(self) -> str:
        return f"sqlite:///{self.database.as_posix()}"


class ResolvedExperimentConfig(ClosedModel):
    name: str
    dataset: ResolvedDatasetConfig
    profile: str
    method: MethodConfig
    seed: ScientificInt
    device: Device
    top_k: PositiveInt
    cache: CacheConfig
    benchmark: BenchmarkConfig
    graph: GraphBuildConfig
    evaluation: EvaluationConfig
    tracking: ResolvedTrackingConfig

    @property
    def method_id(self) -> RetrievalMethodId:
        return RetrievalMethodId(self.method.method)

    @property
    def variant(self) -> str | None:
        return getattr(self.method, "variant", None)

    def normalized(self) -> dict[str, JsonValue]:
        return cast(
            dict[str, JsonValue],
            self.model_dump(mode="json", by_alias=True),
        )


def parse_composed_config(config: DictConfig) -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        OmegaConf.to_container(
            config,
            resolve=True,
            throw_on_missing=True,
            enum_to_str=True,
        )
    )


def resolve_experiment_config(
    config: ExperimentConfig,
    *,
    repository_root: Path,
) -> ResolvedExperimentConfig:
    root = repository_root.resolve()
    split_names: tuple[SplitName, ...] = ("train", "dev", "test")
    resolved_splits: dict[SplitName, ResolvedSplitConfig] = {}
    for split_name in split_names:
        dataset_split = getattr(config.dataset.splits, split_name)
        policy = getattr(config.profile.splits, split_name)
        available = dataset_split.capacity - dataset_split.offset
        count = policy.count if isinstance(policy, FixedCountPolicy) else available
        if count > available:
            raise ValueError(
                f"profile={config.profile.name} split={split_name} requests "
                f"offset+count={dataset_split.offset + count} beyond "
                f"capacity={dataset_split.capacity}"
            )
        resolved_splits[split_name] = ResolvedRawSplitConfig(
            kind="raw",
            source=_absolute_path(root, dataset_split.source),
            offset=dataset_split.offset,
            capacity=dataset_split.capacity,
            count=count,
        )

    _check_dataset_method_compatibility(config.dataset.name, config.method)
    return ResolvedExperimentConfig(
        name=config.name,
        dataset=ResolvedDatasetConfig(
            name=config.dataset.name,
            strict_invalid_examples=config.dataset.strict_invalid_examples,
            splits=resolved_splits,
        ),
        profile=config.profile.name,
        method=config.method,
        seed=config.seed,
        device=config.device,
        top_k=config.top_k,
        cache=config.cache,
        benchmark=config.benchmark,
        graph=config.graph,
        evaluation=config.evaluation,
        tracking=ResolvedTrackingConfig(
            database=_absolute_path(root, config.tracking.database),
            artifact_root=_absolute_path(root, config.tracking.artifact_root),
            experiment_name=config.tracking.experiment_name,
        ),
    )


def _check_dataset_method_compatibility(
    dataset: DatasetName,
    method: MethodConfig,
) -> None:
    from graph_memory.registry import Registry
    from graph_memory.registry.semantics import RetrievalTaskFamily

    family = (
        RetrievalTaskFamily.EXECUTION_PROVENANCE
        if dataset == "twowiki_provenance"
        else RetrievalTaskFamily.EVIDENCE_RETRIEVAL
    )
    method_id = RetrievalMethodId(method.method)
    supported = Registry.methods.get(method_id).input_spec.supported_families
    if family not in supported:
        raise ValueError(
            f"dataset={dataset!r} uses family={family.value!r}, but "
            f"method={method_id.value!r} does not support that family."
        )


def _absolute_path(root: Path, value: Path) -> Path:
    return value.resolve() if value.is_absolute() else (root / value).resolve()


__all__ = [
    "AllAvailableCountPolicy",
    "BenchmarkConfig",
    "Bm25MethodConfig",
    "CacheConfig",
    "ClosedModel",
    "CountPolicy",
    "DatasetConfig",
    "DatasetName",
    "DenseEncoderConfig",
    "DenseFinetuneMethodConfig",
    "DenseFinetuneStageConfig",
    "DenseFinetuneTrainConfig",
    "DenseFinetuneTrainerConfig",
    "DenseFtRgcnMethodConfig",
    "DenseMethodConfig",
    "DenseRgcnMethodConfig",
    "Device",
    "EvaluationConfig",
    "ExecutionProvenanceMethodConfig",
    "ExecutionProvenanceRgcnMethodConfig",
    "ExperimentConfig",
    "FixedCountPolicy",
    "GraphBuildConfig",
    "GraphRAGMethodConfig",
    "MethodConfig",
    "ModelSelectionConfig",
    "NonNegativeFloat",
    "NonNegativeInt",
    "PairSamplingConfig",
    "PairBuildConfig",
    "PositiveFloat",
    "PositiveInt",
    "ProfileConfig",
    "PrepareSplitConfig",
    "ProvenanceRgcnModelSettings",
    "ProvenanceRgcnTrainerSettings",
    "ProvenanceRgcnTrainSettings",
    "ProvenanceRgcnStageConfig",
    "RankingMethodConfig",
    "ResolvedExperimentConfig",
    "ResolvedSplitConfig",
    "RgcnMethodConfig",
    "RgcnModelConfig",
    "RgcnStageConfig",
    "RgcnTrainStageConfig",
    "RgcnTrainConfig",
    "RgcnTrainerConfig",
    "ScientificFloat",
    "ScientificInt",
    "SplitName",
    "TrackingConfig",
    "TrainableRankingConfig",
    "ranking_config",
    "resolve_experiment_config",
    "parse_composed_config",
]

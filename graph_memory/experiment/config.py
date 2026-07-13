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
    field_validator,
    model_validator,
)

from graph_memory.registry.retrieval import RetrievalMethodId
from graph_memory.models.graph_retriever.selection import RgcnSelectionMetric


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

DatasetName: TypeAlias = Literal["hotpotqa", "twowiki", "musique"]
SplitName: TypeAlias = Literal["train", "dev", "test"]
PublicStageName: TypeAlias = Literal[
    "prepare",
    "graphs",
    "pairs",
    "tune",
    "train",
    "retrieve",
    "evaluate",
    "aggregate",
]
ArtifactKind: TypeAlias = Literal["file", "directory"]


class ClosedModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
        validate_default=True,
    )


class ArtifactRef(ClosedModel):
    role: str = Field(min_length=1)
    path: Path
    kind: ArtifactKind


class AliasArtifactRef(ArtifactRef):
    alias_of: Path


ArtifactBinding: TypeAlias = ArtifactRef | AliasArtifactRef


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


class ImportanceDatasetSplitConfig(DatasetSplitBase):
    kind: Literal["importance"]
    labels_source: Path
    importance_path: Path


DatasetSplitConfig: TypeAlias = Annotated[
    Union[RawDatasetSplitConfig, ImportanceDatasetSplitConfig],
    Field(discriminator="kind"),
]


class DatasetSplitsConfig(ClosedModel):
    train: DatasetSplitConfig
    dev: DatasetSplitConfig
    test: DatasetSplitConfig


class DatasetConfig(ClosedModel):
    name: DatasetName
    prepare_script: Path
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


class MemoryStreamScoringConfig(ClosedModel):
    relevance_weight: NonNegativeFloat
    recency_weight: NonNegativeFloat
    importance_weight: NonNegativeFloat
    recency_decay: Annotated[ScientificFloat, Field(gt=0.0, le=1.0)]

    @model_validator(mode="after")
    def validate_positive_weight(self) -> MemoryStreamScoringConfig:
        if self.relevance_weight + self.recency_weight + self.importance_weight <= 0:
            raise ValueError(
                "Memory Stream requires at least one positive scoring weight"
            )
        return self


class MemoryStreamMethodConfig(ClosedModel):
    method: Literal["memory_stream"]
    encoder: DenseEncoderConfig
    scoring: MemoryStreamScoringConfig


class Bm25GraphRerankMethodConfig(ClosedModel):
    method: Literal["bm25_graph_rerank"]


class DenseGraphRerankMethodConfig(ClosedModel):
    method: Literal["dense_graph_rerank"]
    encoder: DenseEncoderConfig


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


class RgcnDecoderConfig(ClosedModel):
    hidden_dim: PositiveInt
    step_embedding_dim: PositiveInt
    frontier_relation_dim: PositiveInt


class RgcnBeamSearchConfig(ClosedModel):
    training_beam_size: Annotated[ScientificInt, Field(gt=0, le=4)]
    inference_beam_size: Annotated[ScientificInt, Field(gt=0, le=4)]
    max_steps: Annotated[ScientificInt, Field(gt=0, le=5)]
    length_penalty_alpha: NonNegativeFloat
    deduplicate_selected_sets: StrictBool

    @model_validator(mode="after")
    def validate_matching_beam_sizes(self) -> RgcnBeamSearchConfig:
        if self.training_beam_size != self.inference_beam_size:
            raise ValueError("training and inference beam sizes must match")
        return self


class RgcnBeamLossConfig(ClosedModel):
    next_action_loss_weight: NonNegativeFloat
    stop_loss_weight: NonNegativeFloat
    aux_node_loss_weight: NonNegativeFloat


class RgcnOptimizerPhaseConfig(ClosedModel):
    decoder_warmup_epochs: NonNegativeInt
    decoder_learning_rate: PositiveFloat
    rgcn_learning_rate: PositiveFloat


class ModelSelectionConfig(ClosedModel):
    best_metric: RgcnSelectionMetric
    higher_is_better: StrictBool


class RgcnTrainConfig(ClosedModel):
    model: RgcnModelConfig
    trainer: RgcnTrainerConfig
    decoder: RgcnDecoderConfig
    beam: RgcnBeamSearchConfig
    loss: RgcnBeamLossConfig
    optimizer_phases: RgcnOptimizerPhaseConfig
    selection: ModelSelectionConfig


class DenseRgcnMethodConfig(ClosedModel):
    method: Literal["dense_rgcn_graph_retriever"]
    encoder: DenseEncoderConfig
    pairs: PairSamplingConfig
    train: RgcnTrainConfig


class DenseFtRgcnMethodConfig(ClosedModel):
    method: Literal["dense_ft_rgcn_graph_retriever"]
    encoder: DenseEncoderConfig
    pairs: PairSamplingConfig
    train: RgcnTrainConfig


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


class DenseFinetuneMethodConfig(ClosedModel):
    method: Literal["dense_ft"]
    encoder: DenseEncoderConfig
    pairs: PairSamplingConfig
    train: DenseFinetuneTrainConfig


MethodConfig: TypeAlias = Annotated[
    Union[
        Bm25MethodConfig,
        DenseMethodConfig,
        MemoryStreamMethodConfig,
        Bm25GraphRerankMethodConfig,
        DenseGraphRerankMethodConfig,
        DenseRgcnMethodConfig,
        DenseFinetuneMethodConfig,
        DenseFtRgcnMethodConfig,
    ],
    Field(discriminator="method"),
]


class MethodConfigs(ClosedModel):
    bm25: Bm25MethodConfig
    dense: DenseMethodConfig
    memory_stream: MemoryStreamMethodConfig
    bm25_graph_rerank: Bm25GraphRerankMethodConfig
    dense_graph_rerank: DenseGraphRerankMethodConfig
    dense_rgcn_graph_retriever: DenseRgcnMethodConfig
    dense_ft: DenseFinetuneMethodConfig
    dense_ft_rgcn_graph_retriever: DenseFtRgcnMethodConfig

    def get(self, method: RetrievalMethodId) -> MethodConfig:
        return getattr(self, method.value)


class GraphBuildConfig(ClosedModel):
    max_query_overlap: NonNegativeInt
    max_entity_neighbors: NonNegativeInt
    max_bridge_edges: NonNegativeInt
    use_spacy: StrictBool


class GraphRerankSearchSpace(ClosedModel):
    lambda_bridge: list[NonNegativeFloat]
    lambda_init: list[NonNegativeFloat]
    lambda_neighbor: list[NonNegativeFloat]
    lambda_path: list[NonNegativeFloat]
    lambda_query: list[NonNegativeFloat]
    max_hops: list[PositiveInt]
    seed_top_s: list[PositiveInt]
    neighbor_type_weights: dict[str, NonNegativeFloat]


class MemoryStreamSearchSpace(ClosedModel):
    relevance_weight: list[NonNegativeFloat]
    recency_weight: list[NonNegativeFloat]
    importance_weight: list[NonNegativeFloat]
    recency_decay: list[Annotated[ScientificFloat, Field(gt=0.0, le=1.0)]]


class SearchSpacesConfig(ClosedModel):
    graph_rerank: GraphRerankSearchSpace
    memory_stream: MemoryStreamSearchSpace


class StageBoundsConfig(ClosedModel):
    from_stage: PublicStageName | None = Field(alias="from")
    to_stage: PublicStageName | None = Field(alias="to")

    @model_validator(mode="after")
    def validate_order(self) -> StageBoundsConfig:
        if self.from_stage is None or self.to_stage is None:
            return self
        order = (
            "prepare",
            "graphs",
            "pairs",
            "tune",
            "train",
            "retrieve",
            "evaluate",
            "aggregate",
        )
        if order.index(self.from_stage) > order.index(self.to_stage):
            raise ValueError(
                f"stages.from={self.from_stage} must not follow stages.to={self.to_stage}"
            )
        return self


class CacheConfig(ClosedModel):
    enabled: StrictBool


class AblationConfig(ClosedModel):
    variants: Literal["all"] | list[str]
    only: StrictBool

    @field_validator("variants")
    @classmethod
    def validate_variants(
        cls, value: Literal["all"] | list[str]
    ) -> Literal["all"] | list[str]:
        if isinstance(value, list):
            if "full_rgcn" in value:
                raise ValueError("full_rgcn is a baseline alias and cannot be executed")
            if len(value) != len(set(value)):
                raise ValueError("ablation variants must be unique")
            if any(not item for item in value):
                raise ValueError("ablation variant names must be non-empty")
        return value


class TrackingConfig(ClosedModel):
    database: Path
    artifact_root: Path
    experiment_name: str = Field(min_length=1)


class ExperimentConfig(ClosedModel):
    name: str = Field(min_length=1)
    dataset: DatasetConfig
    profile: ProfileConfig
    methods: list[RetrievalMethodId] = Field(min_length=1)
    method_configs: MethodConfigs
    seed: ScientificInt
    device: Device
    top_k: PositiveInt
    stages: StageBoundsConfig
    cache: CacheConfig
    ablation: AblationConfig
    graph: GraphBuildConfig
    search_spaces: SearchSpacesConfig
    tracking: TrackingConfig

    @field_validator("methods")
    @classmethod
    def validate_unique_methods(
        cls,
        value: list[RetrievalMethodId],
    ) -> list[RetrievalMethodId]:
        if len(value) != len(set(value)):
            raise ValueError("methods must be unique")
        return value

    @model_validator(mode="after")
    def validate_root_propagation(self) -> ExperimentConfig:
        for method in (
            self.method_configs.dense_rgcn_graph_retriever,
            self.method_configs.dense_ft_rgcn_graph_retriever,
            self.method_configs.dense_ft,
        ):
            if method.pairs.random_seed != self.seed:
                raise ValueError(
                    f"method_configs.{method.method}.pairs.random_seed must interpolate root seed"
                )
            if method.train.trainer.random_seed != self.seed:
                raise ValueError(
                    f"method_configs.{method.method}.train.trainer.random_seed must interpolate root seed"
                )
            if method.train.trainer.device != self.device:
                raise ValueError(
                    f"method_configs.{method.method}.train.trainer.device must interpolate root device"
                )
        return self


class ResolvedRawSplitConfig(ClosedModel):
    kind: Literal["raw"]
    source: Path
    offset: NonNegativeInt
    capacity: PositiveInt
    count: PositiveInt


class ResolvedImportanceSplitConfig(ClosedModel):
    kind: Literal["importance"]
    source: Path
    labels_source: Path
    importance_path: Path
    offset: NonNegativeInt
    capacity: PositiveInt
    count: PositiveInt


ResolvedSplitConfig: TypeAlias = Annotated[
    Union[ResolvedRawSplitConfig, ResolvedImportanceSplitConfig],
    Field(discriminator="kind"),
]


class ResolvedDatasetConfig(ClosedModel):
    name: DatasetName
    prepare_script: Path
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
    methods: list[RetrievalMethodId]
    method_configs: MethodConfigs
    seed: ScientificInt
    device: str
    top_k: PositiveInt
    stages: StageBoundsConfig
    cache: CacheConfig
    ablation: AblationConfig
    graph: GraphBuildConfig
    search_spaces: SearchSpacesConfig
    tracking: ResolvedTrackingConfig

    def normalized(self) -> dict[str, JsonValue]:
        return cast(
            dict[str, JsonValue],
            self.model_dump(mode="json", by_alias=True),
        )


def validate_composed_config(config: DictConfig) -> ExperimentConfig:
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
                f"offset+count={dataset_split.offset + count} beyond capacity={dataset_split.capacity}"
            )
        resolved_splits[split_name] = (
            ResolvedImportanceSplitConfig(
                kind="importance",
                source=_absolute_path(root, dataset_split.source),
                offset=dataset_split.offset,
                capacity=dataset_split.capacity,
                count=count,
                labels_source=_absolute_path(root, dataset_split.labels_source),
                importance_path=_absolute_path(root, dataset_split.importance_path),
            )
            if isinstance(dataset_split, ImportanceDatasetSplitConfig)
            else ResolvedRawSplitConfig(
                kind="raw",
                source=_absolute_path(root, dataset_split.source),
                offset=dataset_split.offset,
                capacity=dataset_split.capacity,
                count=count,
            )
        )

    return ResolvedExperimentConfig(
        name=config.name,
        dataset=ResolvedDatasetConfig(
            name=config.dataset.name,
            prepare_script=_absolute_path(root, config.dataset.prepare_script),
            splits=resolved_splits,
        ),
        profile=config.profile.name,
        methods=list(config.methods),
        method_configs=config.method_configs,
        seed=config.seed,
        device=config.device,
        top_k=config.top_k,
        stages=config.stages,
        cache=config.cache,
        ablation=config.ablation,
        graph=config.graph,
        search_spaces=config.search_spaces,
        tracking=ResolvedTrackingConfig(
            database=_absolute_path(root, config.tracking.database),
            artifact_root=_absolute_path(root, config.tracking.artifact_root),
            experiment_name=config.tracking.experiment_name,
        ),
    )


def _absolute_path(root: Path, value: Path) -> Path:
    return value.resolve() if value.is_absolute() else (root / value).resolve()


__all__ = [
    "AblationConfig",
    "AllAvailableCountPolicy",
    "AliasArtifactRef",
    "ArtifactBinding",
    "ArtifactRef",
    "Bm25GraphRerankMethodConfig",
    "Bm25MethodConfig",
    "CacheConfig",
    "CountPolicy",
    "DatasetConfig",
    "DenseFinetuneMethodConfig",
    "DenseFtRgcnMethodConfig",
    "DenseGraphRerankMethodConfig",
    "DenseMethodConfig",
    "DenseRgcnMethodConfig",
    "ExperimentConfig",
    "FixedCountPolicy",
    "GraphBuildConfig",
    "MethodConfig",
    "MethodConfigs",
    "MemoryStreamMethodConfig",
    "NonNegativeFloat",
    "NonNegativeInt",
    "PositiveFloat",
    "PositiveInt",
    "ProfileConfig",
    "PublicStageName",
    "ResolvedExperimentConfig",
    "ResolvedSplitConfig",
    "RgcnBeamLossConfig",
    "RgcnBeamSearchConfig",
    "RgcnDecoderConfig",
    "RgcnOptimizerPhaseConfig",
    "ScientificFloat",
    "ScientificInt",
    "SearchSpacesConfig",
    "StageBoundsConfig",
    "TrackingConfig",
    "resolve_experiment_config",
    "validate_composed_config",
]

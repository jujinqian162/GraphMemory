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

from graph_memory.models.dense_finetune.contracts import DenseFinetuneDataSettings
from graph_memory.models.dense_finetune.training import (
    DenseFinetuneSelectionSettings,
    DenseFinetuneTrainerSettings,
)
from graph_memory.models.graph_retriever.config.records import RgcnTrainingConfig
from graph_memory.models.graph_retriever.selection import RgcnSelectionSettings
from graph_memory.training_pairs.config import NegativeSamplingConfig
from graph_memory.registry.retrieval import RetrievalMethodId
from graph_memory.retrieval.methods.graphrag import GraphRAGConfig
from graph_memory.retrieval.methods.provenance_path import ProvenancePathConfig


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
    "musique",
    "isetrace",
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
    capacity: PositiveInt | None = None

    @model_validator(mode="after")
    def validate_offset(self) -> DatasetSplitBase:
        if self.capacity is not None and self.offset >= self.capacity:
            raise ValueError(
                f"split offset={self.offset} must be smaller than capacity={self.capacity}"
            )
        return self


class RawDatasetSplitConfig(DatasetSplitBase):
    kind: Literal["raw"]


DatasetSplitConfig: TypeAlias = RawDatasetSplitConfig


class DatasetSplitsConfig(ClosedModel):
    train: DatasetSplitConfig | None = None
    dev: DatasetSplitConfig | None = None
    test: DatasetSplitConfig


class ISETraceChunkingConfig(ClosedModel):
    tokenizer_name: str = Field(min_length=1)
    max_tokens: PositiveInt = 512
    overlap_tokens: NonNegativeInt = 64
    reserved_tokens: NonNegativeInt = 0

    @model_validator(mode="after")
    def _validate_overlap(self) -> "ISETraceChunkingConfig":
        content_tokens = self.max_tokens - self.reserved_tokens
        if content_tokens <= 0:
            raise ValueError("chunk reserved_tokens must be smaller than max_tokens")
        if self.overlap_tokens >= content_tokens:
            raise ValueError(
                "chunk overlap_tokens must be smaller than the content token budget"
            )
        return self


class DatasetConfig(ClosedModel):
    name: DatasetName
    strict_invalid_examples: StrictBool = False
    trajectory_source: Path | None = None
    source_revision: str | None = Field(default=None, min_length=1)
    chunking: ISETraceChunkingConfig | None = None
    splits: DatasetSplitsConfig

    @model_validator(mode="after")
    def _validate_dataset_sources(self) -> "DatasetConfig":
        if self.name == "isetrace":
            if self.trajectory_source is None or self.source_revision is None:
                raise ValueError(
                    "isetrace requires trajectory_source and source_revision"
                )
            if self.chunking is None:
                raise ValueError("isetrace requires chunking settings")
        elif (
            self.trajectory_source is not None
            or self.source_revision is not None
            or self.chunking is not None
        ):
            raise ValueError(
                "trajectory_source/source_revision/chunking are reserved for isetrace"
            )
        return self


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
    per_device_graph_batch_size: PositiveInt
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
    evidence_rgcn: RgcnProfileSettings
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


class GraphRAGMethodConfig(GraphRAGConfig):
    method: Literal["graphrag"]
    encoder: DenseEncoderConfig


class ProvenancePathMethodConfig(ClosedModel):
    method: Literal["provenance_path"]
    encoder: DenseEncoderConfig
    seed_top_s: PositiveInt = 5
    max_path_hops: PositiveInt = 6
    max_partners_per_anchor: PositiveInt = 3
    max_expansions: PositiveInt = 256
    preserve_dense_top_n: NonNegativeInt = 2

    def retrieval_config(self) -> ProvenancePathConfig:
        return ProvenancePathConfig(
            seed_top_s=self.seed_top_s,
            max_path_hops=self.max_path_hops,
            max_partners_per_anchor=self.max_partners_per_anchor,
            max_expansions=self.max_expansions,
            preserve_dense_top_n=self.preserve_dense_top_n,
        )


class PairSamplingConfig(NegativeSamplingConfig):
    pass


class RgcnModelConfig(ClosedModel):
    hidden_dim: PositiveInt
    num_layers: NonNegativeInt
    dropout: Annotated[ScientificFloat, Field(ge=0.0, lt=1.0)]
    ablation: str = Field(min_length=1)


class RgcnTrainerConfig(RgcnTrainingConfig):
    device: Device


class ModelSelectionConfig(RgcnSelectionSettings):
    pass


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


class DenseFinetuneDataConfig(DenseFinetuneDataSettings):
    pass


class DenseFinetuneTrainerConfig(DenseFinetuneTrainerSettings):
    device: Device


class DenseFinetuneSelectionConfig(DenseFinetuneSelectionSettings):
    pass


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
        ProvenancePathMethodConfig,
        RgcnMethodConfig,
        DenseFinetuneMethodConfig,
        DenseFtRgcnMethodConfig,
    ],
    Field(discriminator="method"),
]


class TrainableRankingConfig(ClosedModel):
    method: Literal[
        "dense_ft",
        "dense_rgcn_graph_retriever",
        "dense_ft_rgcn_graph_retriever",
    ]
    variant: str | None = None


RankingMethodConfig: TypeAlias = Annotated[
    Union[
        Bm25MethodConfig,
        DenseMethodConfig,
        GraphRAGMethodConfig,
        ProvenancePathMethodConfig,
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
            ProvenancePathMethodConfig,
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
    count: PositiveInt | None = None
    offset: NonNegativeInt
    seed: ScientificInt
    strict_invalid_examples: StrictBool
    source_revision: str | None = Field(default=None, min_length=1)
    chunking: ISETraceChunkingConfig | None = None


class CacheConfig(ClosedModel):
    refresh: StrictBool = False


class EncodingConfig(ClosedModel):
    """Runtime controls for frozen R-GCN encoding; not scientific identity."""

    enable_gpupool: StrictBool
    chunk_size: PositiveInt


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
    split_seed: ScientificInt = 13
    device: Device
    top_k: PositiveInt
    cache: CacheConfig
    encoding: EncodingConfig
    benchmark: BenchmarkConfig
    graph: GraphBuildConfig
    evaluation: EvaluationConfig
    tracking: TrackingConfig


class ResolvedRawSplitConfig(ClosedModel):
    kind: Literal["raw"]
    source: Path
    offset: NonNegativeInt
    capacity: PositiveInt | None = None
    count: PositiveInt | None = None


ResolvedSplitConfig: TypeAlias = ResolvedRawSplitConfig


class ResolvedDatasetConfig(ClosedModel):
    name: DatasetName
    strict_invalid_examples: StrictBool
    trajectory_source: Path | None = None
    source_revision: str | None = None
    chunking: ISETraceChunkingConfig | None = None
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
    split_seed: ScientificInt = 13
    device: Device
    top_k: PositiveInt
    cache: CacheConfig
    encoding: EncodingConfig
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
        if dataset_split is None:
            continue
        policy = getattr(config.profile.splits, split_name)
        if isinstance(policy, FixedCountPolicy):
            count: int | None = policy.count
        elif dataset_split.capacity is not None:
            count = dataset_split.capacity - dataset_split.offset
        else:
            count = None
        if (
            dataset_split.capacity is not None
            and count is not None
            and count > dataset_split.capacity - dataset_split.offset
        ):
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
    _require_method_splits(config.method, resolved_splits)
    return ResolvedExperimentConfig(
        name=config.name,
        dataset=ResolvedDatasetConfig(
            name=config.dataset.name,
            strict_invalid_examples=config.dataset.strict_invalid_examples,
            trajectory_source=(
                None
                if config.dataset.trajectory_source is None
                else _absolute_path(root, config.dataset.trajectory_source)
            ),
            source_revision=config.dataset.source_revision,
            chunking=config.dataset.chunking,
            splits=resolved_splits,
        ),
        profile=config.profile.name,
        method=config.method,
        seed=config.seed,
        split_seed=config.split_seed,
        device=config.device,
        top_k=config.top_k,
        cache=config.cache,
        encoding=config.encoding,
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
    from graph_memory.registry.retrieval import RetrievalTaskFamily

    family = (
        RetrievalTaskFamily.EXECUTION_PROVENANCE
        if dataset == "isetrace"
        else RetrievalTaskFamily.EVIDENCE_RETRIEVAL
    )
    method_id = RetrievalMethodId(method.method)
    supported = Registry.methods.get(method_id).supported_families
    if family not in supported:
        raise ValueError(
            f"dataset={dataset!r} uses family={family.value!r}, but "
            f"method={method_id.value!r} does not support that family."
        )


def _require_method_splits(
    method: MethodConfig,
    splits: dict[SplitName, ResolvedSplitConfig],
) -> None:
    required: set[SplitName] = {"test"}
    if isinstance(
        method,
        (
            DenseFinetuneMethodConfig,
            RgcnMethodConfig,
            DenseFtRgcnMethodConfig,
        ),
    ):
        required.update({"train", "dev"})
    missing = sorted(required - set(splits))
    if missing:
        raise ValueError(f"method={method.method!r} requires dataset splits={missing}")


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
    "EncodingConfig",
    "EvaluationConfig",
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
    "ProvenancePathMethodConfig",
    "PrepareSplitConfig",
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

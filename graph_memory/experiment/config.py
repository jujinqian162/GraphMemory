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
from graph_memory.retrieval.methods.ids import RetrievalMethodId
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
ProvenanceRgcnVariant: TypeAlias = Literal["full_rgcn", "wo_graph"]
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


class DatasetSplitsConfig(ClosedModel):
    train: RawDatasetSplitConfig | None = None
    dev: RawDatasetSplitConfig | None = None
    test: RawDatasetSplitConfig


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


class ISETraceTrajectoryOriginCounts(ClosedModel):
    natural: NonNegativeInt
    template: NonNegativeInt

    @model_validator(mode="after")
    def _require_trajectories(self) -> "ISETraceTrajectoryOriginCounts":
        if self.natural + self.template <= 0:
            raise ValueError("ISETrace split must request at least one trajectory")
        return self


class ISETraceTrajectorySplitCounts(ClosedModel):
    train: ISETraceTrajectoryOriginCounts
    dev: ISETraceTrajectoryOriginCounts
    test: ISETraceTrajectoryOriginCounts

    @model_validator(mode="after")
    def _require_natural_only_test(self) -> "ISETraceTrajectorySplitCounts":
        if self.test.natural <= 0 or self.test.template != 0:
            raise ValueError(
                "ISETrace test split must request natural > 0 and template = 0"
            )
        return self


class ISETraceTrajectoriesConfig(ClosedModel):
    splits: ISETraceTrajectorySplitCounts


class EvidenceDatasetConfig(ClosedModel):
    name: Literal["hotpotqa", "twowiki", "musique"]
    strict_invalid_examples: StrictBool = False
    splits: DatasetSplitsConfig


class ISETraceDatasetConfig(ClosedModel):
    name: Literal["isetrace"]
    trajectory_source: Path
    natural_query_source: Path
    trajectories: ISETraceTrajectoriesConfig
    chunking: ISETraceChunkingConfig


DatasetConfig: TypeAlias = Annotated[
    Union[EvidenceDatasetConfig, ISETraceDatasetConfig],
    Field(discriminator="name"),
]


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
    provenance_rgcn: RgcnProfileSettings
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


class RgcnModelConfig(ClosedModel):
    hidden_dim: PositiveInt
    num_layers: NonNegativeInt
    dropout: Annotated[ScientificFloat, Field(ge=0.0, lt=1.0)]
    ablation: str = Field(min_length=1)


class RgcnTrainerConfig(RgcnTrainingConfig):
    device: Device


class RgcnTrainConfig(ClosedModel):
    model: RgcnModelConfig
    trainer: RgcnTrainerConfig
    selection: RgcnSelectionSettings


def _effective_rgcn_parts(
    pairs: NegativeSamplingConfig,
    train: RgcnTrainConfig,
    variant: EvidenceRgcnVariant,
) -> tuple[NegativeSamplingConfig, RgcnTrainConfig]:
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
    pairs: NegativeSamplingConfig
    train: RgcnTrainConfig

    def for_variant(self, variant: EvidenceRgcnVariant) -> RgcnStageConfig:
        pairs, train = _effective_rgcn_parts(self.pairs, self.train, variant)
        return self.model_copy(update={"pairs": pairs, "train": train})


class RgcnMethodConfig(RgcnStageConfig):
    method: Literal["dense_rgcn_graph_retriever"]
    variant: EvidenceRgcnVariant = "full_rgcn"

    def effective(self) -> RgcnMethodConfig:
        stage = super().for_variant(self.variant)
        return self.model_copy(update={"pairs": stage.pairs, "train": stage.train})


class ProvenanceRgcnMethodConfig(RgcnStageConfig):
    method: Literal["provenance_rgcn"]
    variant: ProvenanceRgcnVariant = "full_rgcn"

    def effective(self) -> "ProvenanceRgcnMethodConfig":
        stage = super().for_variant(self.variant)
        return self.model_copy(update={"pairs": stage.pairs, "train": stage.train})


class DenseFinetuneTrainerConfig(DenseFinetuneTrainerSettings):
    device: Device


class DenseFinetuneTrainConfig(ClosedModel):
    data: DenseFinetuneDataSettings
    trainer: DenseFinetuneTrainerConfig
    selection: DenseFinetuneSelectionSettings


class DenseFinetuneMethodConfig(ClosedModel):
    method: Literal["dense_ft"]
    encoder: DenseEncoderConfig
    train: DenseFinetuneTrainConfig
    pairs: NegativeSamplingConfig

    def effective_for_dataset(self, dataset: DatasetName) -> DenseFinetuneMethodConfig:
        if dataset != "isetrace":
            return self
        return self.model_copy(
            update={
                "pairs": self.pairs.model_copy(
                    update={"hard_graph_neighbor_per_positive": 0}
                )
            }
        )


class DenseFtRgcnMethodConfig(ClosedModel):
    method: Literal["dense_ft_rgcn_graph_retriever"]
    variant: EvidenceRgcnVariant = "full_rgcn"
    seed: DenseFinetuneMethodConfig
    rgcn: RgcnStageConfig

    def effective_rgcn(self) -> RgcnStageConfig:
        return self.rgcn.for_variant(self.variant)


MethodConfig: TypeAlias = Annotated[
    Union[
        Bm25MethodConfig,
        DenseMethodConfig,
        GraphRAGMethodConfig,
        ProvenancePathMethodConfig,
        ProvenanceRgcnMethodConfig,
        RgcnMethodConfig,
        DenseFinetuneMethodConfig,
        DenseFtRgcnMethodConfig,
    ],
    Field(discriminator="method"),
]


class PairBuildConfig(ClosedModel):
    method: Literal[
        "dense_ft",
        "dense_rgcn_graph_retriever",
        "dense_ft_rgcn_graph_retriever",
        "provenance_rgcn",
    ]
    sampling: NegativeSamplingConfig
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
    trajectory_splits: ISETraceTrajectorySplitCounts | None = None
    chunking: ISETraceChunkingConfig | None = None


class CacheConfig(ClosedModel):
    refresh: StrictBool = False


class EncodingConfig(ClosedModel):
    """Runtime controls for frozen R-GCN encoding; not scientific identity."""

    enable_gpupool: StrictBool
    chunk_size: PositiveInt


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
    graph: GraphBuildConfig
    tracking: TrackingConfig


class ResolvedRawSplitConfig(ClosedModel):
    kind: Literal["raw"]
    source: Path
    offset: NonNegativeInt
    capacity: PositiveInt | None = None
    count: PositiveInt | None = None


class ResolvedDatasetConfig(ClosedModel):
    name: DatasetName
    strict_invalid_examples: StrictBool
    trajectory_source: Path | None = None
    natural_query_source: Path | None = None
    source_revision: str | None = None
    trajectories: ISETraceTrajectoriesConfig | None = None
    chunking: ISETraceChunkingConfig | None = None
    splits: dict[SplitName, ResolvedRawSplitConfig]


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
    graph: GraphBuildConfig
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
    resolved_splits: dict[SplitName, ResolvedRawSplitConfig] = {}
    if isinstance(config.dataset, ISETraceDatasetConfig):
        natural_source = _absolute_path(root, config.dataset.natural_query_source)
        for split_name in split_names:
            policy = getattr(config.profile.splits, split_name)
            count = (
                policy.count
                if config.profile.name != "full"
                and isinstance(policy, FixedCountPolicy)
                else None
            )
            resolved_splits[split_name] = ResolvedRawSplitConfig(
                kind="raw",
                source=natural_source,
                offset=0,
                capacity=None,
                count=count,
            )
    else:
        for split_name in split_names:
            dataset_split = getattr(config.dataset.splits, split_name)
            if dataset_split is None:
                continue
            policy = getattr(config.profile.splits, split_name)
            if isinstance(policy, FixedCountPolicy):
                count = policy.count
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
    if isinstance(config.dataset, ISETraceDatasetConfig):
        from graph_memory.datasets.isetrace.registration import ISETRACE_REVISION

        strict_invalid_examples = False
        trajectory_source = _absolute_path(root, config.dataset.trajectory_source)
        natural_query_source = _absolute_path(root, config.dataset.natural_query_source)
        registered_revision: str | None = ISETRACE_REVISION
        trajectories: ISETraceTrajectoriesConfig | None = config.dataset.trajectories
        chunking: ISETraceChunkingConfig | None = config.dataset.chunking
    else:
        strict_invalid_examples = config.dataset.strict_invalid_examples
        trajectory_source = None
        natural_query_source = None
        registered_revision = None
        trajectories = None
        chunking = None

    resolved_method = _resolve_method_config(config.method, config.dataset.name)
    return ResolvedExperimentConfig(
        name=config.name,
        dataset=ResolvedDatasetConfig(
            name=config.dataset.name,
            strict_invalid_examples=strict_invalid_examples,
            trajectory_source=trajectory_source,
            natural_query_source=natural_query_source,
            source_revision=registered_revision,
            trajectories=trajectories,
            chunking=chunking,
            splits=resolved_splits,
        ),
        profile=config.profile.name,
        method=resolved_method,
        seed=config.seed,
        split_seed=config.split_seed,
        device=config.device,
        top_k=config.top_k,
        cache=config.cache,
        encoding=config.encoding,
        graph=config.graph,
        tracking=ResolvedTrackingConfig(
            database=_absolute_path(root, config.tracking.database),
            artifact_root=_absolute_path(root, config.tracking.artifact_root),
            experiment_name=config.tracking.experiment_name,
        ),
    )


def _resolve_method_config(
    method: MethodConfig,
    dataset: DatasetName,
) -> MethodConfig:
    if isinstance(method, DenseFinetuneMethodConfig):
        return method.effective_for_dataset(dataset)
    if isinstance(method, (RgcnMethodConfig, ProvenanceRgcnMethodConfig)):
        return method.effective()
    if isinstance(method, DenseFtRgcnMethodConfig):
        return method.model_copy(update={"rgcn": method.effective_rgcn()})
    return method


def _check_dataset_method_compatibility(
    dataset: DatasetName,
    method: MethodConfig,
) -> None:
    is_provenance = dataset == "isetrace"
    provenance_only = isinstance(
        method, (ProvenancePathMethodConfig, ProvenanceRgcnMethodConfig)
    )
    evidence_only = isinstance(method, (RgcnMethodConfig, DenseFtRgcnMethodConfig))
    if provenance_only and not is_provenance:
        raise ValueError(
            f"dataset={dataset!r} does not support provenance method={method.method!r}."
        )
    if evidence_only and is_provenance:
        raise ValueError(
            f"dataset={dataset!r} does not support evidence method={method.method!r}."
        )


def _require_method_splits(
    method: MethodConfig,
    splits: dict[SplitName, ResolvedRawSplitConfig],
) -> None:
    required: set[SplitName] = {"test"}
    if isinstance(
        method,
        (
            DenseFinetuneMethodConfig,
            ProvenanceRgcnMethodConfig,
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
    "Bm25MethodConfig",
    "CacheConfig",
    "ClosedModel",
    "CountPolicy",
    "DatasetConfig",
    "DatasetName",
    "DenseEncoderConfig",
    "DenseFinetuneMethodConfig",
    "DenseFinetuneTrainConfig",
    "DenseFinetuneTrainerConfig",
    "DenseFtRgcnMethodConfig",
    "DenseMethodConfig",
    "Device",
    "EncodingConfig",
    "ExperimentConfig",
    "FixedCountPolicy",
    "GraphBuildConfig",
    "GraphRAGMethodConfig",
    "ISETraceChunkingConfig",
    "ISETraceDatasetConfig",
    "ISETraceTrajectoriesConfig",
    "ISETraceTrajectoryOriginCounts",
    "ISETraceTrajectorySplitCounts",
    "MethodConfig",
    "NonNegativeFloat",
    "NonNegativeInt",
    "PairBuildConfig",
    "PositiveFloat",
    "PositiveInt",
    "ProfileConfig",
    "ProvenancePathMethodConfig",
    "ProvenanceRgcnMethodConfig",
    "ProvenanceRgcnVariant",
    "PrepareSplitConfig",
    "ResolvedExperimentConfig",
    "RgcnMethodConfig",
    "RgcnModelConfig",
    "RgcnStageConfig",
    "RgcnTrainConfig",
    "RgcnTrainerConfig",
    "ScientificFloat",
    "ScientificInt",
    "SplitName",
    "TrackingConfig",
    "resolve_experiment_config",
    "parse_composed_config",
]

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal, TypeAlias, Union

from pydantic import Field

from graph_memory.experiment.config import (
    ClosedModel,
    DatasetName,
    DenseEncoderConfig,
    DenseFinetuneTrainConfig,
    GraphBuildConfig,
    GraphRerankSearchSpace,
    MemoryStreamScoringConfig,
    MemoryStreamSearchSpace,
    NonNegativeInt,
    PairSamplingConfig,
    PositiveInt,
    RgcnTrainConfig,
    ScientificInt,
    SplitName,
)
from graph_memory.registry.retrieval import RetrievalMethodId


class PrepareOutputs(ClosedModel):
    input: Path
    labels: Path
    combined: Path
    summary: Path


class RawPrepareStageConfig(ClosedModel):
    stage: Literal["prepare"]
    kind: Literal["raw"]
    dataset: DatasetName
    split: SplitName
    source: Path
    outputs: PrepareOutputs
    count: PositiveInt
    seed: ScientificInt
    offset: NonNegativeInt
    strict_invalid_examples: bool


class ImportancePrepareStageConfig(ClosedModel):
    stage: Literal["prepare"]
    kind: Literal["importance"]
    dataset: Literal["hotpotqa"]
    split: Literal["dev", "test"]
    canonical_inputs: Path
    canonical_labels: Path
    importance: Path
    outputs: PrepareOutputs
    count: PositiveInt
    offset: NonNegativeInt


PrepareStageConfig: TypeAlias = Annotated[
    Union[RawPrepareStageConfig, ImportancePrepareStageConfig],
    Field(discriminator="kind"),
]


class GraphStageConfig(ClosedModel):
    stage: Literal["graphs"]
    dataset: DatasetName
    split: SplitName
    tasks: Path
    output: Path
    summary: Path
    graph: GraphBuildConfig


class PairOutputs(ClosedModel):
    pairs: Path
    pair_summary: Path
    summary: Path


class PairStageConfig(ClosedModel):
    stage: Literal["pairs"]
    dataset: DatasetName
    method: Literal[
        "dense_rgcn_graph_retriever",
        "dense_ft",
        "dense_ft_rgcn_graph_retriever",
    ]
    variant: str | None
    tasks: Path
    labels: Path
    graphs: Path
    outputs: PairOutputs
    sampling: PairSamplingConfig
    hard_dense_encoder: DenseEncoderConfig


class GraphRerankTuneStageConfig(ClosedModel):
    stage: Literal["tune"]
    kind: Literal["graph_rerank"]
    dataset: DatasetName
    method: Literal["bm25_graph_rerank", "dense_graph_rerank"]
    tasks: Path
    labels: Path
    graphs: Path
    selected_config: Path
    candidates: Path
    summary: Path
    top_k: PositiveInt
    seed_encoder: DenseEncoderConfig | None
    search_space: GraphRerankSearchSpace


class MemoryStreamTuneStageConfig(ClosedModel):
    stage: Literal["tune"]
    kind: Literal["memory_stream"]
    dataset: Literal["hotpotqa"]
    method: Literal["memory_stream"]
    tasks: Path
    labels: Path
    graphs: Path
    importance: Path
    selected_config: Path
    candidates: Path
    summary: Path
    top_k: PositiveInt
    encoder: DenseEncoderConfig
    search_space: MemoryStreamSearchSpace


TuneStageConfig: TypeAlias = Annotated[
    Union[GraphRerankTuneStageConfig, MemoryStreamTuneStageConfig],
    Field(discriminator="kind"),
]


class RgcnTrainStageConfig(ClosedModel):
    stage: Literal["train"]
    method: Literal["dense_rgcn_graph_retriever", "dense_ft_rgcn_graph_retriever"]
    variant: str | None
    dataset: DatasetName
    train_tasks: Path
    train_labels: Path
    train_graphs: Path
    train_pairs: Path
    dev_tasks: Path
    dev_labels: Path
    dev_graphs: Path
    output_dir: Path
    checkpoint_dir: Path
    metrics: Path
    summary: Path
    seed_checkpoint: Path | None
    encoder: DenseEncoderConfig
    pairs: PairSamplingConfig
    train: RgcnTrainConfig


class DenseFinetuneTrainStageConfig(ClosedModel):
    stage: Literal["train"]
    method: Literal["dense_ft"]
    variant: str | None
    dataset: DatasetName
    train_tasks: Path
    train_labels: Path
    train_pairs: Path
    dev_tasks: Path
    dev_labels: Path
    output_dir: Path
    model_dir: Path
    metrics: Path
    summary: Path
    encoder: DenseEncoderConfig
    pairs: PairSamplingConfig
    train: DenseFinetuneTrainConfig


TrainStageConfig: TypeAlias = Annotated[
    Union[RgcnTrainStageConfig, DenseFinetuneTrainStageConfig],
    Field(discriminator="method"),
]


class Bm25RetrieveStageConfig(ClosedModel):
    stage: Literal["retrieve"]
    method: Literal["bm25"]
    variant: str | None
    dataset: DatasetName
    tasks: Path
    output: Path
    summary: Path
    top_k: PositiveInt


class DenseRetrieveStageConfig(ClosedModel):
    stage: Literal["retrieve"]
    method: Literal["dense"]
    variant: str | None
    dataset: DatasetName
    tasks: Path
    output: Path
    summary: Path
    top_k: PositiveInt
    encoder: DenseEncoderConfig


class MemoryStreamRetrieveStageConfig(ClosedModel):
    stage: Literal["retrieve"]
    method: Literal["memory_stream"]
    variant: str | None
    dataset: Literal["hotpotqa"]
    tasks: Path
    output: Path
    summary: Path
    top_k: PositiveInt
    encoder: DenseEncoderConfig
    selected_config: Path
    importance: Path
    scoring: MemoryStreamScoringConfig
    capped_test_count: PositiveInt


class Bm25GraphRerankRetrieveStageConfig(ClosedModel):
    stage: Literal["retrieve"]
    method: Literal["bm25_graph_rerank"]
    variant: str | None
    dataset: DatasetName
    tasks: Path
    graphs: Path
    output: Path
    summary: Path
    top_k: PositiveInt
    selected_config: Path
    seed_method: Literal["bm25"]


class DenseGraphRerankRetrieveStageConfig(ClosedModel):
    stage: Literal["retrieve"]
    method: Literal["dense_graph_rerank"]
    variant: str | None
    dataset: DatasetName
    tasks: Path
    graphs: Path
    output: Path
    summary: Path
    top_k: PositiveInt
    selected_config: Path
    seed_method: Literal["dense"]
    encoder: DenseEncoderConfig


class RgcnRetrieveStageConfig(ClosedModel):
    stage: Literal["retrieve"]
    method: Literal["dense_rgcn_graph_retriever", "dense_ft_rgcn_graph_retriever"]
    variant: str | None
    dataset: DatasetName
    tasks: Path
    graphs: Path
    output: Path
    summary: Path
    top_k: PositiveInt
    checkpoint: Path
    device: str


class DenseFinetuneRetrieveStageConfig(ClosedModel):
    stage: Literal["retrieve"]
    method: Literal["dense_ft"]
    variant: str | None
    dataset: DatasetName
    tasks: Path
    output: Path
    summary: Path
    top_k: PositiveInt
    model_dir: Path
    device: str


RetrieveStageConfig: TypeAlias = Annotated[
    Union[
        Bm25RetrieveStageConfig,
        DenseRetrieveStageConfig,
        MemoryStreamRetrieveStageConfig,
        Bm25GraphRerankRetrieveStageConfig,
        DenseGraphRerankRetrieveStageConfig,
        RgcnRetrieveStageConfig,
        DenseFinetuneRetrieveStageConfig,
    ],
    Field(discriminator="method"),
]


class EvaluateStageConfig(ClosedModel):
    stage: Literal["evaluate"]
    dataset: DatasetName
    method: str
    variant: str | None
    predictions: Path
    labels: Path
    graphs: Path
    metrics: Path
    failure_cases: Path
    summary: Path
    failure_case_limit: NonNegativeInt
    top_k: PositiveInt


class AblationSelection(ClosedModel):
    method: RetrievalMethodId
    variant: str = Field(min_length=1)


class AggregateStageBase(ClosedModel):
    stage: Literal["aggregate"]
    metrics: list[Path]
    main: Path
    path: Path
    efficiency: Path
    summary: Path


class OrdinaryAggregateStageConfig(AggregateStageBase):
    kind: Literal["ordinary"]


class AblationAggregateStageConfig(AggregateStageBase):
    kind: Literal["ablation"]
    ablation_metrics: list[Path]
    ablation_index: Path
    ablation: Path
    ablation_selections: list[AblationSelection]


AggregateStageConfig: TypeAlias = Annotated[
    Union[OrdinaryAggregateStageConfig, AblationAggregateStageConfig],
    Field(discriminator="kind"),
]


StageConfig: TypeAlias = (
    PrepareStageConfig
    | GraphStageConfig
    | PairStageConfig
    | TuneStageConfig
    | TrainStageConfig
    | RetrieveStageConfig
    | EvaluateStageConfig
    | AggregateStageConfig
)


__all__ = [
    "AggregateStageConfig",
    "AblationAggregateStageConfig",
    "AblationSelection",
    "Bm25GraphRerankRetrieveStageConfig",
    "Bm25RetrieveStageConfig",
    "DenseFinetuneRetrieveStageConfig",
    "DenseFinetuneTrainStageConfig",
    "DenseGraphRerankRetrieveStageConfig",
    "DenseRetrieveStageConfig",
    "EvaluateStageConfig",
    "GraphRerankTuneStageConfig",
    "GraphStageConfig",
    "ImportancePrepareStageConfig",
    "MemoryStreamRetrieveStageConfig",
    "MemoryStreamTuneStageConfig",
    "OrdinaryAggregateStageConfig",
    "PairStageConfig",
    "PrepareStageConfig",
    "RawPrepareStageConfig",
    "RetrieveStageConfig",
    "RgcnRetrieveStageConfig",
    "RgcnTrainStageConfig",
    "StageConfig",
    "TrainStageConfig",
    "TuneStageConfig",
]

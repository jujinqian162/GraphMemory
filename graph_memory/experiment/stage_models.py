from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal, TypeAlias, Union

from pydantic import Field

from graph_memory.experiment.config import (
    ClosedModel,
    DatasetName,
    Device,
    DenseEncoderConfig,
    DenseFinetuneTrainConfig,
    GraphBuildConfig,
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


PrepareStageConfig: TypeAlias = RawPrepareStageConfig


class EvidenceGraphStageConfig(ClosedModel):
    stage: Literal["evidence_graphs"]
    dataset: DatasetName
    split: SplitName
    tasks: Path
    output: Path
    evidence_graph: GraphBuildConfig


class PairOutputs(ClosedModel):
    pairs: Path
    pair_summary: Path


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
    evidence_graphs: Path
    outputs: PairOutputs
    sampling: PairSamplingConfig
    hard_dense_encoder: DenseEncoderConfig
    device: Device


class RgcnTrainStageBase(ClosedModel):
    stage: Literal["train"]
    variant: str | None
    dataset: DatasetName
    train_tasks: Path
    train_labels: Path
    train_evidence_graphs: Path
    train_pairs: Path
    dev_tasks: Path
    dev_labels: Path
    dev_evidence_graphs: Path
    output_dir: Path
    checkpoint_dir: Path
    metrics: Path
    encoder: DenseEncoderConfig
    pairs: PairSamplingConfig
    train: RgcnTrainConfig


class OrdinaryRgcnTrainStageConfig(RgcnTrainStageBase):
    method: Literal["dense_rgcn_graph_retriever"]


class SeededRgcnTrainStageConfig(RgcnTrainStageBase):
    method: Literal["dense_ft_rgcn_graph_retriever"]
    seed_model_dir: Path


RgcnTrainStageConfig: TypeAlias = (
    OrdinaryRgcnTrainStageConfig | SeededRgcnTrainStageConfig
)


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
    encoder: DenseEncoderConfig
    pairs: PairSamplingConfig
    train: DenseFinetuneTrainConfig


TrainStageConfig: TypeAlias = Annotated[
    Union[
        OrdinaryRgcnTrainStageConfig,
        SeededRgcnTrainStageConfig,
        DenseFinetuneTrainStageConfig,
    ],
    Field(discriminator="method"),
]


class Bm25RetrieveStageConfig(ClosedModel):
    stage: Literal["retrieve"]
    method: Literal["bm25"]
    variant: str | None
    dataset: DatasetName
    tasks: Path
    output: Path
    top_k: PositiveInt


class DenseRetrieveStageConfig(ClosedModel):
    stage: Literal["retrieve"]
    method: Literal["dense"]
    variant: str | None
    dataset: DatasetName
    tasks: Path
    output: Path
    top_k: PositiveInt
    encoder: DenseEncoderConfig


class GraphRAGRetrieveStageConfig(ClosedModel):
    stage: Literal["retrieve"]
    method: Literal["graphrag"]
    variant: str | None
    dataset: DatasetName
    tasks: Path
    output: Path
    top_k: PositiveInt
    encoder: DenseEncoderConfig
    seed_top_s: PositiveInt
    restart_probability: float
    max_iterations: PositiveInt
    convergence_tolerance: float
    semantic_weight: float
    entity_weight: float


class ExecutionProvenanceRetrieveStageConfig(ClosedModel):
    stage: Literal["retrieve"]
    method: Literal["execution_provenance_retriever"]
    variant: str | None
    dataset: DatasetName
    tasks: Path
    output: Path
    top_k: PositiveInt
    encoder: DenseEncoderConfig
    seed_top_s: PositiveInt
    max_hops: PositiveInt
    top_paths: PositiveInt
    max_path_expansions: PositiveInt
    semantic_weight: float
    dependency_weight: float
    binding_weight: float
    grounding_weight: float
    hop_penalty: float
    invalidation_penalty: float


class RgcnRetrieveStageConfig(ClosedModel):
    stage: Literal["retrieve"]
    method: Literal["dense_rgcn_graph_retriever", "dense_ft_rgcn_graph_retriever"]
    variant: str | None
    dataset: DatasetName
    tasks: Path
    evidence_graphs: Path
    output: Path
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
    top_k: PositiveInt
    model_dir: Path
    device: str


RetrieveStageConfig: TypeAlias = Annotated[
    Union[
        Bm25RetrieveStageConfig,
        DenseRetrieveStageConfig,
        GraphRAGRetrieveStageConfig,
        ExecutionProvenanceRetrieveStageConfig,
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
    evidence_graphs: Path | None
    metrics: Path
    failure_cases: Path
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
    | EvidenceGraphStageConfig
    | PairStageConfig
    | TrainStageConfig
    | RetrieveStageConfig
    | EvaluateStageConfig
    | AggregateStageConfig
)


__all__ = [
    "AggregateStageConfig",
    "AblationAggregateStageConfig",
    "AblationSelection",
    "Bm25RetrieveStageConfig",
    "DenseFinetuneRetrieveStageConfig",
    "DenseFinetuneTrainStageConfig",
    "DenseRetrieveStageConfig",
    "EvidenceGraphStageConfig",
    "EvaluateStageConfig",
    "GraphRAGRetrieveStageConfig",
    "OrdinaryAggregateStageConfig",
    "OrdinaryRgcnTrainStageConfig",
    "PairStageConfig",
    "PrepareStageConfig",
    "RawPrepareStageConfig",
    "RetrieveStageConfig",
    "RgcnRetrieveStageConfig",
    "RgcnTrainStageConfig",
    "SeededRgcnTrainStageConfig",
    "StageConfig",
    "TrainStageConfig",
]

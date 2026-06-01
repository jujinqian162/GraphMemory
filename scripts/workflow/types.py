from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Mapping, TypeVar


class StageId(StrEnum):
    """Closed lifecycle stages. / 封闭的实验生命周期阶段。"""

    PREPARE = "prepare"
    GRAPHS = "graphs"
    PAIRS = "pairs"
    TUNE = "tune"
    TRAIN = "train"
    RETRIEVE = "retrieve"
    EVALUATE = "evaluate"
    AGGREGATE = "aggregate"


class WorkflowId(StrEnum):
    """Registered workflow families. / 已注册的 workflow 类型。"""

    STATELESS_RETRIEVAL = "stateless_retrieval"
    GRAPH_RERANK = "graph_rerank"
    RGCN_TRAINABLE_RETRIEVAL = "rgcn_trainable_retrieval"


class ArtifactRole(StrEnum):
    """Semantic artifacts used by workflow steps. / workflow 步骤使用的语义 artifact。"""

    INPUTS = "inputs"
    LABELS = "labels"
    GRAPHS = "graphs"
    TRAIN_PAIRS = "train_pairs"
    TRAIN_PAIR_SUMMARY = "train_pair_summary"
    TRAIN_PAIR_RUN_SUMMARY = "train_pair_run_summary"
    EFFECTIVE_TRAINING_CONFIG = "effective_training_config"
    TRAIN_METRICS = "train_metrics"
    TRAIN_RUN_SUMMARY = "train_run_summary"
    TUNED_CONFIG = "tuned_config"
    CHECKPOINT = "checkpoint"
    PREDICTIONS = "predictions"
    METRICS = "metrics"
    FAILURE_CASES = "failure_cases"
    MAIN_TABLE = "main_table"
    ABLATION_TABLE = "ablation_table"


class ChangeDimension(StrEnum):
    """Variant dimensions that invalidate workflow work. / variant 使 workflow 失效的维度。"""

    PAIR_SAMPLING = "pair_sampling"
    MODEL_STRUCTURE = "model_structure"
    MODEL_GRAPH_VIEW = "model_graph_view"


class ArtifactState(StrEnum):
    """Observable artifact states. / 可观测 artifact 状态。"""

    MISSING = "missing"
    COMPLETE = "complete"
    STALE = "stale"
    ALIAS = "alias"


class RgcnAblationVariant(StrEnum):
    """Supported first-table R-GCN variants. / 首张 R-GCN 消融表支持的 variant。"""

    FULL_RGCN = "full_rgcn"
    WO_BRIDGE = "wo_bridge"
    WO_ENTITY_OVERLAP = "wo_entity_overlap"
    WO_SEQUENTIAL = "wo_sequential"
    WO_QUERY_OVERLAP = "wo_query_overlap"
    WO_GRAPH = "wo_graph"
    WO_EDGE_TYPE = "wo_edge_type"
    WO_EDGE_WEIGHT = "wo_edge_weight"
    WO_SEED_SCORE = "wo_seed_score"
    WO_HARD_NEGATIVES = "wo_hard_negatives"


@dataclass(frozen=True)
class WorkflowStepSpec:
    """One declarative lifecycle step. / 一个声明式生命周期步骤。"""

    stage: StageId
    inputs: tuple[ArtifactRole, ...]
    outputs: tuple[ArtifactRole, ...]
    invalidated_by: frozenset[ChangeDimension] = frozenset()
    command_adapter: str | None = None


@dataclass(frozen=True)
class WorkflowSpec:
    """An ordered method-family lifecycle. / 一个有序的方法族生命周期。"""

    identifier: WorkflowId
    steps: tuple[WorkflowStepSpec, ...]


@dataclass(frozen=True)
class VariantSpec:
    """A suite variant and its minimal config diff. / suite variant 及其最小配置差异。"""

    identifier: RgcnAblationVariant
    changed_dimensions: frozenset[ChangeDimension]
    training_config_override: Mapping[str, Any] = field(default_factory=dict)
    baseline_alias: bool = False


@dataclass(frozen=True)
class AblationSuiteSpec:
    """Variants registered for one public method. / 为一个公开方法注册的 variants。"""

    method: str
    variants: tuple[VariantSpec, ...]


@dataclass(frozen=True)
class RunUnit:
    """One ordinary or variant execution unit. / 一个普通或 variant 执行单元。"""

    method: str
    variant: str | None = None

    @property
    def qualifier(self) -> str:
        return self.method if self.variant is None else f"{self.method}/{self.variant}"


@dataclass(frozen=True)
class ArtifactAlias:
    """A variant artifact reference reused from main. / 从主实验复用的 variant artifact 引用。"""

    role: ArtifactRole
    source: str
    target: str


@dataclass(frozen=True)
class VariantArtifactNamespace:
    """Resolved variant paths and reusable aliases. / variant 路径与可复用 alias。"""

    method: str
    variant: str
    invalidated_from: StageId | None
    paths: Mapping[ArtifactRole, str]
    local_paths: Mapping[ArtifactRole, str]
    aliases: tuple[ArtifactAlias, ...]

    def path(self, role: ArtifactRole) -> str:
        return self.paths[role]


@dataclass(frozen=True)
class StageCommand:
    """A concrete low-level command. / 一个具体的底层脚本命令。"""

    stage: StageId
    argv: list[str]
    method: str | None = None
    split: str | None = None
    variant: str | None = None


@dataclass(frozen=True)
class ConfigEntry:
    """One discoverable config. / 一个可发现的配置。"""

    kind: str
    name: str
    path: str


EnumValue = TypeVar("EnumValue", bound=StrEnum)


def parse_closed_value(enum_type: type[EnumValue], value: str, *, label: str) -> EnumValue:
    """Parse a serialized closed value and report its allowed choices."""

    try:
        return enum_type(value)
    except ValueError as error:
        allowed = ", ".join(member.value for member in enum_type)
        raise ValueError(f"Unknown {label}={value!r}; allowed values: {allowed}") from error

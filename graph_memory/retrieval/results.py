from __future__ import annotations

from pydantic import JsonValue, model_validator

from graph_memory.contracts.model import (
    DomainModel,
    FiniteFloat,
    NonEmptyStr,
    NonNegativeFiniteFloat,
    NonNegativeInt,
)
from graph_memory.graphs.contracts import GraphEdge
from graph_memory.retrieval.contracts import NativeRetrievalTrace
from graph_memory.retrieval.methods.ids import RetrievalMethodId
from graph_memory.trajectories import SourceSpan


class RankedNodeRecord(DomainModel):
    node_id: NonEmptyStr
    score: FiniteFloat
    source_spans: tuple[SourceSpan, ...] = ()
    token_count: NonNegativeInt = 0


class RetrievedSubgraph(DomainModel):
    nodes: tuple[NonEmptyStr, ...]
    edges: tuple[GraphEdge, ...]

    @model_validator(mode="after")
    def _validate_subgraph(self) -> "RetrievedSubgraph":
        if len(self.nodes) != len(set(self.nodes)):
            raise ValueError("retrieved subgraph node IDs must be unique")
        valid = set(self.nodes) | {"q"}
        for edge in self.edges:
            if edge.source not in valid or edge.target not in valid:
                raise ValueError(
                    f"retrieved edge {edge.source}->{edge.target} leaves subgraph"
                )
        return self


class RankedResultMetadata(DomainModel):
    native_trace: NativeRetrievalTrace | None = None


class RankedResult(DomainModel):
    task_id: NonEmptyStr
    method: RetrievalMethodId
    ranked_nodes: tuple[RankedNodeRecord, ...]
    retrieved_subgraph: RetrievedSubgraph
    latency_ms: NonNegativeFiniteFloat
    input_tokens: NonNegativeInt
    metadata: RankedResultMetadata | None = None
    debug: dict[str, JsonValue] | None = None

    @model_validator(mode="after")
    def _validate_ranking(self) -> "RankedResult":
        node_ids = [node.node_id for node in self.ranked_nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("ranked node IDs must be unique")
        scores = [node.score for node in self.ranked_nodes]
        if any(current > previous for previous, current in zip(scores, scores[1:])):
            raise ValueError("ranked nodes must be sorted descending")
        return self

    @property
    def ranked_node_ids(self) -> tuple[str, ...]:
        return tuple(node.node_id for node in self.ranked_nodes)


__all__ = [
    "RankedNodeRecord",
    "RankedResult",
    "RankedResultMetadata",
    "RetrievedSubgraph",
]

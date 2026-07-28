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
from graph_memory.retrieval.requests import TextRankingRequest


class RankedNodeRecord(DomainModel):
    node_id: NonEmptyStr
    score: FiniteFloat


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


class RankedResultEnvelope(DomainModel):
    request: TextRankingRequest
    result: RankedResult

    @model_validator(mode="after")
    def _validate_context(self) -> "RankedResultEnvelope":
        if self.result.task_id != self.request.task_id:
            raise ValueError("ranked result task_id must match request task_id")
        valid = self.request.candidate_ids
        observed = set(self.result.ranked_node_ids)
        if observed != valid or len(self.result.ranked_node_ids) != len(valid):
            missing = sorted(valid - observed)
            extra = sorted(observed - valid)
            raise ValueError(
                "ranking must include every candidate exactly once; "
                f"missing={missing} extra={extra}"
            )
        unknown_subgraph = sorted(
            set(self.result.retrieved_subgraph.nodes) - valid - {"q"}
        )
        if unknown_subgraph:
            raise ValueError(
                f"retrieved subgraph references unknown candidates={unknown_subgraph}"
            )
        metadata = self.result.metadata
        if metadata is not None and metadata.native_trace is not None:
            metadata.native_trace.validate_candidate_context(valid)
        return self


class RankedResultBatch(DomainModel):
    requests: tuple[TextRankingRequest, ...]
    results: tuple[RankedResult, ...]

    @model_validator(mode="after")
    def _validate_batch(self) -> "RankedResultBatch":
        request_by_id = {request.task_id: request for request in self.requests}
        if len(request_by_id) != len(self.requests):
            raise ValueError("ranking request task IDs must be unique")
        result_by_id = {result.task_id: result for result in self.results}
        if len(result_by_id) != len(self.results):
            raise ValueError("ranked result task IDs must be unique")
        if set(request_by_id) != set(result_by_id):
            missing = sorted(set(request_by_id) - set(result_by_id))
            extra = sorted(set(result_by_id) - set(request_by_id))
            raise ValueError(
                f"ranked result task coverage mismatch; missing={missing} extra={extra}"
            )
        for task_id, result in result_by_id.items():
            RankedResultEnvelope(request=request_by_id[task_id], result=result)
        return self


__all__ = [
    "RankedNodeRecord",
    "RankedResult",
    "RankedResultBatch",
    "RankedResultEnvelope",
    "RankedResultMetadata",
    "RetrievedSubgraph",
]

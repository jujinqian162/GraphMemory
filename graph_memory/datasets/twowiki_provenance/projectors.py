from __future__ import annotations

from collections.abc import Sequence

from graph_memory.contracts.graphs import EvidenceGraph
from graph_memory.contracts.ranking import RankedResult
from graph_memory.datasets.twowiki_provenance.records import (
    ProvenanceGraphRecord,
    TwoWikiProvenanceLabelRecord,
    TwoWikiProvenanceRankingRecord,
)
from graph_memory.evaluation.requests import EvidenceEvaluationRequest, EvidenceLabel
from graph_memory.graphs.provenance import (
    ExecutionProvenanceEdge,
    ExecutionProvenanceGraph,
    ExecutionProvenanceNode,
    FieldBinding,
    ProvenanceEdgeType,
    ProvenanceNodeType,
)
from graph_memory.retrieval.requests import (
    ExecutionProvenanceRankingRequest,
    TextCandidate,
    TextRankingRequest,
)


class TwoWikiProvenanceToTextRankingRequest:
    def project(self, record: TwoWikiProvenanceRankingRecord) -> TextRankingRequest:
        return TextRankingRequest(
            task_id=record["task_id"],
            query_text=record["question"],
            candidates=tuple(
                TextCandidate(
                    item_id=candidate["output_id"],
                    text=f"{candidate['title']}. {candidate['text']}",
                    metadata={
                        "title": candidate["title"],
                        "source_ref": candidate["title"],
                        "sequence_index": candidate["sentence_index"],
                        "position": candidate["position"],
                        "question_type": record["question_type"],
                    },
                )
                for candidate in record["candidates"]
            ),
        )


class TwoWikiProvenanceToExecutionProvenanceRankingRequest:
    def project(
        self, record: TwoWikiProvenanceRankingRecord
    ) -> ExecutionProvenanceRankingRequest:
        text_request = TwoWikiProvenanceToTextRankingRequest().project(record)
        return ExecutionProvenanceRankingRequest(
            task_id=record["task_id"],
            query_text=record["question"],
            candidates=text_request.candidates,
            graph=provenance_graph_from_record(record["graph"]),
        )


class TwoWikiProvenanceToEvidenceEvaluationRequest:
    def project(
        self,
        *,
        predictions: Sequence[RankedResult],
        labels: Sequence[TwoWikiProvenanceLabelRecord],
        graphs: Sequence[EvidenceGraph],
    ) -> EvidenceEvaluationRequest:
        return EvidenceEvaluationRequest(
            predictions=predictions,
            labels=tuple(
                EvidenceLabel(
                    task_id=label["task_id"],
                    gold_answer=label["gold_answer"],
                    gold_evidence_item_ids=tuple(label["gold_evidence_output_ids"]),
                    gold_dependency_edges=tuple(
                        _dependency_edge(edge)
                        for edge in label["gold_dependency_edges"]
                    ),
                )
                for label in labels
            ),
            graphs=graphs,
        )


def provenance_graph_from_record(
    record: ProvenanceGraphRecord,
) -> ExecutionProvenanceGraph:
    return ExecutionProvenanceGraph(
        task_id=record["task_id"],
        nodes=tuple(
            ExecutionProvenanceNode(
                node_id=node["node_id"],
                node_type=ProvenanceNodeType(node["node_type"]),
                text=node["text"],
                metadata=node["metadata"],
            )
            for node in record["nodes"]
        ),
        edges=tuple(
            ExecutionProvenanceEdge(
                source=edge["source"],
                target=edge["target"],
                edge_type=ProvenanceEdgeType(edge["edge_type"]),
                binding=(
                    FieldBinding(**edge["binding"])
                    if edge["binding"] is not None
                    else None
                ),
                weight=edge["weight"],
                metadata=edge["metadata"],
            )
            for edge in record["edges"]
        ),
    )


def _dependency_edge(edge: Sequence[str]) -> tuple[str, str]:
    if len(edge) != 2:
        raise ValueError(
            f"Gold 2Wiki provenance dependency edge must have two IDs, got {len(edge)}."
        )
    return edge[0], edge[1]


__all__ = [
    "TwoWikiProvenanceToEvidenceEvaluationRequest",
    "TwoWikiProvenanceToExecutionProvenanceRankingRequest",
    "TwoWikiProvenanceToTextRankingRequest",
    "provenance_graph_from_record",
]

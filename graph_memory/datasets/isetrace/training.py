from __future__ import annotations

from collections.abc import Sequence

from graph_memory.datasets.isetrace.benchmark_records import (
    ISETraceLabelRecord,
    ISETraceRankingRecord,
)
from graph_memory.evaluation.requests import EvidenceLabel
from graph_memory.graphs.provenance import ProvenanceGraph
from graph_memory.models.graph_retriever.provenance import provenance_training_label
from graph_memory.retrieval.requests import (
    ExecutionProvenanceRankingRequest,
    TextRankingRequest,
)
from graph_memory.trajectories import source_spans_overlap


def adapt_provenance_training_split(
    rankings: Sequence[ISETraceRankingRecord],
    labels: Sequence[ISETraceLabelRecord],
    graphs: Sequence[ProvenanceGraph],
) -> tuple[list[ExecutionProvenanceRankingRequest], list[EvidenceLabel]]:
    """Compile one prepared ISETrace split into provenance model supervision."""

    labels_by_id = {label.task_id: label for label in labels}
    graphs_by_id = {graph.graph_id: graph for graph in graphs}
    if len(labels_by_id) != len(labels):
        raise ValueError("ISETrace training label task IDs must be unique")
    if len(graphs_by_id) != len(graphs):
        raise ValueError("ISETrace training graph IDs must be unique")
    requests: list[ExecutionProvenanceRankingRequest] = []
    compiled_labels: list[EvidenceLabel] = []
    for ranking in rankings:
        try:
            graph = graphs_by_id[ranking.graph_id]
            label_record = labels_by_id[ranking.task_id]
        except KeyError as error:
            raise ValueError(
                "ISETrace training rankings, labels, and graphs must align"
            ) from error
        request = ExecutionProvenanceRankingRequest(
            task_id=ranking.task_id,
            query_text=ranking.query_text,
            candidates=ranking.provenance_candidates,
            graph=graph,
        )
        requests.append(request)
        compiled_labels.append(
            provenance_training_label(
                request,
                gold_spans=label_record.gold_evidence_spans,
            )
        )
    request_ids = {request.task_id for request in requests}
    if request_ids != set(labels_by_id):
        raise ValueError("ISETrace training requests and labels must align")
    return requests, compiled_labels


def adapt_flat_dense_training_split(
    rankings: Sequence[ISETraceRankingRecord],
    labels: Sequence[ISETraceLabelRecord],
) -> tuple[list[TextRankingRequest], list[EvidenceLabel]]:
    """Compile exact ISETrace spans into flat-chunk Dense-FT supervision."""

    labels_by_id = {label.task_id: label for label in labels}
    if len(labels_by_id) != len(labels):
        raise ValueError("ISETrace flat training label task IDs must be unique")

    requests: list[TextRankingRequest] = []
    compiled_labels: list[EvidenceLabel] = []
    for ranking in rankings:
        label = labels_by_id.get(ranking.task_id)
        if label is None or label.graph_id != ranking.graph_id:
            raise ValueError("ISETrace flat training rankings and labels must align")
        positive_ids = tuple(
            candidate.item_id
            for candidate in ranking.flat_candidates
            if any(
                source_spans_overlap(candidate_span, gold_span)
                for candidate_span in candidate.source_spans
                for gold_span in label.gold_evidence_spans
            )
        )
        if not positive_ids:
            raise ValueError(
                f"ISETrace flat task={ranking.task_id!r} has no positive candidates"
            )
        requests.append(
            TextRankingRequest(
                task_id=ranking.task_id,
                query_text=ranking.query_text,
                candidates=ranking.flat_candidates,
            )
        )
        compiled_labels.append(
            EvidenceLabel(
                task_id=ranking.task_id,
                gold_answer="",
                gold_evidence_item_ids=positive_ids,
                gold_dependency_edges=(),
            )
        )

    request_ids = {request.task_id for request in requests}
    if request_ids != set(labels_by_id):
        raise ValueError("ISETrace flat training requests and labels must align")
    return requests, compiled_labels


__all__ = ["adapt_flat_dense_training_split", "adapt_provenance_training_split"]

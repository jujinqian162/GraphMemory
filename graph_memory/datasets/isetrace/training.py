from __future__ import annotations

from collections.abc import Sequence

from graph_memory.datasets.isetrace.benchmark_records import (
    ISETraceLabelRecord,
    ISETraceRankingRecord,
)
from graph_memory.evaluation.requests import EvidenceLabel
from graph_memory.graphs.provenance import ProvenanceGraph
from graph_memory.models.graph_retriever.provenance import provenance_training_label
from graph_memory.query_synthesis.provenance.contracts import (
    TemplateSupervisionRecord,
)
from graph_memory.retrieval.requests import ProvenanceRgcnRequest


def adapt_provenance_training_split(
    rankings: Sequence[ISETraceRankingRecord],
    labels: Sequence[ISETraceLabelRecord],
    graphs: Sequence[ProvenanceGraph],
    template_supervision: Sequence[TemplateSupervisionRecord],
) -> tuple[list[ProvenanceRgcnRequest], list[EvidenceLabel]]:
    """Compile one prepared ISETrace split into provenance model supervision."""

    labels_by_id = {label.task_id: label for label in labels}
    graphs_by_id = {graph.graph_id: graph for graph in graphs}
    templates_by_id = {
        record.task_id: record for record in template_supervision
    }
    if len(labels_by_id) != len(labels):
        raise ValueError("ISETrace training label task IDs must be unique")
    if len(graphs_by_id) != len(graphs):
        raise ValueError("ISETrace training graph IDs must be unique")
    requests: list[ProvenanceRgcnRequest] = []
    compiled_labels: list[EvidenceLabel] = []
    for ranking in rankings:
        try:
            graph = graphs_by_id[ranking.graph_id]
            label_record = labels_by_id[ranking.task_id]
        except KeyError as error:
            raise ValueError(
                "ISETrace training rankings, labels, and graphs must align"
            ) from error
        request = ProvenanceRgcnRequest(
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
                template=templates_by_id.get(ranking.task_id),
            )
        )
    request_ids = {request.task_id for request in requests}
    if request_ids != set(labels_by_id):
        raise ValueError("ISETrace training requests and labels must align")
    if set(templates_by_id) - request_ids:
        raise ValueError("ISETrace template supervision has unknown task IDs")
    return requests, compiled_labels


__all__ = ["adapt_provenance_training_split"]

from __future__ import annotations

import pytest

from graph_memory.models.graph_retriever.checkpoint import (
    load_rgcn_checkpoint,
    save_rgcn_checkpoint,
)
from graph_memory.models.graph_retriever.config.records import RgcnTrainingConfig
from graph_memory.models.graph_retriever.factory import build_model_from_config
from graph_memory.registry.retrieval_builders import build_retrieval
from graph_memory.registry.retrieval import (
    ProvenanceRgcnBuildPayload,
    ProvenanceRgcnRetrievalSettings,
)
from graph_memory.models.graph_retriever.provenance import (
    provenance_train_pair_task,
    provenance_training_label,
    tensorize_provenance_training_task,
)
from graph_memory.models.graph_retriever.provenance_training import (
    train_provenance_graph_retriever,
)
from graph_memory.query_synthesis.provenance.contracts import (
    TemplateSupervisionRecord,
)
from graph_memory.retrieval.execution.service import run_retrieval
from graph_memory.retrieval.methods.ids import RetrievalMethodId
from graph_memory.retrieval.requests import TextRankingRequest
from graph_memory.training_pairs import build_train_pairs
from graph_memory.training_pairs.config import NegativeSamplingConfig
from graph_memory.trajectories import SourceSpan
from tests.test_provenance_rgcn_tensorization import (
    DeterministicEmbeddingProvider,
    _graph_and_request,
    _model_config,
)


def _template_supervision():
    graph, request = _graph_and_request(
        task_id="template-task", query_text="Which result completed the chain?"
    )
    template = TemplateSupervisionRecord(
        task_id=request.task_id,
        graph_id=graph.graph_id,
        query_text=request.query_text,
        focus_output_ids=("output:c4",),
        participant_output_ids=("output:c2", "output:c3", "output:c4"),
        positive_candidate_ids=("output-content:c4:chunk:0",),
        motif_id="motif:multi_hop_flow:test",
        motif_type="multi_hop_flow",
        query_intent="downstream_result",
        graph_fingerprint=graph.fingerprint(),
    )
    return request, template


def test_natural_exact_span_maps_only_overlapping_provenance_candidate() -> None:
    _graph, request = _graph_and_request(
        task_id="natural-task", query_text="What was verified?"
    )
    candidate = next(
        candidate
        for candidate in request.candidates
        if candidate.item_id == "output-content:c4:chunk:0"
    )
    source_span = candidate.source_spans[0]
    assert source_span.char_start is not None
    label = provenance_training_label(
        request,
        gold_spans=(
            source_span.model_copy(update={"char_start": source_span.char_start + 1}),
        ),
    )

    assert label.gold_evidence_item_ids == ("output-content:c4:chunk:0",)
    assert label.query_intent is None
    assert label.motif_type is None


def test_provenance_label_compilation_fails_fast_without_a_positive() -> None:
    _graph, request = _graph_and_request(
        task_id="no-positive-task", query_text="Find missing evidence."
    )

    with pytest.raises(ValueError, match="has no positive candidates"):
        provenance_training_label(
            request,
            gold_spans=(
                SourceSpan(
                    event_id="missing-event",
                    json_pointer="/content",
                    char_start=0,
                    char_end=1,
                ),
            ),
        )


def test_template_focus_drives_positives_and_participants_remain_negatives() -> None:
    request, template = _template_supervision()
    label = provenance_training_label(request, gold_spans=(), template=template)
    result = build_train_pairs(
        [provenance_train_pair_task(request, label)],
        NegativeSamplingConfig(
            random_seed=17,
            easy_random_per_positive=0,
            hard_bm25_per_positive=0,
            hard_dense_per_positive=0,
            hard_graph_neighbor_per_positive=10,
            hard_pool_size=20,
        ),
    )

    positives = {pair.node_id for pair in result.pairs if pair.label == 1}
    graph_negatives = {
        pair.node_id
        for pair in result.pairs
        if pair.sample_type == "hard_graph_neighbor"
    }
    assert positives == {"output-content:c4:chunk:0"}
    assert {
        "output-content:c2:chunk:0",
        "output-content:c3:chunk:0",
    }.issubset(graph_negatives)
    assert graph_negatives <= {
        candidate.item_id for candidate in request.candidates
    }

    task = tensorize_provenance_training_task(
        request,
        result.pairs,
        model_config=_model_config(),
        text_embedding_provider=DeterministicEmbeddingProvider(),
    )
    assert task.sample_node_ids == [pair.node_id for pair in result.pairs]
    assert task.labels.tolist() == [float(pair.label) for pair in result.pairs]


def test_tiny_mixed_provenance_training_uses_natural_dev_selection(
    tmp_path,
) -> None:
    graph, base_request = _graph_and_request(
        task_id="base-task", query_text="Which result completed the chain?"
    )
    requests = [
        base_request.model_copy(
            update={"task_id": task_id, "query_text": query_text}
        )
        for task_id, query_text in (
            ("train-natural", "What was verified?"),
            ("train-template", "Which result completed the chain?"),
            ("dev-natural", "What result was verified?"),
            ("dev-template", "Find the completed chain result."),
        )
    ]
    candidate = next(
        candidate
        for candidate in base_request.candidates
        if candidate.item_id == "output-content:c4:chunk:0"
    )

    def label_for(request, *, template: bool):
        supervision = (
            TemplateSupervisionRecord(
                task_id=request.task_id,
                graph_id=graph.graph_id,
                query_text=request.query_text,
                focus_output_ids=("output:c4",),
                participant_output_ids=("output:c2", "output:c3", "output:c4"),
                positive_candidate_ids=("output-content:c4:chunk:0",),
                motif_id="motif:multi_hop_flow:mixed",
                motif_type="multi_hop_flow",
                query_intent="downstream_result",
                graph_fingerprint=graph.fingerprint(),
            )
            if template
            else None
        )
        return provenance_training_label(
            request,
            gold_spans=() if template else candidate.source_spans,
            template=supervision,
        )

    train_labels = [
        label_for(requests[0], template=False),
        label_for(requests[1], template=True),
    ]
    pair_result = build_train_pairs(
        [
            provenance_train_pair_task(request, label)
            for request, label in zip(requests[:2], train_labels, strict=True)
        ],
        NegativeSamplingConfig(
            random_seed=19,
            easy_random_per_positive=1,
            hard_bm25_per_positive=0,
            hard_dense_per_positive=0,
            hard_graph_neighbor_per_positive=1,
            hard_pool_size=10,
        ),
    )
    dev_labels = [
        label_for(requests[2], template=False),
        label_for(requests[3], template=True),
    ]
    result = train_provenance_graph_retriever(
        train_requests=requests[:2],
        train_labels=train_labels,
        train_pairs=pair_result.pairs,
        dev_requests=requests[2:],
        dev_labels=dev_labels,
        dev_query_origins={"dev-natural": "natural", "dev-template": "template"},
        model_config=_model_config(),
        training_config=RgcnTrainingConfig(
            learning_rate=1e-3,
            per_device_graph_batch_size=2,
            epochs=1,
            random_seed=19,
        ),
        text_embedding_provider=DeterministicEmbeddingProvider(),
        device="cpu",
    )

    assert result.best_epoch == 1
    assert result.global_step == 1
    metrics = result.metric_records[0]
    assert metrics["dev_query_selection_origin"] == "natural"
    assert "dev_natural_recall_at_5" in metrics
    assert "dev_template_recall_at_5" in metrics
    assert metrics["selection_metric_value"] == metrics["dev_natural_recall_at_5"]

    template_only = train_provenance_graph_retriever(
        train_requests=[requests[1]],
        train_labels=[train_labels[1]],
        train_pairs=[
            pair for pair in pair_result.pairs if pair.task_id == "train-template"
        ],
        dev_requests=[requests[3]],
        dev_labels=[dev_labels[1]],
        dev_query_origins={"dev-template": "template"},
        model_config=_model_config(),
        training_config=RgcnTrainingConfig(
            learning_rate=1e-3,
            per_device_graph_batch_size=1,
            epochs=1,
            random_seed=19,
        ),
        text_embedding_provider=DeterministicEmbeddingProvider(),
        device="cpu",
    )
    template_metrics = template_only.metric_records[0]
    assert template_metrics["dev_query_selection_origin"] == "template"
    assert "dev_natural_recall_at_5" not in template_metrics
    assert template_metrics["selection_metric_value"] == template_metrics[
        "dev_template_recall_at_5"
    ]

    checkpoint_path = tmp_path / "mixed-provenance-rgcn.pt"
    selected_model = build_model_from_config(result.model_config)
    selected_model.load_state_dict(result.best_model_state_dict)
    save_rgcn_checkpoint(
        checkpoint_path,
        method_name=RetrievalMethodId.PROVENANCE_RGCN,
        model=selected_model,
        optimizer_state_dict=result.optimizer_state_dict,
        scheduler_state_dict=result.scheduler_state_dict,
        epoch=result.best_epoch,
        global_step=result.global_step,
        best_dev_metric=result.best_dev_metric,
        model_config=result.model_config,
        training_config=result.training_config,
    )
    dev_request = requests[2]
    text_request = TextRankingRequest(
        task_id=dev_request.task_id,
        query_text=dev_request.query_text,
        candidates=dev_request.candidates,
    )
    settings = ProvenanceRgcnRetrievalSettings(
        top_k=3,
        checkpoint=checkpoint_path,
        device="cpu",
    )
    payload = ProvenanceRgcnBuildPayload(
        text_requests=[text_request],
        provenance_graphs=[graph],
        graph_ids_by_task_id={dev_request.task_id: graph.graph_id},
        text_embedding_provider=DeterministicEmbeddingProvider(),
    )
    first = build_retrieval(settings, payload)
    second = build_retrieval(settings, payload)
    first_results = run_retrieval(
        retrieval_method=first.method,
        requests=first.execution_requests,
        top_k=3,
    )
    second_results = run_retrieval(
        retrieval_method=second.method,
        requests=second.execution_requests,
        top_k=3,
    )

    assert first_results[0].ranked_nodes == second_results[0].ranked_nodes
    assert (
        first_results[0].retrieved_subgraph
        == second_results[0].retrieved_subgraph
    )
    assert first_results[0].method is RetrievalMethodId.PROVENANCE_RGCN
    candidate_spans = {
        candidate.item_id: candidate.source_spans for candidate in dev_request.candidates
    }
    assert all(
        ranked.source_spans == candidate_spans[ranked.node_id]
        for ranked in first_results[0].ranked_nodes
    )


def test_provenance_checkpoint_round_trip_requires_matching_method(tmp_path) -> None:
    model_config = _model_config()
    training_config = RgcnTrainingConfig(epochs=2)
    model = build_model_from_config(model_config)
    checkpoint_path = tmp_path / "provenance-rgcn.pt"

    payload = save_rgcn_checkpoint(
        checkpoint_path,
        method_name=RetrievalMethodId.PROVENANCE_RGCN,
        model=model,
        optimizer_state_dict={},
        scheduler_state_dict={},
        epoch=1,
        global_step=4,
        best_dev_metric=0.75,
        model_config=model_config,
        training_config=training_config,
    )
    loaded = load_rgcn_checkpoint(
        checkpoint_path,
        expected_method=RetrievalMethodId.PROVENANCE_RGCN,
        map_location="cpu",
    )

    assert payload["method_name"] == RetrievalMethodId.PROVENANCE_RGCN
    assert payload["epoch"] == 1
    assert loaded.model_config == model_config
    assert loaded.training_config == training_config
    with pytest.raises(ValueError, match="must match"):
        save_rgcn_checkpoint(
            tmp_path / "invalid-method-config.pt",
            method_name=RetrievalMethodId.DENSE_RGCN_GRAPH_RETRIEVER,
            model=model,
            optimizer_state_dict={},
            scheduler_state_dict={},
            epoch=1,
            global_step=4,
            best_dev_metric=0.75,
            model_config=model_config,
            training_config=training_config,
        )
    with pytest.raises(ValueError, match="does not match"):
        load_rgcn_checkpoint(
            checkpoint_path,
            expected_method=RetrievalMethodId.DENSE_RGCN_GRAPH_RETRIEVER,
            map_location="cpu",
        )

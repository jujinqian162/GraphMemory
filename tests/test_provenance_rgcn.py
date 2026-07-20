from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from graph_memory.datasets.twowiki_provenance import (
    TwoWikiProvenanceToEvidenceEvaluationRequest,
    TwoWikiProvenanceToExecutionProvenanceRankingRequest,
    convert_twowiki_source_records,
)
from graph_memory.models.provenance_rgcn import (
    ExecutionProvenanceRGCN,
    ExecutionProvenanceRgcnRetriever,
    ProvenanceRgcnModelConfig,
    ProvenanceDevMetrics,
    ProvenanceRgcnTrainingConfig,
    compute_provenance_loss,
    default_provenance_rgcn_model_config,
    load_provenance_rgcn_checkpoint,
    save_provenance_rgcn_checkpoint,
    tensorize_provenance_request,
    train_provenance_rgcn,
)
from graph_memory.models.graph_retriever.internals.neural import (
    SharedRelationTransform,
    TypedRelationTransform,
)
from graph_memory.models.provenance_rgcn.contracts import ProvenanceModelOutput
from graph_memory.models.provenance_rgcn.inference import _apply_structured_reranking
from graph_memory.contracts.training_pairs import TrainPairRecord
from graph_memory.registry import Registry
from graph_memory.registry.retrieval import (
    ProvenanceRgcnBuildPayload,
    ProvenanceRgcnRetrievalSettings,
    RetrievalMethodId,
)
from graph_memory.retrieval.contracts import ExecutionProvenanceTrace
from graph_memory.retrieval.contracts import RankedNode
from graph_memory.retrieval.requests import TextRankingRequest
from graph_memory.training_pairs import build_provenance_train_pairs, build_train_pairs
from graph_memory.training_pairs.config import (
    NegativeSamplingConfig,
    ProvenanceNegativeSamplingConfig,
)
from graph_memory.training_pairs.requests import (
    ProvenanceTrainPairBuildTask,
    TrainPairBuildTask,
)


class TinyEncoder:
    vocabulary = (
        "alpha",
        "bridge",
        "country",
        "wrong",
        "retrieve",
        "evidence",
        "city",
        "artist",
    )

    def encode(
        self,
        texts: Sequence[str],
        batch_size: int = 64,
        normalize_embeddings: bool = True,
    ) -> object:
        _ = batch_size
        rows = []
        for text in texts:
            lowered = text.casefold()
            vector = np.asarray(
                [float(lowered.count(token)) for token in self.vocabulary],
                dtype=np.float32,
            )
            if normalize_embeddings:
                norm = float(np.linalg.norm(vector))
                if norm:
                    vector = vector / norm
            rows.append(vector)
        return np.asarray(rows, dtype=np.float32)


class OrderedRanker:
    def __init__(self, *, reverse: bool = False) -> None:
        self.reverse = reverse

    @property
    def method_name(self) -> str:
        return "ordered_test_ranker"

    def rank(self, request: TextRankingRequest) -> list[RankedNode]:
        candidate_ids = [candidate.item_id for candidate in request.candidates]
        if self.reverse:
            candidate_ids.reverse()
        return [
            RankedNode(node_id, float(len(candidate_ids) - index))
            for index, node_id in enumerate(candidate_ids)
        ]


def test_tensorizer_and_model_preserve_full_typed_graph() -> None:
    request, _label = _request_and_label()
    config = _model_config()

    tensor = tensorize_provenance_request(request, encoder=TinyEncoder(), config=config)
    model = ExecutionProvenanceRGCN(config)
    output = model(tensor)

    assert tensor.graph_batch.node_embeddings.shape[0] == len(request.graph.nodes)
    assert tensor.graph_batch.edge_index.shape[1] == len(request.graph.edges) * 2
    assert tensor.candidate_ids == tuple(
        candidate.item_id for candidate in request.candidates
    )
    assert tensor.logical_transitions
    assert "feeds:evidence:context:semantic_reference_forward" in (
        config.relation_vocab[index]
        for index in tensor.graph_batch.relation_ids.tolist()
    )
    assert output.candidate_logits.shape == (len(request.candidates),)
    assert output.edge_logits.shape == (len(tensor.logical_transitions),)


def test_provenance_model_ablations_remove_named_graph_signals() -> None:
    request, _label = _request_and_label()
    full = _default_model_config("full_rgcn")
    wo_graph = _default_model_config("wo_graph")
    wo_edge_type = _default_model_config("wo_edge_type")
    wo_edge_weight = _default_model_config("wo_edge_weight")

    full_model = ExecutionProvenanceRGCN(full)
    edge_type_model = ExecutionProvenanceRGCN(wo_edge_type)
    assert isinstance(
        full_model.graph_encoder.layers[0].message_transform,
        TypedRelationTransform,
    )
    assert isinstance(
        edge_type_model.graph_encoder.layers[0].message_transform,
        SharedRelationTransform,
    )
    assert wo_graph.num_layers == 0
    assert len(ExecutionProvenanceRGCN(wo_graph).graph_encoder.layers) == 0

    full_tensor = tensorize_provenance_request(
        request, encoder=TinyEncoder(), config=full
    )
    uniform_tensor = tensorize_provenance_request(
        request, encoder=TinyEncoder(), config=wo_edge_weight
    )
    assert torch.equal(
        uniform_tensor.graph_batch.edge_index, full_tensor.graph_batch.edge_index
    )
    assert torch.equal(
        uniform_tensor.graph_batch.relation_ids, full_tensor.graph_batch.relation_ids
    )
    feed_weights_by_source: dict[str, list[float]] = {}
    for edge in request.graph.edges:
        if edge.edge_type.value == "feeds":
            feed_weights_by_source.setdefault(edge.source, []).append(edge.weight)
    source_means = {
        source: sum(weights) / len(weights)
        for source, weights in feed_weights_by_source.items()
    }
    expected_message_weights: list[float] = []
    for edge in request.graph.edges:
        resolved = (
            source_means[edge.source]
            if edge.edge_type.value == "feeds"
            else edge.weight
        )
        expected_message_weights.extend((resolved, resolved))
    assert torch.allclose(
        uniform_tensor.graph_batch.edge_weights,
        torch.tensor(expected_message_weights, dtype=torch.float32),
    )
    assert all(value == pytest.approx(0.75) for value in source_means.values())
    assert any(edge.weight != 1.0 for edge in request.graph.edges)


def test_shuffled_feed_diagnostic_changes_only_feed_message_topology() -> None:
    request, _label = _request_and_label()
    native_config = _default_model_config("full_rgcn")
    shuffled_config = _default_model_config("diagnostic_shuffled_feed")

    native = tensorize_provenance_request(
        request, encoder=TinyEncoder(), config=native_config
    )
    shuffled = tensorize_provenance_request(
        request, encoder=TinyEncoder(), config=shuffled_config
    )
    repeated = tensorize_provenance_request(
        request, encoder=TinyEncoder(), config=shuffled_config
    )

    assert torch.equal(shuffled.graph_batch.edge_index, repeated.graph_batch.edge_index)
    assert torch.equal(
        native.graph_batch.relation_ids, shuffled.graph_batch.relation_ids
    )
    assert torch.equal(
        native.graph_batch.edge_weights, shuffled.graph_batch.edge_weights
    )
    feed_columns: list[int] = []
    non_feed_columns: list[int] = []
    for edge_index, edge in enumerate(request.graph.edges):
        columns = [2 * edge_index, 2 * edge_index + 1]
        if edge.edge_type.value == "feeds":
            feed_columns.extend(columns)
        else:
            non_feed_columns.extend(columns)
    assert torch.equal(
        native.graph_batch.edge_index[:, non_feed_columns],
        shuffled.graph_batch.edge_index[:, non_feed_columns],
    )
    assert not torch.equal(
        native.graph_batch.edge_index[:, feed_columns],
        shuffled.graph_batch.edge_index[:, feed_columns],
    )


def test_training_loss_consumes_only_materialized_candidate_pairs() -> None:
    request, label = _request_and_label()
    config = _model_config()
    training = ProvenanceRgcnTrainingConfig(edge_loss_weight=0.0)
    tensor = tensorize_provenance_request(request, encoder=TinyEncoder(), config=config)
    model = ExecutionProvenanceRGCN(config)
    output = model(tensor)
    gold_id = label.gold_evidence_item_ids[0]
    negative_id = next(
        candidate_id
        for candidate_id in tensor.candidate_ids
        if candidate_id not in set(label.gold_evidence_item_ids)
    )
    pairs: list[TrainPairRecord] = [
        {
            "task_id": request.task_id,
            "node_id": gold_id,
            "label": 1,
            "sample_type": "positive",
        },
        {
            "task_id": request.task_id,
            "node_id": negative_id,
            "label": 0,
            "sample_type": "hard_bm25",
        },
    ]
    first = compute_provenance_loss(
        tensor, output, label, pairs, config, training
    )
    unpaired_index = next(
        index
        for index, candidate_id in enumerate(tensor.candidate_ids)
        if candidate_id not in {gold_id, negative_id}
    )
    changed_logits = output.candidate_logits.clone()
    changed_logits[unpaired_index] += 100.0
    second = compute_provenance_loss(
        tensor,
        ProvenanceModelOutput(
            node_states=output.node_states,
            candidate_logits=changed_logits,
            edge_logits=output.edge_logits,
        ),
        label,
        pairs,
        config,
        training,
    )

    assert torch.equal(first.candidate, second.candidate)
    assert first.total.item() > 0.0


def test_candidate_loss_is_pairwise_over_every_positive_negative_comparison() -> None:
    request, label = _request_and_label()
    tensor = tensorize_provenance_request(
        request, encoder=TinyEncoder(), config=_model_config()
    )
    logits = torch.tensor(
        [2.0, 1.0, -1.0, 0.5, -0.5, 0.0], dtype=torch.float32
    )
    output = ProvenanceModelOutput(
        node_states=torch.empty((len(request.graph.nodes), 1)),
        candidate_logits=logits,
        edge_logits=torch.zeros(len(tensor.logical_transitions)),
    )
    gold_ids = set(label.gold_evidence_item_ids)
    pairs = _train_pairs(request, label)
    loss = compute_provenance_loss(
        tensor,
        output,
        label,
        pairs,
        _model_config(),
        ProvenanceRgcnTrainingConfig(edge_loss_weight=0.0),
    )
    positive_indices = [
        index
        for index, candidate_id in enumerate(tensor.candidate_ids)
        if candidate_id in gold_ids
    ]
    negative_indices = [
        index
        for index, candidate_id in enumerate(tensor.candidate_ids)
        if candidate_id not in gold_ids
    ]
    expected = torch.nn.functional.softplus(
        -(logits[positive_indices].unsqueeze(1) - logits[negative_indices].unsqueeze(0))
    ).mean()

    assert loss.candidate == pytest.approx(expected)
    assert loss.comparison_count == len(positive_indices) * len(negative_indices)


def test_structured_reranking_is_bounded_stable_and_can_be_disabled() -> None:
    request, _label = _request_and_label()
    config = replace(
        _model_config(),
        structured_pool_size=6,
        structured_seed_top_s=5,
        preserve_node_top_n=2,
        edge_accept_threshold=0.5,
    )
    tensor = tensorize_provenance_request(request, encoder=TinyEncoder(), config=config)
    transition = tensor.logical_transitions[0]
    remaining = [
        candidate_id
        for candidate_id in tensor.candidate_ids
        if candidate_id not in {transition.source_id, transition.target_id}
    ]
    raw_ids = [
        *remaining[:2],
        transition.source_id,
        *remaining[2:],
        transition.target_id,
    ]
    raw = [
        RankedNode(node_id, float(len(raw_ids) - index))
        for index, node_id in enumerate(raw_ids)
    ]
    logits = torch.full((len(tensor.logical_transitions),), -10.0)
    logits[0] = 10.0

    structured = _apply_structured_reranking(
        raw,
        tensor.logical_transitions,
        logits,
        config=config,
        promotion_enabled=True,
    )
    final_ids = [item.node_id for item in structured.ranked_nodes]
    assert final_ids[:2] == raw_ids[:2]
    assert final_ids.index(transition.target_id) == final_ids.index(transition.source_id) + 1
    assert sorted(item.score for item in structured.ranked_nodes) == sorted(
        item.score for item in raw
    )
    assert any(trace.decision == "promoted" for trace in structured.traces)

    disabled = _apply_structured_reranking(
        raw,
        tensor.logical_transitions,
        logits,
        config=config,
        promotion_enabled=False,
    )
    assert [item.node_id for item in disabled.ranked_nodes] == raw_ids
    assert any(trace.decision == "promotion_disabled" for trace in disabled.traces)

    below_threshold = _apply_structured_reranking(
        raw,
        tensor.logical_transitions,
        torch.full((len(tensor.logical_transitions),), -10.0),
        config=config,
        promotion_enabled=True,
    )
    assert below_threshold.selected_transitions == ()
    assert below_threshold.abstained_source_ids


def test_structured_reranking_records_above_source_noop_and_target_conflict() -> None:
    request, _label = _request_and_label()
    config = replace(
        _model_config(),
        structured_pool_size=6,
        structured_seed_top_s=5,
        preserve_node_top_n=2,
    )
    tensor = tensorize_provenance_request(request, encoder=TinyEncoder(), config=config)
    by_target: dict[str, list[int]] = {}
    for index, transition in enumerate(tensor.logical_transitions):
        by_target.setdefault(transition.target_id, []).append(index)
    conflict_indices = next(indices for indices in by_target.values() if len(indices) >= 2)
    first = tensor.logical_transitions[conflict_indices[0]]
    second = tensor.logical_transitions[conflict_indices[1]]
    remaining = [
        candidate_id
        for candidate_id in tensor.candidate_ids
        if candidate_id not in {first.target_id, first.source_id, second.source_id}
    ]
    raw_ids = [first.target_id, first.source_id, second.source_id, *remaining]
    raw = [
        RankedNode(node_id, float(len(raw_ids) - index))
        for index, node_id in enumerate(raw_ids)
    ]
    logits = torch.full((len(tensor.logical_transitions),), -10.0)
    logits[conflict_indices[0]] = 9.0
    logits[conflict_indices[1]] = 8.0

    result = _apply_structured_reranking(
        raw,
        tensor.logical_transitions,
        logits,
        config=config,
        promotion_enabled=True,
    )

    assert [item.node_id for item in result.ranked_nodes] == raw_ids
    assert any(trace.decision == "already_above_source" for trace in result.traces)
    assert any(trace.decision == "target_conflict" for trace in result.traces)
    assert sum(
        transition.target_id == first.target_id
        for transition, _score in result.selected_transitions
    ) == 1


def test_provenance_pair_build_uses_real_bm25_and_dense_samplers() -> None:
    request, label = _request_and_label()
    text_request = TextRankingRequest(
        request.task_id, request.query_text, request.candidates
    )

    result = build_train_pairs(
        [TrainPairBuildTask(text_request=text_request, label=label)],
        NegativeSamplingConfig(
            easy_random_per_positive=0,
            hard_bm25_per_positive=1,
            hard_dense_per_positive=1,
            hard_graph_neighbor_per_positive=0,
            hard_pool_size=10,
        ),
        bm25_retriever=OrderedRanker(),
        dense_retriever=OrderedRanker(reverse=True),
    )

    assert result.summary["negative_count_by_type"] == {
        "hard_bm25": 2,
        "hard_dense": 2,
    }
    assert {pair["sample_type"] for pair in result.pairs} == {
        "positive",
        "hard_bm25",
        "hard_dense",
    }


def test_provenance_native_pairs_are_unique_and_use_hardness_precedence() -> None:
    request, label = _request_and_label()
    text_request = TextRankingRequest(
        request.task_id, request.query_text, request.candidates
    )

    result = build_provenance_train_pairs(
        [
            ProvenanceTrainPairBuildTask(
                text_request=text_request,
                graph=request.graph,
                label=label,
            )
        ],
        ProvenanceNegativeSamplingConfig(
            random_seed=13,
            easy_random_per_positive=2,
            hard_bm25_per_positive=1,
            hard_dense_per_positive=1,
            hard_graph_neighbor_per_positive=0,
            hard_provenance_successor_per_positive=2,
            hard_provenance_predecessor_per_positive=1,
            hard_pool_size=10,
        ),
        bm25_retriever=OrderedRanker(),
        dense_retriever=OrderedRanker(reverse=True),
    )

    negatives = [pair for pair in result.pairs if pair["label"] == 0]
    assert len({pair["node_id"] for pair in negatives}) == len(negatives)
    assert "hard_provenance_successor" in {
        pair["sample_type"] for pair in negatives
    }
    assert "hard_provenance_predecessor" in {
        pair["sample_type"] for pair in negatives
    }
    requested_counts = result.summary.get("requested_negative_count_by_type")
    assert requested_counts == {
        "easy_random": 4,
        "hard_bm25": 2,
        "hard_dense": 2,
        "hard_provenance_predecessor": 2,
        "hard_provenance_successor": 4,
    }
    source_overlap = result.summary.get("source_overlap_by_task")
    assert source_overlap
    precedence = {
        "hard_provenance_successor": 0,
        "hard_provenance_predecessor": 1,
        "hard_dense": 2,
        "hard_bm25": 3,
        "easy_random": 4,
    }
    for sources in source_overlap[request.task_id].values():
        assert sources == sorted(sources, key=precedence.__getitem__)
    assert sum(result.summary["negative_count_by_type"].values()) == len(negatives)


def test_provenance_wo_hard_negatives_retains_only_easy_random() -> None:
    request, label = _request_and_label()
    result = build_provenance_train_pairs(
        [
            ProvenanceTrainPairBuildTask(
                text_request=TextRankingRequest(
                    request.task_id, request.query_text, request.candidates
                ),
                graph=request.graph,
                label=label,
            )
        ],
        ProvenanceNegativeSamplingConfig(
            easy_random_per_positive=2,
            hard_bm25_per_positive=0,
            hard_dense_per_positive=0,
            hard_graph_neighbor_per_positive=0,
            hard_provenance_successor_per_positive=0,
            hard_provenance_predecessor_per_positive=0,
        ),
    )

    assert {
        pair["sample_type"] for pair in result.pairs if pair["label"] == 0
    } == {"easy_random"}


def test_checkpoint_family_round_trip_and_rejection(
    tmp_path: Path,
) -> None:
    config = _model_config()
    training = ProvenanceRgcnTrainingConfig()
    model = ExecutionProvenanceRGCN(config)
    checkpoint = tmp_path / "best.pt"
    payload = save_provenance_rgcn_checkpoint(
        checkpoint,
        method_name="execution_provenance_rgcn_retriever",
        model=model,
        model_config=config,
        training_config=training,
    )

    loaded = load_provenance_rgcn_checkpoint(checkpoint)

    assert loaded.model_config == config
    assert loaded.training_config == training
    assert "beam_width" not in loaded.payload["model_config"]
    assert "max_steps" not in loaded.payload["model_config"]
    assert loaded.model_config.ablation_name == "full_rgcn"
    assert loaded.model_config.message_transform_type == "typed"
    assert loaded.model_config.edge_weight_policy == "artifact"

    legacy_payload = deepcopy(payload)
    for field_name in (
        "ablation_name",
        "message_transform_type",
        "edge_weight_policy",
    ):
        legacy_payload["model_config"].pop(field_name)
    legacy_checkpoint = tmp_path / "legacy-full.pt"
    torch.save(legacy_payload, legacy_checkpoint)
    with pytest.raises(ValueError, match="incomplete for schema v3"):
        load_provenance_rgcn_checkpoint(legacy_checkpoint)
    assert "path_loss_weight" not in loaded.payload["training_config"]
    evidence_checkpoint = tmp_path / "evidence.pt"
    torch.save(
        {"schema_version": 2, "method_name": "dense_rgcn_graph_retriever"},
        evidence_checkpoint,
    )
    with pytest.raises(ValueError, match="family mismatch"):
        load_provenance_rgcn_checkpoint(evidence_checkpoint)

    provenance_v2 = deepcopy(payload)
    provenance_v2["schema_version"] = 2
    provenance_v2_checkpoint = tmp_path / "provenance-v2.pt"
    torch.save(provenance_v2, provenance_v2_checkpoint)
    with pytest.raises(ValueError, match="schema version"):
        load_provenance_rgcn_checkpoint(provenance_v2_checkpoint)


def test_joint_dev_selection_uses_declared_components_and_earlier_ties() -> None:
    metrics = ProvenanceDevMetrics(
        full_support_at_5=0.8,
        mrr=0.6,
        edge_precision_at_10=0.5,
        edge_recall_at_10=0.5,
        edge_f1_at_10=0.4,
        average_emitted_edges=1.0,
        abstention_rate=0.2,
    )
    same_joint_higher_full_support = replace(
        metrics,
        full_support_at_5=0.85,
        mrr=0.5,
        edge_f1_at_10=0.4,
    )

    assert metrics.joint == pytest.approx(0.65)
    assert same_joint_higher_full_support.joint == pytest.approx(metrics.joint)
    assert same_joint_higher_full_support.selection_key(2) > metrics.selection_key(1)
    assert metrics.selection_key(1) > metrics.selection_key(2)


def test_minimal_train_and_inference_return_logical_paths() -> None:
    request, label = _request_and_label()
    config = _model_config()
    result = train_provenance_rgcn(
        train_requests=[request],
        train_labels=[label],
        train_pairs=_train_pairs(request, label),
        model_config=config,
        training_config=ProvenanceRgcnTrainingConfig(epochs=1),
        encoder=TinyEncoder(),
    )
    retriever = ExecutionProvenanceRgcnRetriever(result.model, TinyEncoder(), config)

    prediction = retriever.rank_task(request, top_k=len(request.candidates))

    assert len(prediction.ranked_nodes) == len(request.candidates)
    assert isinstance(prediction.trace.native_trace, ExecutionProvenanceTrace)
    assert prediction.trace.native_trace.paths
    assert prediction.trace.native_trace.edges
    assert prediction.trace.retrieved_edges
    assert result.metric_records[0]["epoch"] == 1


def test_registry_builds_provenance_rgcn_from_its_checkpoint_family(
    tmp_path: Path,
) -> None:
    request, _label = _request_and_label()
    config = _model_config()
    checkpoint = tmp_path / "best.pt"
    save_provenance_rgcn_checkpoint(
        checkpoint,
        method_name="execution_provenance_rgcn_retriever",
        model=ExecutionProvenanceRGCN(config),
        model_config=config,
        training_config=ProvenanceRgcnTrainingConfig(),
    )

    built = Registry.retrieval.build(
        ProvenanceRgcnRetrievalSettings(
            top_k=5,
            checkpoint=checkpoint,
            device="cpu",
        ),
        ProvenanceRgcnBuildPayload(
            provenance_requests=[request],
            dense_encoder=TinyEncoder(),
        ),
    )
    prediction = built.method.rank_task(request, top_k=5)

    assert built.provenance.method is (
        RetrievalMethodId.EXECUTION_PROVENANCE_RGCN_RETRIEVER
    )
    assert built.execution_tasks[0].method_request is request
    assert len(prediction.ranked_nodes) == len(request.candidates)


def _request_and_label():
    raw = convert_twowiki_source_records(
        [_source_example()], candidate_cap=6, seed=13, strict=True
    ).records[0]
    request = TwoWikiProvenanceToExecutionProvenanceRankingRequest().project(
        raw["ranking"]
    )
    label = (
        TwoWikiProvenanceToEvidenceEvaluationRequest()
        .project(predictions=[], labels=[raw["label"]], graphs=[])
        .labels[0]
    )
    return request, label


def _model_config() -> ProvenanceRgcnModelConfig:
    return ProvenanceRgcnModelConfig(
        encoder_model="tiny",
        encoder_dim=len(TinyEncoder.vocabulary),
        query_prefix="",
        passage_prefix="",
        encoder_batch_size=8,
        hidden_dim=16,
        node_type_dim=4,
        num_layers=1,
        dropout=0.0,
    )


def _default_model_config(ablation_name: str) -> ProvenanceRgcnModelConfig:
    return default_provenance_rgcn_model_config(
        encoder_model="tiny",
        encoder_dim=len(TinyEncoder.vocabulary),
        query_prefix="",
        passage_prefix="",
        encoder_batch_size=8,
        hidden_dim=16,
        node_type_dim=4,
        num_layers=1,
        dropout=0.0,
        ablation_name=ablation_name,
    )


def _train_pairs(request, label) -> list[TrainPairRecord]:
    gold_ids = set(label.gold_evidence_item_ids)
    return [
        {
            "task_id": request.task_id,
            "node_id": candidate.item_id,
            "label": 1 if candidate.item_id in gold_ids else 0,
            "sample_type": (
                "positive" if candidate.item_id in gold_ids else "hard_dense"
            ),
        }
        for candidate in request.candidates
    ]


def _source_example() -> dict[str, object]:
    return {
        "_id": "rgcn",
        "type": "compositional",
        "question": "In which country is the birthplace of Alpha located?",
        "context": [
            ["Alpha", ["Alpha was born in Bridge City.", "Alpha is an artist."]],
            [
                "Bridge City",
                ["Bridge City is located in Country Z.", "It has a river."],
            ],
            ["Wrong One", ["Wrong One is located elsewhere."]],
            ["Wrong Two", ["Wrong Two mentions Alpha but not the answer."]],
        ],
        "supporting_facts": [["Alpha", 0], ["Bridge City", 0]],
        "evidences": [
            ["Alpha", "birth place", "Bridge City"],
            ["Bridge City", "country", "Country Z"],
        ],
        "evidences_id": [],
        "answer_id": "Country Z",
        "answer": "Country Z",
    }

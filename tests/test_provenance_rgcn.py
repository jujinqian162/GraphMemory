from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import cast

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
    collate_provenance_tasks,
    collate_provenance_training_tasks,
    compute_provenance_batch_loss,
    compute_provenance_loss,
    default_provenance_rgcn_model_config,
    load_provenance_rgcn_checkpoint,
    save_provenance_rgcn_checkpoint,
    split_provenance_output,
    tensorize_provenance_request,
    tensorize_provenance_task,
    materialize_provenance_training_task,
    train_provenance_rgcn,
)
from graph_memory.models.graph_retriever.internals.neural import (
    SharedRelationTransform,
    TypedRelationTransform,
)
from graph_memory.models.provenance_rgcn.contracts import ProvenanceModelOutput
from graph_memory.models.provenance_rgcn.inference import (
    _apply_structured_reranking,
    rank_provenance_task_output,
)
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
        show_progress_bar: bool = False,
    ) -> object:
        _ = batch_size, show_progress_bar
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


class CountingTinyEncoder(TinyEncoder):
    def __init__(self) -> None:
        self.calls = 0

    def encode(self, *args, **kwargs):
        self.calls += 1
        return super().encode(*args, **kwargs)


class ProgressRecordingTinyEncoder(TinyEncoder):
    def __init__(self) -> None:
        self.show_progress_bars: list[bool] = []

    def encode(self, *args, **kwargs):
        self.show_progress_bars.append(kwargs["show_progress_bar"])
        return super().encode(*args, **kwargs)


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


def test_provenance_tensorization_disables_encoder_batch_progress() -> None:
    request, _label = _request_and_label()
    encoder = ProgressRecordingTinyEncoder()

    tensorize_provenance_task(request, encoder=encoder, config=_model_config())

    assert encoder.show_progress_bars == [False]


def test_disconnected_union_matches_separate_provenance_forwards() -> None:
    request, label = _request_and_label()
    second_request = _retask_request(request, "rgcn-second")
    second_label = replace(
        label,
        task_id=second_request.task_id,
        gold_dependency_edges=(),
    )
    config = _model_config()
    tasks = [
        tensorize_provenance_task(item, encoder=TinyEncoder(), config=config)
        for item in (request, second_request)
    ]
    batch = collate_provenance_tasks(tasks)
    model = ExecutionProvenanceRGCN(config)
    model.eval()

    with torch.no_grad():
        batched_output = model(batch)
        split_outputs = split_provenance_output(batch, batched_output)
        separate_outputs = [
            model(collate_provenance_tasks([task])) for task in tasks
        ]

    assert batch.graph_batch.task_node_offsets == [
        0,
        len(request.graph.nodes),
        len(request.graph.nodes) + len(second_request.graph.nodes),
    ]
    assert batch.candidate_offsets == [
        0,
        len(request.candidates),
        len(request.candidates) + len(second_request.candidates),
    ]
    assert batch.transition_offsets[-1] == sum(
        len(task.logical_transitions) for task in tasks
    )
    for task_index in range(batch.task_count):
        node_start = batch.graph_batch.task_node_offsets[task_index]
        node_end = batch.graph_batch.task_node_offsets[task_index + 1]
        candidate_start = batch.candidate_offsets[task_index]
        candidate_end = batch.candidate_offsets[task_index + 1]
        transition_start = batch.transition_offsets[task_index]
        transition_end = batch.transition_offsets[task_index + 1]
        assert bool(
            (
                (batch.candidate_node_indices[candidate_start:candidate_end] >= node_start)
                & (batch.candidate_node_indices[candidate_start:candidate_end] < node_end)
            ).all()
        )
        assert torch.equal(
            batch.candidate_query_indices[candidate_start:candidate_end],
            torch.full(
                (candidate_end - candidate_start,),
                int(batch.graph_batch.query_node_indices[task_index]),
                dtype=torch.long,
            ),
        )
        if transition_end > transition_start:
            transition_nodes = batch.transition_node_indices[
                :, transition_start:transition_end
            ]
            assert bool(((transition_nodes >= node_start) & (transition_nodes < node_end)).all())
    for task_index, (split, separate) in enumerate(
        zip(split_outputs, separate_outputs, strict=True)
    ):
        assert torch.allclose(split.node_states, separate.node_states, atol=1e-6)
        assert torch.allclose(
            split.candidate_logits, separate.candidate_logits, atol=1e-6
        )
        assert torch.allclose(split.edge_logits, separate.edge_logits, atol=1e-6)
        batched_result = rank_provenance_task_output(
            candidate_ids=batch.candidate_ids_by_task[task_index],
            transitions=batch.logical_transitions_by_task[task_index],
            output=split,
            config=config,
            top_k=6,
        )
        separate_result = rank_provenance_task_output(
            candidate_ids=tasks[task_index].candidate_ids,
            transitions=tasks[task_index].logical_transitions,
            output=separate,
            config=config,
            top_k=6,
        )
        assert [item.node_id for item in batched_result.ranked_nodes] == [
            item.node_id for item in separate_result.ranked_nodes
        ]
        assert torch.allclose(
            torch.tensor([item.score for item in batched_result.ranked_nodes]),
            torch.tensor([item.score for item in separate_result.ranked_nodes]),
            atol=1e-6,
        )
        assert {
            (edge["source"], edge["target"])
            for edge in batched_result.trace.retrieved_edges
        } == {
            (edge["source"], edge["target"])
            for edge in separate_result.trace.retrieved_edges
        }
        batched_trace = batched_result.trace.native_trace
        separate_trace = separate_result.trace.native_trace
        assert isinstance(batched_trace, ExecutionProvenanceTrace)
        assert isinstance(separate_trace, ExecutionProvenanceTrace)
        assert [
            (item.source_id, item.target_id, item.decision)
            for item in batched_trace.structured_transitions
        ] == [
            (item.source_id, item.target_id, item.decision)
            for item in separate_trace.structured_transitions
        ]
        assert batched_trace.abstained_source_ids == separate_trace.abstained_source_ids

    training_tasks = [
        materialize_provenance_training_task(
            item,
            item_label,
            _train_pairs(item, item_label),
            encoder=TinyEncoder(),
            config=config,
        )
        for item, item_label in (
            (request, label),
            (second_request, second_label),
        )
    ]
    training_batch = collate_provenance_training_tasks(training_tasks)
    with torch.no_grad():
        output = model(training_batch.tensor)
    batch_loss = compute_provenance_batch_loss(
        training_batch,
        output,
        config,
        ProvenanceRgcnTrainingConfig(),
    )
    single_losses = [
        compute_provenance_loss(
            collate_provenance_tasks([task.tensor]),
            separate_output,
            item_label,
            _train_pairs(item, item_label),
            config,
            ProvenanceRgcnTrainingConfig(),
        )
        for task, separate_output, item, item_label in zip(
            training_tasks,
            separate_outputs,
            (request, second_request),
            (label, second_label),
            strict=True,
        )
    ]
    assert batch_loss.total == pytest.approx(
        torch.stack([loss.total for loss in single_losses]).mean()
    )
    assert batch_loss.comparison_count == sum(
        loss.comparison_count for loss in single_losses
    )


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
    task = ProvenanceTrainPairBuildTask(
        text_request=text_request,
        graph=request.graph,
        label=label,
    )

    result = build_provenance_train_pairs(
        [task],
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

    easy_only = build_provenance_train_pairs(
        [task],
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
        pair["sample_type"] for pair in easy_only.pairs if pair["label"] == 0
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
    assert loaded.payload["schema_version"] == 4
    assert (
        loaded.payload["candidate_loss_protocol"]
        == "provenance-candidate-loss-v2"
    )
    assert loaded.payload["batch_semantics"] == "disconnected_union_task_graphs"
    assert "batch_size" not in loaded.payload["training_config"]

    legacy_batch_payload = deepcopy(payload)
    legacy_batch_payload["schema_version"] = 3
    legacy_batch_payload["training_config"] = {"batch_size": 128}
    legacy_batch_checkpoint = tmp_path / "legacy-batch-v3.pt"
    torch.save(legacy_batch_payload, legacy_batch_checkpoint)
    with pytest.raises(ValueError, match="does not use disconnected-union"):
        load_provenance_rgcn_checkpoint(legacy_batch_checkpoint)

    legacy_payload = deepcopy(payload)
    for field_name in (
        "ablation_name",
        "message_transform_type",
        "edge_weight_policy",
    ):
        legacy_payload["model_config"].pop(field_name)
    legacy_checkpoint = tmp_path / "legacy-full.pt"
    torch.save(legacy_payload, legacy_checkpoint)
    with pytest.raises(ValueError, match="incomplete for schema v4"):
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


def test_provenance_dataloader_reuses_materialized_features_across_epochs() -> None:
    request, label = _request_and_label()
    encoder = CountingTinyEncoder()

    result = train_provenance_rgcn(
        train_requests=[request],
        train_labels=[label],
        train_pairs=_train_pairs(request, label),
        model_config=_model_config(),
        training_config=ProvenanceRgcnTrainingConfig(epochs=2),
        encoder=encoder,
    )

    assert encoder.calls == 2  # one train item plus one ordered dev item
    assert [record["global_step"] for record in result.metric_records] == [1, 2]


def test_provenance_tail_graph_batch_produces_its_own_optimizer_step() -> None:
    request, label = _request_and_label()
    requests = [
        _retask_request(request, f"rgcn-tail-{index}") for index in range(3)
    ]
    labels = [replace(label, task_id=item.task_id) for item in requests]
    pairs = [
        pair
        for item, item_label in zip(requests, labels, strict=True)
        for pair in _train_pairs(item, item_label)
    ]
    result = train_provenance_rgcn(
        train_requests=requests,
        train_labels=labels,
        train_pairs=pairs,
        model_config=_model_config(),
        training_config=ProvenanceRgcnTrainingConfig(
            learning_rate=0.01,
            per_device_graph_batch_size=2,
            epochs=1,
        ),
        encoder=TinyEncoder(),
    )

    metrics = result.metric_records[0]
    assert result.global_step == 2
    assert metrics["train_optimizer_step_count"] == 2
    assert metrics["actual_tasks_per_optimizer_step"] == [2, 1]
    assert metrics["train_task_count"] == 3
    assert cast(int, metrics["materialized_cpu_tensor_bytes"]) > 0
    assert metrics["train_candidates_per_task_max"] == 6
    assert cast(float, metrics["train_tasks_per_second"]) > 0.0


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


def _retask_request(request, task_id: str):
    return replace(request, task_id=task_id, graph=replace(request.graph, task_id=task_id))


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

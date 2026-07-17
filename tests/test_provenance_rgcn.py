from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
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
from graph_memory.training_pairs import build_train_pairs
from graph_memory.training_pairs.config import NegativeSamplingConfig
from graph_memory.training_pairs.requests import TrainPairBuildTask


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
    assert torch.all(uniform_tensor.graph_batch.edge_weights == 1.0)
    assert any(edge.weight != 1.0 for edge in request.graph.edges)


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
    legacy = load_provenance_rgcn_checkpoint(legacy_checkpoint)
    assert legacy.model_config.ablation_name == "full_rgcn"
    assert legacy.model_config.message_transform_type == "typed"
    assert legacy.model_config.edge_weight_policy == "artifact"
    assert "path_loss_weight" not in loaded.payload["training_config"]
    evidence_checkpoint = tmp_path / "evidence.pt"
    torch.save(
        {"schema_version": 2, "method_name": "dense_rgcn_graph_retriever"},
        evidence_checkpoint,
    )
    with pytest.raises(ValueError, match="family mismatch"):
        load_provenance_rgcn_checkpoint(evidence_checkpoint)


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

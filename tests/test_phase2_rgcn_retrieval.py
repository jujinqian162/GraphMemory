import math
from pathlib import Path

import torch
import pytest

from graph_memory.datasets.selection import text_ranking_requests_for_dataset
from graph_memory.models.graph_retriever.checkpoint import load_rgcn_checkpoint
import graph_memory.registry.retrieval_builders as retrieval_builders
from graph_memory.registry.retrieval_builders import build_retrieval
from graph_memory.models.graph_retriever.checkpoint import save_rgcn_checkpoint
from graph_memory.models.graph_retriever.config.defaults import default_model_config
from graph_memory.models.graph_retriever.factory import build_model_from_config
from graph_memory.experiment.config import (
    DenseFtRgcnMethodConfig,
    RgcnMethodConfig,
)
from graph_memory.models.graph_retriever.inference import (
    CheckpointGraphRetrieverLoader,
)
from graph_memory.retrieval.methods.ids import RetrievalMethodId
from graph_memory.retrieval.contracts import RankedNode, RetrievalMethodResult
from graph_memory.retrieval.execution.service import run_retrieval as execute_retrieval
from graph_memory.retrieval.requests import EvidenceGraphRankingRequest
from tests.rgcn_fixtures import (
    FakeRetriever,
    FakeTextEmbeddingProvider,
    tiny_graphs,
    tiny_model_config,
    tiny_task_inputs,
    tiny_training_config,
)
from graph_memory.retrieval.signals import RetrieverSeedSignalProvider


def _ranking_requests(task_inputs):
    return text_ranking_requests_for_dataset("hotpotqa", task_inputs)


class TinyTrainableRetriever:
    name = "dense_rgcn_graph_retriever"

    def rank_task(self, task_input, *, top_k: int):
        return RetrievalMethodResult(
            ranked_nodes=(
                RankedNode(node_id="m0", score=3.0),
                RankedNode(node_id="m1", score=2.0),
                RankedNode(node_id="m2", score=1.0),
            ),
        )


def run_retrieval(
    *,
    method,
    task_inputs,
    graphs,
    top_k,
    checkpoint_path=None,
    text_embedding_provider=None,
    seed_signal_provider=None,
    device="cpu",
):
    if method != "dense_rgcn_graph_retriever":
        raise ValueError(f"Unsupported test method: {method}")
    if checkpoint_path is None:
        raise ValueError("checkpoint_path is required")
    retrieval_method, _provenance, execution_requests = build_retrieval(
        RgcnMethodConfig.model_validate(
            {
                "method": "dense_rgcn_graph_retriever",
                "variant": "full_rgcn",
                "encoder": {
                    "model_name": "fake-encoder",
                    "query_prefix": "query: ",
                    "passage_prefix": "passage: ",
                    "batch_size": 64,
                },
                "pairs": {
                    "random_seed": 13,
                    "easy_random_per_positive": 1,
                    "hard_bm25_per_positive": 1,
                    "hard_dense_per_positive": 0,
                    "hard_graph_neighbor_per_positive": 1,
                    "hard_pool_size": 10,
                },
                "train": {
                    "model": {
                        "hidden_dim": 8,
                        "num_layers": 1,
                        "dropout": 0.0,
                        "ablation": "full_rgcn",
                    },
                    "trainer": {**tiny_training_config().model_dump(), "device": device},
                    "selection": {
                        "best_metric": "dev_composite",
                        "higher_is_better": True,
                    },
                },
            }
        ),
        text_requests=_ranking_requests(task_inputs),
        checkpoint=Path(checkpoint_path),
        evidence_graphs=graphs,
        text_embedding_provider=text_embedding_provider,
        seed_signal_provider=seed_signal_provider,
        device=device,
    )
    return execute_retrieval(
        retrieval_method=retrieval_method,
        requests=execution_requests,
        top_k=top_k,
    )


def write_tiny_checkpoint(
    path: Path,
    *,
    model_config=None,
    method_name: str = "dense_rgcn_graph_retriever",
) -> None:
    effective_model_config = model_config or tiny_model_config()
    model = build_model_from_config(effective_model_config)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(0.01)
    save_rgcn_checkpoint(
        path,
        method_name=method_name,
        model=model,
        epoch=1,
        global_step=1,
        best_dev_metric=1.0,
        model_config=effective_model_config,
        training_config=tiny_training_config(),
    )


def fake_checkpoint_providers(checkpoint_path, method_id, **kwargs):
    return (
        FakeTextEmbeddingProvider(),
        RetrieverSeedSignalProvider(FakeRetriever()),
        load_rgcn_checkpoint(
            checkpoint_path,
            expected_method=method_id.value,
            map_location="cpu",
        ),
    )


def tiny_graph_ranking_request():
    record = tiny_task_inputs()[0]
    text_request = text_ranking_requests_for_dataset("hotpotqa", [record])[0]
    signals = RetrieverSeedSignalProvider(FakeRetriever()).score_task(text_request)
    return EvidenceGraphRankingRequest(
        task_id=record.task_id,
        query_text=record.question,
        candidates=text_request.candidates,
        graph=tiny_graphs()[0],
        initial_scores={signal.node_id: signal.score for signal in signals},
    )


def test_trainable_retriever_ranks_all_memory_nodes_without_labels(tmp_path: Path):
    checkpoint_path = tmp_path / "best.pt"
    write_tiny_checkpoint(checkpoint_path)
    retriever = CheckpointGraphRetrieverLoader().load(
        checkpoint_path,
        text_embedding_provider=FakeTextEmbeddingProvider(),
        seed_signal_provider=RetrieverSeedSignalProvider(FakeRetriever()),
        device="cpu",
    )

    result = retriever.rank_task(tiny_graph_ranking_request(), top_k=2)
    ranked_nodes = result.ranked_nodes
    retrieved_edges = result.trace.retrieved_edges

    top_node_ids = {node.node_id for node in ranked_nodes[:2]}
    assert {node.node_id for node in ranked_nodes} == {"m0", "m1", "m2"}
    assert all(math.isfinite(node.score) for node in ranked_nodes)
    assert len({node.node_id for node in ranked_nodes}) == len(ranked_nodes)
    assert all(
        edge.source in top_node_ids and edge.target in top_node_ids
        for edge in retrieved_edges
    )


def test_evidence_checkpoint_uses_explicit_graph_batch_schema(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "best.pt"
    write_tiny_checkpoint(checkpoint_path)

    checkpoint = load_rgcn_checkpoint(checkpoint_path, map_location="cpu")

    assert checkpoint.payload["schema_version"] == 4
    training = checkpoint.payload["training_config"]
    assert "batch_size" not in training
    assert training["per_device_graph_batch_size"] == 1
    assert set(training) == {
        "optimizer_name",
        "learning_rate",
        "per_device_graph_batch_size",
        "max_grad_norm",
        "random_seed",
        "pos_weight_enabled",
        "epochs",
    }


def test_checkpoint_loader_rejects_legacy_beam_schema(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "legacy-beam.pt"
    write_tiny_checkpoint(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    payload.pop("schema_version")
    payload["model_config"]["decoder_config"] = {"hidden_dim": 8}
    torch.save(payload, checkpoint_path)

    with pytest.raises(ValueError, match="schema_version"):
        load_rgcn_checkpoint(checkpoint_path, map_location="cpu")


def test_evidence_checkpoint_rejects_legacy_batch_size_semantics(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "legacy-batch.pt"
    write_tiny_checkpoint(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    payload["schema_version"] = 2
    payload["training_config"] = {
        "optimizer_name": "AdamW",
        "learning_rate": 0.01,
        "batch_size": 128,
        "max_grad_norm": 1.0,
        "random_seed": 13,
        "pos_weight_enabled": False,
        "epochs": 1,
    }
    torch.save(payload, checkpoint_path)

    with pytest.raises(ValueError, match="batch_size"):
        load_rgcn_checkpoint(checkpoint_path, map_location="cpu")


def test_edge_view_retriever_excludes_hidden_edges_from_prediction_subgraph(
    tmp_path: Path,
):
    checkpoint_path = tmp_path / "best.pt"
    model_config = default_model_config(
        method_name="dense_rgcn_graph_retriever",
        encoder_model="fake-encoder",
        encoder_dim=4,
        query_prefix="query: ",
        passage_prefix="passage: ",
        encoder_batch_size=64,
        hidden_dim=8,
        num_layers=1,
        dropout=0.0,
        ablation_name="wo_bridge",
    )
    write_tiny_checkpoint(checkpoint_path, model_config=model_config)
    retriever = CheckpointGraphRetrieverLoader().load(
        checkpoint_path,
        text_embedding_provider=FakeTextEmbeddingProvider(),
        seed_signal_provider=RetrieverSeedSignalProvider(FakeRetriever()),
        device="cpu",
    )

    result = retriever.rank_task(tiny_graph_ranking_request(), top_k=3)
    retrieved_edges = result.trace.retrieved_edges

    assert all(edge.edge_type != "bridge" for edge in retrieved_edges)


def test_trainable_method_runs_from_static_retrieval_dispatch(
    tmp_path: Path,
):
    checkpoint_path = tmp_path / "best.pt"
    write_tiny_checkpoint(checkpoint_path)

    predictions = run_retrieval(
        method="dense_rgcn_graph_retriever",
        task_inputs=tiny_task_inputs(),
        graphs=tiny_graphs(),
        top_k=2,
        checkpoint_path=checkpoint_path,
        text_embedding_provider=FakeTextEmbeddingProvider(),
        seed_signal_provider=RetrieverSeedSignalProvider(FakeRetriever()),
    )

    assert predictions[0].method == "dense_rgcn_graph_retriever"
    assert predictions[0].retrieved_subgraph.nodes == tuple(
        ranked_node.node_id for ranked_node in predictions[0].ranked_nodes[:2]
    )
    assert predictions[0].metadata is None


def test_evidence_rgcn_builder_accepts_dense_ft_seeded_rgcn_checkpoint(
    tmp_path: Path,
):
    checkpoint_path = tmp_path / "best.pt"
    seeded_model_config = tiny_model_config().model_copy(
        update={
            "method_name": RetrievalMethodId.DENSE_FT_RGCN_GRAPH_RETRIEVER.value
        }
    )
    write_tiny_checkpoint(
        checkpoint_path,
        model_config=seeded_model_config,
        method_name=RetrievalMethodId.DENSE_FT_RGCN_GRAPH_RETRIEVER.value,
    )

    retrieval_method, provenance, _requests = build_retrieval(
        DenseFtRgcnMethodConfig.model_construct(
            method="dense_ft_rgcn_graph_retriever",
            variant="full_rgcn",
        ),
        text_requests=_ranking_requests(tiny_task_inputs()),
        checkpoint=checkpoint_path,
        evidence_graphs=tiny_graphs(),
        text_embedding_provider=FakeTextEmbeddingProvider(),
        seed_signal_provider=RetrieverSeedSignalProvider(FakeRetriever()),
        device="cpu",
    )

    assert retrieval_method.name == RetrievalMethodId.DENSE_FT_RGCN_GRAPH_RETRIEVER.value
    assert provenance.method is RetrievalMethodId.DENSE_FT_RGCN_GRAPH_RETRIEVER
    assert provenance.model == checkpoint_path
    assert provenance.encoder is not None
    assert provenance.encoder.model_name == "fake-encoder"


def test_run_retrieval_passes_device_to_trainable_retriever(
    monkeypatch, tmp_path: Path
):
    checkpoint_path = tmp_path / "best.pt"
    write_tiny_checkpoint(checkpoint_path)
    captured: dict[str, object] = {}

    def fake_from_checkpoint(_loader, checkpoint_path_arg, *, device="cpu", **kwargs):
        captured["checkpoint_path"] = checkpoint_path_arg
        captured["device"] = device
        return TinyTrainableRetriever()

    monkeypatch.setattr(
        CheckpointGraphRetrieverLoader, "load", fake_from_checkpoint
    )
    monkeypatch.setattr(
        retrieval_builders, "_evidence_rgcn_providers", fake_checkpoint_providers
    )

    predictions = run_retrieval(
        method="dense_rgcn_graph_retriever",
        task_inputs=tiny_task_inputs(),
        graphs=tiny_graphs(),
        top_k=1,
        checkpoint_path=checkpoint_path,
        device="cuda:7",
    )

    assert captured["checkpoint_path"] == checkpoint_path
    assert captured["device"] == "cuda:7"
    assert predictions[0].method == "dense_rgcn_graph_retriever"

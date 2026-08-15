import math
from pathlib import Path

import torch

from graph_memory.datasets.selection import text_ranking_requests_for_dataset
from graph_memory.models.graph_retriever.checkpoint import load_rgcn_checkpoint
import graph_memory.registry.retrieval_builders as retrieval_builders
from graph_memory.registry.retrieval_builders import build_retrieval
from graph_memory.models.graph_retriever.checkpoint import save_rgcn_checkpoint
from graph_memory.models.graph_retriever.factory import build_model_from_config
from graph_memory.experiment.config import (
    RgcnMethodConfig,
)
from graph_memory.models.graph_retriever.inference import (
    CheckpointGraphRetrieverLoader,
)
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
                    "trainer": {
                        **tiny_training_config().model_dump(),
                        "device": device,
                    },
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

    monkeypatch.setattr(CheckpointGraphRetrieverLoader, "load", fake_from_checkpoint)
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

import json
import math
from inspect import Parameter, signature
from pathlib import Path

import torch

import graph_memory.registry.retrieval_builders as retrieval_builders
from graph_memory.models.graph_retriever.checkpoint import save_trainable_checkpoint
from graph_memory.models.graph_retriever.config.defaults import default_model_config
from graph_memory.models.graph_retriever.factory import build_model_from_config
from graph_memory.models.graph_retriever.inference import CheckpointGraphRetrieverLoader
from graph_memory.registry.retrieval import CheckpointGraphBuildPayload
from graph_memory.registry.retrieval_builders import RETRIEVAL_REGISTRY
from graph_memory.retrieval.methods.trainable_graph import TrainableGraphRetrievalMethod
from graph_memory.retrieval.execution.service import run_retrieval as execute_retrieval
from graph_memory.retrieval.contracts import RankedNode, RetrievalMethodResult
from graph_memory.retrieval_registry import METHOD_REGISTRY, get_method_spec, get_supported_methods
from graph_memory.validation import validate_ranked_results
from scripts.run_retrieval import build_parser as build_retrieval_parser
from scripts.run_retrieval import main as run_retrieval_cli_main
from tests.test_phase2_rgcn_training import (
    FakeRetriever,
    FakeTextEmbeddingProvider,
    tiny_graphs,
    tiny_model_config,
    tiny_task_inputs,
    tiny_training_config,
)
from graph_memory.retrieval.signals import RetrieverSeedSignalProvider


class TinyTrainableRetriever:
    name = "dense_rgcn_graph_retriever"

    def rank_task(self, task_input, *, top_k: int):
        return RetrievalMethodResult(
            ranked_nodes=[
                RankedNode(node_id="m0", score=3.0),
                RankedNode(node_id="m1", score=2.0),
                RankedNode(node_id="m2", score=1.0),
            ],
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
    settings = RETRIEVAL_REGISTRY.settings_from_runtime(
        method=method,
        top_k=top_k,
        checkpoint=checkpoint_path,
        device=device,
    )
    method_object = RETRIEVAL_REGISTRY.build(
        settings,
        CheckpointGraphBuildPayload(
            task_inputs=task_inputs,
            graphs=graphs,
            text_embedding_provider=text_embedding_provider,
            seed_signal_provider=seed_signal_provider,
        )
    )
    return execute_retrieval(retrieval_method=method_object, task_inputs=task_inputs, top_k=top_k)


def write_tiny_checkpoint(path: Path, *, model_config=None) -> None:
    effective_model_config = model_config or tiny_model_config()
    model = build_model_from_config(effective_model_config)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(0.01)
    save_trainable_checkpoint(
        path,
        method_name="dense_rgcn_graph_retriever",
        model=model,
        optimizer_state_dict={},
        scheduler_state_dict={},
        epoch=1,
        global_step=1,
        best_dev_metric=1.0,
        model_config=effective_model_config,
        training_config=tiny_training_config(),
    )


def fake_checkpoint_providers(settings, payload):
    return FakeTextEmbeddingProvider(), RetrieverSeedSignalProvider(FakeRetriever())


def test_trainable_retriever_ranks_all_memory_nodes_without_labels(tmp_path: Path):
    checkpoint_path = tmp_path / "best.pt"
    write_tiny_checkpoint(checkpoint_path)
    retriever = TrainableGraphRetrievalMethod.from_checkpoint(
        checkpoint_path,
        graphs=tiny_graphs(),
        text_embedding_provider=FakeTextEmbeddingProvider(),
        seed_signal_provider=RetrieverSeedSignalProvider(FakeRetriever()),
    )

    result = retriever.rank_task(tiny_task_inputs()[0], top_k=2)
    ranked_nodes = result.ranked_nodes
    retrieved_edges = result.trace.retrieved_edges

    top_node_ids = {node.node_id for node in ranked_nodes[:2]}
    assert {node.node_id for node in ranked_nodes} == {"m0", "m1", "m2"}
    assert all(math.isfinite(node.score) for node in ranked_nodes)
    assert ranked_nodes == sorted(ranked_nodes, key=lambda node: (-node.score, node.node_id))
    assert all(edge["source"] in top_node_ids and edge["target"] in top_node_ids for edge in retrieved_edges)


def test_checkpoint_loader_requires_assembled_runtime_providers() -> None:
    parameters = signature(CheckpointGraphRetrieverLoader.load).parameters

    assert parameters["text_embedding_provider"].default is Parameter.empty
    assert parameters["seed_signal_provider"].default is Parameter.empty
    assert "dense_encoder" not in parameters


def test_edge_view_retriever_excludes_hidden_edges_from_prediction_subgraph(tmp_path: Path):
    checkpoint_path = tmp_path / "best.pt"
    model_config = default_model_config(
        encoder_model="fake-encoder",
        encoder_dim=4,
        query_prefix="query: ",
        passage_prefix="passage: ",
        hidden_dim=8,
        num_layers=1,
        dropout=0.0,
        ablation_name="wo_bridge",
    )
    write_tiny_checkpoint(checkpoint_path, model_config=model_config)
    retriever = TrainableGraphRetrievalMethod.from_checkpoint(
        checkpoint_path,
        graphs=tiny_graphs(),
        text_embedding_provider=FakeTextEmbeddingProvider(),
        seed_signal_provider=RetrieverSeedSignalProvider(FakeRetriever()),
    )

    result = retriever.rank_task(tiny_task_inputs()[0], top_k=3)
    retrieved_edges = result.trace.retrieved_edges

    assert all(edge["edge_type"] != "bridge" for edge in retrieved_edges)


def test_trainable_method_is_registered_and_run_retrieval_accepts_checkpoint(tmp_path: Path):
    checkpoint_path = tmp_path / "best.pt"
    write_tiny_checkpoint(checkpoint_path)

    assert "dense_rgcn_graph_retriever" in get_supported_methods()
    spec = get_method_spec("dense_rgcn_graph_retriever")
    assert spec.requires_graphs is True
    assert spec.requires_checkpoint is True
    assert spec.seed_method == "dense"
    assert METHOD_REGISTRY["dense_rgcn_graph_retriever"].builder_id == "trainable_graph"
    choices = build_retrieval_parser()._option_string_actions["--method"].choices
    assert choices is not None
    assert "dense_rgcn_graph_retriever" in choices

    predictions = run_retrieval(
        method="dense_rgcn_graph_retriever",
        task_inputs=tiny_task_inputs(),
        graphs=tiny_graphs(),
        top_k=2,
        checkpoint_path=checkpoint_path,
        text_embedding_provider=FakeTextEmbeddingProvider(),
        seed_signal_provider=RetrieverSeedSignalProvider(FakeRetriever()),
    )

    validate_ranked_results(predictions, {task["task_id"]: task for task in tiny_task_inputs()})
    assert predictions[0]["method"] == "dense_rgcn_graph_retriever"
    assert predictions[0]["retrieved_subgraph"]["nodes"] == [
        ranked_node["node_id"] for ranked_node in predictions[0]["ranked_nodes"][:2]
    ]


def test_run_retrieval_cli_writes_trainable_ranked_results(monkeypatch, tmp_path: Path):
    checkpoint_path = tmp_path / "best.pt"
    tasks_path = tmp_path / "test.input.json"
    graphs_path = tmp_path / "test.graphs.json"
    output_path = tmp_path / "ranked.json"
    checkpoint_path.write_bytes(b"placeholder")
    tasks_path.write_text(json.dumps(tiny_task_inputs()), encoding="utf-8")
    graphs_path.write_text(json.dumps(tiny_graphs()), encoding="utf-8")

    def fake_from_checkpoint(checkpoint_path_arg, *, graphs, device="cpu", **kwargs):
        assert checkpoint_path_arg == checkpoint_path
        assert device == "cuda:7"
        assert graphs == tiny_graphs()
        return TinyTrainableRetriever()

    monkeypatch.setattr(TrainableGraphRetrievalMethod, "from_checkpoint", fake_from_checkpoint)
    monkeypatch.setattr(retrieval_builders, "_checkpoint_graph_providers", fake_checkpoint_providers)

    exit_code = run_retrieval_cli_main(
        [
            "--method",
            "dense_rgcn_graph_retriever",
            "--tasks",
            str(tasks_path),
            "--graphs",
            str(graphs_path),
            "--checkpoint",
            str(checkpoint_path),
            "--output",
            str(output_path),
            "--top_k",
            "2",
            "--device",
            "cuda:7",
        ],
    )

    assert exit_code == 0
    predictions = json.loads(output_path.read_text(encoding="utf-8"))
    run_summary = json.loads(output_path.with_name("ranked.run_summary.json").read_text(encoding="utf-8"))
    assert predictions[0]["method"] == "dense_rgcn_graph_retriever"
    assert run_summary["status"] == "success"
    assert "labels" not in run_summary["inputs"]


def test_run_retrieval_passes_device_to_trainable_retriever(monkeypatch, tmp_path: Path):
    checkpoint_path = tmp_path / "best.pt"
    checkpoint_path.write_bytes(b"placeholder")
    captured: dict[str, object] = {}

    def fake_from_checkpoint(checkpoint_path_arg, *, graphs, device="cpu", **kwargs):
        captured["checkpoint_path"] = checkpoint_path_arg
        captured["graphs"] = graphs
        captured["device"] = device
        return TinyTrainableRetriever()

    monkeypatch.setattr(TrainableGraphRetrievalMethod, "from_checkpoint", fake_from_checkpoint)
    monkeypatch.setattr(retrieval_builders, "_checkpoint_graph_providers", fake_checkpoint_providers)

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
    assert predictions[0]["method"] == "dense_rgcn_graph_retriever"

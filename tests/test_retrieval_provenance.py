from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from graph_memory.config import CONFIG_LOADER
from graph_memory.contracts.graphs import MemoryGraph
from graph_memory.contracts.tasks import MemoryTaskInput
from graph_memory.io import read_json, write_json
from graph_memory.models.dense_finetune.metadata import (
    DenseFinetuneModelMetadata,
    DenseFinetuneSelectionMetadata,
    write_dense_ft_model_metadata,
)
from graph_memory.models.graph_retriever.checkpoint import save_rgcn_checkpoint
from graph_memory.models.graph_retriever.config.records import (
    NodeFeatureConfig,
    RgcnModelConfig,
    RgcnTrainingConfig,
)
from graph_memory.registry import Registry
from graph_memory.registry.retrieval import (
    Bm25RetrievalSettings,
    CheckpointGraphBuildPayload,
    CheckpointGraphRetrievalSettings,
    DenseFinetunedRetrievalSettings,
    FlatRetrievalBuildPayload,
    RetrievalMethodId,
)
from graph_memory.registry.stage_configs import RetrieveIO, RetrieveStageConfig
from graph_memory.retrieval.contracts import RetrievalMethodResult
from graph_memory.retrieval.methods.trainable_graph import TrainableGraphRetrievalMethod
from scripts import run_retrieval


class FakeEncoder:
    def encode(self, texts, batch_size=64, normalize_embeddings=True):
        vectors = []
        for text in texts:
            lowered = text.lower()
            vectors.append(
                [
                    1.0 if "eiffel" in lowered else 0.0,
                    1.0 if "paris" in lowered else 0.0,
                    1.0 if "seine" in lowered else 0.0,
                ]
            )
        array = np.array(vectors, dtype=float)
        if normalize_embeddings:
            norms = np.linalg.norm(array, axis=1, keepdims=True)
            norms[norms == 0.0] = 1.0
            array = array / norms
        return array


def retrieval_task_inputs() -> list[MemoryTaskInput]:
    return [
        {
            "task_id": "hotpot_ex1",
            "query": "Which river runs through the city with the Eiffel Tower?",
            "memory_items": [
                {
                    "id": "m0",
                    "node_type": "document_sentence",
                    "text": "The Eiffel Tower is in Paris.",
                    "source": "Eiffel Tower",
                    "sentence_id": 0,
                    "position": 0,
                },
                {
                    "id": "m1",
                    "node_type": "document_sentence",
                    "text": "The Seine runs through Paris.",
                    "source": "Paris",
                    "sentence_id": 0,
                    "position": 1,
                },
            ],
        }
    ]


class _FakeProvider:
    embedding_dim = 4

    def encode_task_nodes(self, task_input, node_ids):
        return torch.zeros((len(node_ids), 4), dtype=torch.float32)

    def score_task(self, task_input):
        return []


class _FakeRgcnMethod:
    name = RetrievalMethodId.DENSE_RGCN_GRAPH_RETRIEVER.value

    def rank_task(self, task_input, *, top_k: int) -> RetrievalMethodResult:
        return RetrievalMethodResult(ranked_nodes=[])


def _rgcn_model_config() -> RgcnModelConfig:
    return RgcnModelConfig(
        method_name=RetrievalMethodId.DENSE_RGCN_GRAPH_RETRIEVER.value,
        encoder_model="checkpoint-encoder",
        encoder_dim=4,
        query_prefix="Q: ",
        passage_prefix="P: ",
        encoder_batch_size=11,
        hidden_dim=8,
        num_layers=1,
        dropout=0.0,
        feature_config=NodeFeatureConfig(),
        relation_vocab=("query_overlap_forward",),
        graph_encoder_type="rgcn",
        message_transform_type="typed",
        edge_weight_policy="artifact",
        enabled_edge_types=("query_overlap",),
        ablation_name="full_rgcn",
    )


def test_bm25_builder_provenance_omits_model_device_and_encoder() -> None:
    built = Registry.retrieval.build(
        Bm25RetrievalSettings(top_k=2),
        FlatRetrievalBuildPayload(task_inputs=retrieval_task_inputs()),
    )

    assert built.method.name == "bm25"
    assert built.provenance.method is RetrievalMethodId.BM25
    assert built.provenance.model is None
    assert built.provenance.device is None
    assert built.provenance.encoder is None


def test_dense_ft_builder_provenance_uses_model_directory_device_and_metadata(tmp_path: Path) -> None:
    model_dir = tmp_path / "best_model"
    write_dense_ft_model_metadata(
        model_dir=model_dir,
        metadata=DenseFinetuneModelMetadata(
            base_model="metadata-base",
            query_prefix="Q: ",
            passage_prefix="P: ",
            batch_size=7,
            device="cuda:0",
            selection=DenseFinetuneSelectionMetadata(
                selected_metric="eval_dev_cos_sim_map@100",
                higher_is_better=True,
            ),
        ),
    )

    built = Registry.retrieval.build(
        DenseFinetunedRetrievalSettings(top_k=2, checkpoint=model_dir, device="cuda:7"),
        FlatRetrievalBuildPayload(task_inputs=retrieval_task_inputs(), dense_encoder=FakeEncoder()),
    )

    assert built.provenance.method is RetrievalMethodId.DENSE_FT
    assert built.provenance.model == model_dir
    assert built.provenance.device == "cuda:7"
    assert built.provenance.encoder is not None
    assert built.provenance.encoder.model_name == "metadata-base"
    assert built.provenance.encoder.query_prefix == "Q: "
    assert built.provenance.encoder.passage_prefix == "P: "
    assert built.provenance.encoder.batch_size == 7


def test_rgcn_builder_provenance_uses_checkpoint_encoder_and_device(
    monkeypatch,
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "best.pt"
    save_rgcn_checkpoint(
        checkpoint_path,
        method_name=RetrievalMethodId.DENSE_RGCN_GRAPH_RETRIEVER.value,
        model=torch.nn.Linear(4, 1),
        optimizer_state_dict={},
        scheduler_state_dict={},
        epoch=1,
        global_step=1,
        best_dev_metric=0.5,
        model_config=_rgcn_model_config(),
        training_config=RgcnTrainingConfig(),
    )
    monkeypatch.setattr(
        TrainableGraphRetrievalMethod,
        "from_checkpoint",
        lambda *args, **kwargs: _FakeRgcnMethod(),
    )
    task_inputs: list[MemoryTaskInput] = [
        {
            "task_id": "task",
            "query": "query",
            "memory_items": [
                {
                    "id": "m0",
                    "node_type": "document_sentence",
                    "text": "evidence",
                    "source": "doc",
                    "sentence_id": 0,
                    "position": 0,
                }
            ],
        }
    ]
    graphs: list[MemoryGraph] = [
        {
            "task_id": "task",
            "nodes": [
                {"id": "q", "node_type": "question", "text": "query"},
                task_inputs[0]["memory_items"][0],
            ],
            "edges": [
                {
                    "source": "q",
                    "target": "m0",
                    "edge_type": "query_overlap",
                    "weight": 1.0,
                    "directed": True,
                }
            ],
        }
    ]
    provider = _FakeProvider()

    built = Registry.retrieval.build(
        CheckpointGraphRetrievalSettings(
            top_k=2,
            checkpoint=checkpoint_path,
            device="cuda:7",
        ),
        CheckpointGraphBuildPayload(
            task_inputs=task_inputs,
            graphs=graphs,
            text_embedding_provider=provider,
            seed_signal_provider=provider,
        ),
    )

    assert built.provenance.method is RetrievalMethodId.DENSE_RGCN_GRAPH_RETRIEVER
    assert built.provenance.model == checkpoint_path
    assert built.provenance.device == "cuda:7"
    assert built.provenance.encoder is not None
    assert built.provenance.encoder.model_name == "checkpoint-encoder"
    assert built.provenance.encoder.query_prefix == "Q: "
    assert built.provenance.encoder.passage_prefix == "P: "
    assert built.provenance.encoder.batch_size == 11


def test_run_retrieval_summary_serializes_builder_provenance_for_bm25(tmp_path: Path) -> None:
    tasks_path = tmp_path / "tasks.json"
    output_path = tmp_path / "predictions.json"
    summary_path = tmp_path / "predictions.run_summary.json"
    config_path = tmp_path / "retrieve.json"
    write_json(tasks_path, retrieval_task_inputs())
    config = RetrieveStageConfig(
        io=RetrieveIO(
            tasks=tasks_path,
            graphs=None,
            output=output_path,
            summary=summary_path,
        ),
        job=Bm25RetrievalSettings(top_k=2),
    )
    write_json(config_path, CONFIG_LOADER.to_json(config))

    assert run_retrieval.main(["--config", str(config_path)]) == 0
    summary = read_json(summary_path)

    assert summary["effective_config"]["provenance"] == {
        "method": "bm25",
        "model": None,
        "device": None,
        "encoder": None,
        "importance": None,
    }
    assert "encoder_model" not in summary["effective_config"]

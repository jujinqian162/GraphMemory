from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from graph_memory.experiment.artifacts import (
    ArtifactKind,
    ArtifactPublisher,
    FrozenEmbeddingsArtifactRef,
    ProcessedAssetStore,
)
from graph_memory.io import write_json
from graph_memory.models.frozen_embeddings import node_ids_digest
from graph_memory.stages.encodings import (
    resolve_encoding_devices,
    write_sentence_transformer_embeddings,
)
from graph_memory.stages.frozen_embeddings import FrozenEmbeddingStore


class PoolRecordingSentenceTransformer:
    def __init__(self) -> None:
        self.target_devices: list[str] | None = None
        self.multi_process_chunks: list[list[str]] = []
        self.pool_stopped = False

    def get_sentence_embedding_dimension(self) -> int:
        return 3

    def encode(self, texts, **_kwargs):
        raise AssertionError("multi-GPU encoding must not use the single-device path")

    def start_multi_process_pool(self, target_devices):
        self.target_devices = target_devices
        return {"processes": [object(), object()]}

    def encode_multi_process(self, texts, pool, **_kwargs):
        assert pool["processes"]
        self.multi_process_chunks.append(list(texts))
        return np.asarray(
            [[float(len(text)), float(index), 1.0] for index, text in enumerate(texts)],
            dtype=np.float32,
        )

    def stop_multi_process_pool(self, pool) -> None:
        assert pool["processes"]
        self.pool_stopped = True


def test_frozen_encoding_uses_root_device_when_gpu_pool_is_disabled() -> None:
    assert resolve_encoding_devices(False, "cuda:7") == ("cuda:7",)


def test_frozen_encoding_uses_all_pytorch_visible_devices_when_gpu_pool_is_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)

    assert resolve_encoding_devices(True, "cuda:7") == (
        "cuda:0",
        "cuda:1",
        "cuda:2",
        "cuda:3",
    )


def test_frozen_encoding_requires_a_pytorch_visible_cuda_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)

    with pytest.raises(RuntimeError, match="visible to PyTorch"):
        resolve_encoding_devices(True, "cuda:7")
def test_frozen_encoder_streams_sentence_transformers_multi_gpu_pool(
    tmp_path: Path,
) -> None:
    model = PoolRecordingSentenceTransformer()
    output = tmp_path / "embeddings.npy"

    dimension = write_sentence_transformer_embeddings(
        model,
        ["a", "bb", "ccc", "dddd", "eeeee"],
        output=output,
        pool_devices=("cuda:1", "cuda:3"),
        batch_size=2,
        chunk_size=2,
    )

    matrix = np.load(output, allow_pickle=False)
    assert dimension == 3
    assert matrix.shape == (5, 3)
    assert matrix.dtype == np.float32
    assert model.target_devices == ["cuda:1", "cuda:3"]
    assert [len(chunk) for chunk in model.multi_process_chunks] == [2, 2, 1]
    assert model.pool_stopped


def test_frozen_embedding_artifact_loads_as_memory_mapped_task_partitions(
    tmp_path: Path,
) -> None:
    with ArtifactPublisher(
        ProcessedAssetStore(tmp_path / "processed"),
        kind=ArtifactKind.FROZEN_EMBEDDINGS,
        namespace="hotpotqa",
        origin={"stage": "test"},
    ) as publisher:
        np.save(
            publisher.workspace / "embeddings.npy",
            np.arange(12, dtype=np.float32).reshape(4, 3),
            allow_pickle=False,
        )
        write_json(
            publisher.workspace / "index.json",
            {
                "version": 1,
                "family": "evidence",
                "embedding_dim": 3,
                "row_count": 4,
                "groups": [
                    {
                        "split": "train",
                        "task_id": "train-task",
                        "row_start": 0,
                        "row_end": 2,
                        "node_ids_digest": node_ids_digest(["q", "m0"]),
                    },
                    {
                        "split": "dev",
                        "task_id": "dev-task",
                        "row_start": 2,
                        "row_end": 4,
                        "node_ids_digest": node_ids_digest(["q", "m1"]),
                    },
                ],
            },
        )
        reference = publisher.publish(
            {"embeddings": "embeddings.npy", "index": "index.json"}
        )

    assert isinstance(reference, FrozenEmbeddingsArtifactRef)
    store = FrozenEmbeddingStore(reference)
    train = store.partition("train")["train-task"]
    train.validate_node_ids(["q", "m0"])
    assert train.values.tolist() == [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]

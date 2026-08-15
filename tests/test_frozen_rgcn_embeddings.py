from __future__ import annotations

from pathlib import Path

import numpy as np

from graph_memory.experiment.artifacts import (
    ArtifactKind,
    ArtifactPublisher,
    FrozenEmbeddingsArtifactRef,
    ProcessedAssetStore,
)
from graph_memory.io import write_json
from graph_memory.models.frozen_embeddings import node_ids_digest
from graph_memory.stages.frozen_embeddings import FrozenEmbeddingStore


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

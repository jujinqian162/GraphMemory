from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray

from graph_memory.experiment.artifacts import (
    FrozenEmbeddingsArtifactRef,
    artifact_payload_path,
)
from graph_memory.io import read_json
from graph_memory.models.frozen_embeddings import (
    EmbeddingSplit,
    FrozenEmbeddingIndex,
    FrozenTaskEmbeddings,
)


class FrozenEmbeddingStore:
    """Memory-mapped frozen embeddings indexed by split and task ID."""

    def __init__(self, reference: FrozenEmbeddingsArtifactRef) -> None:
        index = FrozenEmbeddingIndex.model_validate(
            read_json(artifact_payload_path(reference, "index"))
        )
        matrix = np.load(
            artifact_payload_path(reference, "embeddings"),
            mmap_mode="c",
            allow_pickle=False,
        )
        if matrix.dtype != np.float32 or matrix.shape != (
            index.row_count,
            index.embedding_dim,
        ):
            raise ValueError(
                "Frozen embedding matrix does not match its index: "
                f"matrix={matrix.shape}/{matrix.dtype} "
                f"index={(index.row_count, index.embedding_dim)}/float32."
            )
        self.reference = reference
        self.index = index
        self._matrix: NDArray[np.float32] = matrix

    @property
    def embedding_dim(self) -> int:
        return self.index.embedding_dim

    def partition(self, split: EmbeddingSplit) -> dict[str, FrozenTaskEmbeddings]:
        result: dict[str, FrozenTaskEmbeddings] = {}
        for group in self.index.groups:
            if group.split != split:
                continue
            if group.task_id in result:
                raise ValueError(
                    f"Duplicate frozen embedding task_id={group.task_id!r} in split={split}."
                )
            values = torch.from_numpy(self._matrix[group.row_start : group.row_end])
            result[group.task_id] = FrozenTaskEmbeddings(
                task_id=group.task_id,
                node_ids_digest=group.node_ids_digest,
                values=values,
            )
        if not result:
            raise ValueError(f"Frozen embeddings contain no split={split!r} groups.")
        return result


__all__ = ["FrozenEmbeddingStore"]

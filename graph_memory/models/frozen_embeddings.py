from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal

import torch
from pydantic import Field

from graph_memory.contracts.model import DomainModel


EmbeddingSplit = Literal["train", "dev"]


class FrozenEmbeddingGroup(DomainModel):
    split: EmbeddingSplit
    task_id: str = Field(min_length=1)
    row_start: int = Field(ge=0)
    row_end: int = Field(gt=0)
    node_ids_digest: str = Field(min_length=1)


class FrozenEmbeddingIndex(DomainModel):
    version: Literal[1] = 1
    family: Literal["evidence"]
    embedding_dim: int = Field(gt=0)
    row_count: int = Field(gt=0)
    groups: tuple[FrozenEmbeddingGroup, ...] = Field(min_length=1)


@dataclass(frozen=True)
class FrozenTaskEmbeddings:
    task_id: str
    node_ids_digest: str
    values: torch.Tensor

    def validate_node_ids(self, node_ids: tuple[str, ...] | list[str]) -> None:
        observed = node_ids_digest(node_ids)
        if observed != self.node_ids_digest:
            raise ValueError(
                "Frozen embedding node order changed for "
                f"task_id={self.task_id!r}."
            )
        if self.values.shape[0] != len(node_ids):
            raise ValueError(
                "Frozen embedding row count changed for "
                f"task_id={self.task_id!r}."
            )


def node_ids_digest(node_ids: tuple[str, ...] | list[str]) -> str:
    digest = hashlib.sha256()
    for node_id in node_ids:
        encoded = node_id.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


__all__ = [
    "EmbeddingSplit",
    "FrozenEmbeddingGroup",
    "FrozenEmbeddingIndex",
    "FrozenTaskEmbeddings",
    "node_ids_digest",
]

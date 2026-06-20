from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from graph_memory.contracts.common import NodeId, TaskId
from graph_memory.embeddings.contracts import SentenceEncoder


def format_dense_query(request: Any, *, query_prefix: str) -> str:
    return query_prefix + request.query_text


def format_dense_passage(candidate: Any, *, passage_prefix: str) -> str:
    return passage_prefix + candidate.text


@dataclass(frozen=True)
class DenseTaskEncodingRequest:
    ranking_request: Any
    node_ids: tuple[NodeId, ...]


@dataclass(frozen=True)
class DenseTaskEncodingResult:
    task_id: TaskId
    node_ids: tuple[NodeId, ...]
    embeddings: NDArray[np.float64]


class DenseEncodingService:
    def __init__(
        self,
        *,
        encoder: SentenceEncoder,
        query_prefix: str = "query: ",
        passage_prefix: str = "passage: ",
        batch_size: int = 64,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("Dense encoder batch_size must be positive.")
        self.encoder = encoder
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.batch_size = batch_size
        self._embedding_dim: int | None = None

    @property
    def embedding_dim(self) -> int:
        if self._embedding_dim is None:
            for getter_name in ("get_embedding_dimension", "get_sentence_embedding_dimension"):
                getter = getattr(self.encoder, getter_name, None)
                if callable(getter):
                    value = getter()
                    if isinstance(value, int) and value > 0:
                        self._embedding_dim = value
                        break
            if self._embedding_dim is None:
                matrix = self._encode_texts(["dimension probe"], batch_size=1)
                self._embedding_dim = int(matrix.shape[1])
        return self._embedding_dim

    def encode_task(self, request: DenseTaskEncodingRequest) -> DenseTaskEncodingResult:
        return self.encode_tasks([request])[0]

    def encode_tasks(
        self,
        requests: Sequence[DenseTaskEncodingRequest],
    ) -> list[DenseTaskEncodingResult]:
        if not requests:
            return []

        flattened_texts: list[str] = []
        row_counts: list[int] = []
        for request in requests:
            texts = self._texts_for_request(request)
            flattened_texts.extend(texts)
            row_counts.append(len(texts))

        matrix = self._encode_texts(flattened_texts, batch_size=self.batch_size)
        results: list[DenseTaskEncodingResult] = []
        row_offset = 0
        for request, row_count in zip(requests, row_counts, strict=True):
            next_offset = row_offset + row_count
            results.append(
                DenseTaskEncodingResult(
                    task_id=request.ranking_request.task_id,
                    node_ids=request.node_ids,
                    embeddings=matrix[row_offset:next_offset].copy(),
                )
            )
            row_offset = next_offset
        return results

    def _texts_for_request(self, request: DenseTaskEncodingRequest) -> list[str]:
        ranking_request = request.ranking_request
        text_by_node_id = {"q": format_dense_query(ranking_request, query_prefix=self.query_prefix)}
        for candidate in ranking_request.candidates:
            text_by_node_id[candidate.item_id] = format_dense_passage(
                candidate,
                passage_prefix=self.passage_prefix,
            )
        try:
            return [text_by_node_id[node_id] for node_id in request.node_ids]
        except KeyError as error:
            raise ValueError(
                f"Unknown dense encoding node_id={error.args[0]} for task_id={ranking_request.task_id}."
            ) from error

    def _encode_texts(self, texts: Sequence[str], *, batch_size: int) -> NDArray[np.float64]:
        embeddings = self.encoder.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
        )
        matrix = np.asarray(embeddings, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != len(texts) or matrix.shape[1] <= 0:
            raise ValueError(
                "Dense encoder returned an invalid embedding shape: "
                f"expected=({len(texts)}, embedding_dim) observed={matrix.shape}."
            )
        observed_dim = int(matrix.shape[1])
        if self._embedding_dim is not None and observed_dim != self._embedding_dim:
            raise ValueError(
                "Dense encoder returned an inconsistent embedding shape: "
                f"expected_dim={self._embedding_dim} observed_dim={observed_dim}."
            )
        self._embedding_dim = observed_dim
        return matrix

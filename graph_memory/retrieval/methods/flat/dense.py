from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np

from graph_memory.contracts.tasks import MemoryTaskInput
from graph_memory.embeddings import (
    DenseEncodingService,
    DenseTaskEncodingRequest,
    SentenceEncoder,
    load_sentence_transformer,
)
from graph_memory.retrieval.contracts import RankedNode


@dataclass(frozen=True)
class DenseConfig:
    model_name: str = "intfloat/e5-base-v2"
    query_prefix: str = "query: "
    passage_prefix: str = "passage: "
    batch_size: int = 64


class DenseTaskRetriever:
    method_name = "dense"

    def __init__(
        self,
        model_name: str = "intfloat/e5-base-v2",
        batch_size: int = 64,
        query_prefix: str = "query: ",
        passage_prefix: str = "passage: ",
        config: DenseConfig | None = None,
        encoder: SentenceEncoder | None = None,
    ) -> None:
        self.config = config or DenseConfig(
            model_name=model_name,
            query_prefix=query_prefix,
            passage_prefix=passage_prefix,
            batch_size=batch_size,
        )
        self.encoder = encoder if encoder is not None else self._load_encoder(self.config.model_name)
        self.encoding_service = DenseEncodingService(
            encoder=self.encoder,
            query_prefix=self.config.query_prefix,
            passage_prefix=self.config.passage_prefix,
            batch_size=self.config.batch_size,
        )

    def rank(self, task_input: MemoryTaskInput) -> list[RankedNode]:
        return self.rank_many([task_input])[0]

    def rank_many(self, task_inputs: list[MemoryTaskInput]) -> list[list[RankedNode]]:
        requests = [
            DenseTaskEncodingRequest(
                task_input=task_input,
                node_ids=("q", *(memory_item["id"] for memory_item in task_input["memory_items"])),
            )
            for task_input in task_inputs
        ]
        return [
            self._rank_from_embeddings(request.task_input, result.embeddings)
            for request, result in zip(requests, self.encoding_service.encode_tasks(requests), strict=True)
        ]

    @staticmethod
    def _rank_from_embeddings(
        task_input: MemoryTaskInput,
        embeddings: np.ndarray,
    ) -> list[RankedNode]:
        query_vector = embeddings[0]
        passage_matrix = embeddings[1:]
        scores = passage_matrix @ query_vector
        ranked_nodes = [
            RankedNode(node_id=memory_item["id"], score=float(score))
            for memory_item, score in zip(task_input["memory_items"], scores)
        ]
        return sorted(ranked_nodes, key=lambda ranked_node: (-ranked_node.score, ranked_node.node_id))

    @staticmethod
    def _load_encoder(model_name: str) -> SentenceEncoder:
        try:
            return cast(SentenceEncoder, cast(object, load_sentence_transformer(model_name)))
        except RuntimeError as error:
            raise RuntimeError(
                "sentence-transformers is required for dense retrieval unless a test encoder is provided."
            ) from error

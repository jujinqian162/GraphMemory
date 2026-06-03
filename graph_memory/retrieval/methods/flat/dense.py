from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np

from graph_memory.contracts.tasks import MemoryTaskInput
from graph_memory.retrieval.contracts import DenseEncoder, RankedNode


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
        encoder: DenseEncoder | None = None,
    ) -> None:
        self.config = config or DenseConfig(
            model_name=model_name,
            query_prefix=query_prefix,
            passage_prefix=passage_prefix,
            batch_size=batch_size,
        )
        self.encoder = encoder if encoder is not None else self._load_encoder(self.config.model_name)

    def rank(self, task_input: MemoryTaskInput) -> list[RankedNode]:
        query_text = self.config.query_prefix + task_input["query"]
        passages = [
            self.config.passage_prefix + f'{memory_item["source"]}. {memory_item["text"]}'
            for memory_item in task_input["memory_items"]
        ]
        query_embedding = self.encoder.encode(
            [query_text],
            batch_size=self.config.batch_size,
            normalize_embeddings=True,
        )
        passage_embeddings = self.encoder.encode(
            passages,
            batch_size=self.config.batch_size,
            normalize_embeddings=True,
        )
        query_vector = np.asarray(query_embedding, dtype=float)[0]
        passage_matrix = np.asarray(passage_embeddings, dtype=float)
        scores = passage_matrix @ query_vector
        ranked_nodes = [
            RankedNode(node_id=memory_item["id"], score=float(score))
            for memory_item, score in zip(task_input["memory_items"], scores)
        ]
        return sorted(ranked_nodes, key=lambda ranked_node: (-ranked_node.score, ranked_node.node_id))

    @staticmethod
    def _load_encoder(model_name: str) -> DenseEncoder:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise RuntimeError(
                "sentence-transformers is required for dense retrieval unless a test encoder is provided."
            ) from error
        return cast(DenseEncoder, cast(object, SentenceTransformer(model_name)))

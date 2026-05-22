from __future__ import annotations

from typing import Any

import numpy as np

from graph_memory.types import DenseConfig, MemoryTaskInput, RankedNode


class DenseTaskRetriever:
    method_name = "dense"

    def __init__(
        self,
        model_name: str = "intfloat/e5-base-v2",
        batch_size: int = 64,
        query_prefix: str = "query: ",
        passage_prefix: str = "passage: ",
        encoder: Any | None = None,
    ) -> None:
        self.config = DenseConfig(
            model_name=model_name,
            query_prefix=query_prefix,
            passage_prefix=passage_prefix,
            batch_size=batch_size,
        )
        self.encoder = encoder if encoder is not None else self._load_encoder(model_name)

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
    def _load_encoder(model_name: str) -> Any:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise RuntimeError(
                "sentence-transformers is required for dense retrieval unless a test encoder is provided."
            ) from error
        return SentenceTransformer(model_name)

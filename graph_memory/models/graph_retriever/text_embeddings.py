from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

import numpy as np
import torch
from torch import Tensor

from graph_memory.contracts.tasks import MemoryTaskInput
from graph_memory.models.graph_retriever.contracts import SentenceEncoder


@dataclass(frozen=True)
class DenseTextEmbeddingProvider:
    """
    Sentence-transformer text embedding provider used by trainable graph retrieval.
    可训练图检索使用的 sentence-transformer 文本 embedding provider。
    """

    model_name: str = "intfloat/e5-base-v2"
    query_prefix: str = "query: "
    passage_prefix: str = "passage: "
    batch_size: int = 64
    encoder: SentenceEncoder | None = None
    embedding_dim: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        if self.encoder is None:
            object.__setattr__(self, "encoder", self._load_encoder(self.model_name))
        object.__setattr__(self, "embedding_dim", self._detect_embedding_dim())

    def encode_task_nodes(self, task_input: MemoryTaskInput, node_ids: list[str]) -> Tensor:
        text_by_node_id = {"q": self.query_prefix + task_input["query"]}
        for memory_item in task_input["memory_items"]:
            text_by_node_id[memory_item["id"]] = self.passage_prefix + f'{memory_item["source"]}. {memory_item["text"]}'
        texts = [text_by_node_id[node_id] for node_id in node_ids]
        encoder = self.encoder
        if encoder is None:
            raise RuntimeError("DenseTextEmbeddingProvider encoder is not initialized.")
        embeddings = encoder.encode(texts, batch_size=self.batch_size, normalize_embeddings=True)
        return torch.tensor(np.asarray(embeddings, dtype=np.float32), dtype=torch.float32)

    def _detect_embedding_dim(self) -> int:
        encoder = self.encoder
        if encoder is None:
            raise RuntimeError("DenseTextEmbeddingProvider encoder is not initialized.")
        getter = getattr(encoder, "get_sentence_embedding_dimension", None)
        if callable(getter):
            value = getter()
            if isinstance(value, int) and value > 0:
                return value
        sample = encoder.encode(["dimension probe"], batch_size=1, normalize_embeddings=True)
        matrix = np.asarray(sample)
        if matrix.ndim != 2 or matrix.shape[1] <= 0:
            raise ValueError("Dense encoder returned an invalid embedding shape during dimension probing.")
        return int(matrix.shape[1])

    @staticmethod
    def _load_encoder(model_name: str) -> SentenceEncoder:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise RuntimeError(
                "sentence-transformers is required for trainable retrieval unless an embedding provider is injected."
            ) from error
        return cast(SentenceEncoder, cast(object, SentenceTransformer(model_name)))

from __future__ import annotations

from typing import Protocol

from torch import Tensor

from graph_memory.contracts.tasks import MemoryTaskInput


class TextEmbeddingProvider(Protocol):
    """
    Replaceable provider for frozen query and memory text embeddings.
    可替换的冻结 query 和 memory 文本 embedding 提供器。
    """

    @property
    def embedding_dim(self) -> int:
        ...

    def encode_task_nodes(self, task_input: MemoryTaskInput, node_ids: list[str]) -> Tensor:
        ...


class SentenceEncoder(Protocol):
    """
    Minimal sentence-transformer-like encoder protocol used by providers.
    provider 使用的最小 sentence-transformer-like encoder 协议。
    """

    def encode(self, texts: list[str], batch_size: int = 64, normalize_embeddings: bool = True) -> object:
        ...

    def get_sentence_embedding_dimension(self) -> int | None:
        ...

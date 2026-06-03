from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, cast

import numpy as np
import torch
from torch import Tensor

from graph_memory.contracts.tasks import MemoryTaskInput
from graph_memory.types import NodeFeatureConfig, Retriever, SeedSignal


class SeedSignalProvider(Protocol):
    """
    Replaceable provider for frozen seed retrieval signals.
    可替换的冻结初始检索信号提供器。

    Methods / 方法:
    - score_task: Return one SeedSignal for every memory node in the task.
      score_task：为 task 中每个 memory node 返回一个 SeedSignal。
    """

    def score_task(self, task_input: MemoryTaskInput) -> list[SeedSignal]:
        ...


class TextEmbeddingProvider(Protocol):
    """
    Replaceable provider for frozen query and memory text embeddings.
    可替换的冻结 query 和 memory 文本 embedding 提供器。

    Fields / 字段:
    - embedding_dim: Output embedding dimension.
      embedding_dim：输出 embedding 维度。

    Methods / 方法:
    - encode_task_nodes: Return embeddings in the same order as `node_ids`.
      encode_task_nodes：按 `node_ids` 的顺序返回 embedding。
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

    Methods / 方法:
    - encode: Return numeric embeddings for input texts.
      encode：为输入文本返回数值 embedding。
    - get_sentence_embedding_dimension: Optionally return embedding dimension.
      get_sentence_embedding_dimension：可选返回 embedding 维度。
    """

    def encode(self, texts: list[str], batch_size: int = 64, normalize_embeddings: bool = True) -> object:
        ...

    def get_sentence_embedding_dimension(self) -> int | None:
        ...


@dataclass(frozen=True)
class RetrieverSeedSignalProvider:
    """
    Seed signal provider backed by an existing flat retriever.
    基于现有 flat retriever 的 seed signal provider。

    Fields / 字段:
    - retriever: Frozen retriever that ranks every memory node.
      retriever：对所有 memory node 排序的冻结检索器。
    """

    retriever: Retriever

    def score_task(self, task_input: MemoryTaskInput) -> list[SeedSignal]:
        ranked_nodes = self.retriever.rank(task_input)
        expected_node_ids = {memory_item["id"] for memory_item in task_input["memory_items"]}
        observed_node_ids = {ranked_node.node_id for ranked_node in ranked_nodes}
        if observed_node_ids != expected_node_ids:
            missing = sorted(expected_node_ids - observed_node_ids)
            extra = sorted(observed_node_ids - expected_node_ids)
            raise ValueError(f"Seed retriever must return every memory node exactly once; missing={missing} extra={extra}.")

        sorted_nodes = sorted(ranked_nodes, key=lambda ranked_node: (-ranked_node.score, ranked_node.node_id))
        denominator = max(1, len(sorted_nodes) - 1)
        return [
            SeedSignal(
                node_id=ranked_node.node_id,
                score=float(ranked_node.score),
                rank=rank,
                rank_percentile=0.0 if len(sorted_nodes) == 1 else (rank - 1) / denominator,
            )
            for rank, ranked_node in enumerate(sorted_nodes, start=1)
        ]


@dataclass(frozen=True)
class DenseTextEmbeddingProvider:
    """
    Sentence-transformer text embedding provider used by trainable graph retrieval.
    可训练图检索使用的 sentence-transformer 文本 embedding provider。

    Fields / 字段:
    - model_name: Sentence-transformer model name or local path.
      model_name：sentence-transformer 模型名或本地路径。
    - query_prefix: Prefix applied to query text.
      query_prefix：query 文本前缀。
    - passage_prefix: Prefix applied to memory text.
      passage_prefix：memory 文本前缀。
    - batch_size: Encoder batch size.
      batch_size：encoder batch size。
    - encoder: Optional injected sentence-transformer-like encoder.
      encoder：可选注入的 sentence-transformer-like encoder。
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


@dataclass(frozen=True)
class NodeFeatureTensors:
    """
    Numeric node and scorer features for one tensorized task graph.
    单个张量化 task graph 的数值 node 与 scorer 特征。

    Fields / 字段:
    - node_ids: Node ids in tensor order.
      node_ids：张量顺序中的 node id。
    - node_features: Tensor `[num_nodes, node_feature_dim]` for input projection.
      node_features：用于 input projection 的 `[num_nodes, node_feature_dim]` tensor。
    - scorer_features: Tensor `[num_nodes, scorer_feature_dim]` for direct scorer input.
      scorer_features：用于 scorer 直接输入的 `[num_nodes, scorer_feature_dim]` tensor。
    - node_feature_names: Feature names matching node_features columns.
      node_feature_names：与 node_features 列对应的特征名。
    - scorer_feature_names: Feature names matching scorer_features columns.
      scorer_feature_names：与 scorer_features 列对应的特征名。
    """

    node_ids: list[str]
    node_features: Tensor
    scorer_features: Tensor
    node_feature_names: tuple[str, ...]
    scorer_feature_names: tuple[str, ...]


@dataclass(frozen=True)
class NodeFeatureBuilder:
    """
    Builds ordered numeric features from frozen seed signals.
    从冻结 seed signal 构造有序数值特征。

    Fields / 字段:
    - config: Ordered node and scorer feature config.
      config：有序 node 和 scorer 特征配置。
    """

    config: NodeFeatureConfig

    def build_node_features(self, *, node_ids: list[str], seed_signals: list[SeedSignal]) -> NodeFeatureTensors:
        signal_by_node_id = {signal.node_id: signal for signal in seed_signals}
        node_rows = [self._feature_row(node_id, self.config.node_feature_names, signal_by_node_id) for node_id in node_ids]
        scorer_rows = [
            self._feature_row(node_id, self.config.scorer_feature_names, signal_by_node_id) for node_id in node_ids
        ]
        return NodeFeatureTensors(
            node_ids=node_ids,
            node_features=torch.tensor(node_rows, dtype=torch.float32),
            scorer_features=torch.tensor(scorer_rows, dtype=torch.float32),
            node_feature_names=self.config.node_feature_names,
            scorer_feature_names=self.config.scorer_feature_names,
        )

    def _feature_row(
        self,
        node_id: str,
        feature_names: tuple[str, ...],
        signal_by_node_id: dict[str, SeedSignal],
    ) -> list[float]:
        row: list[float] = []
        signal = signal_by_node_id.get(node_id)
        for feature_name in feature_names:
            if feature_name == "seed_score":
                row.append(0.0 if signal is None else signal.score)
            elif feature_name == "seed_rank_percentile":
                row.append(1.0 if signal is None else signal.rank_percentile)
            elif feature_name == "is_question_node":
                row.append(1.0 if node_id == "q" else 0.0)
            else:
                raise ValueError(f"Unsupported node feature name: {feature_name}")
        return row

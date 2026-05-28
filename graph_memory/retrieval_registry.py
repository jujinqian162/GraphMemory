from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalMethodSpec:
    """
    Static metadata for one public retrieval method.
    一个公开检索方法的静态元数据。

    Fields / 字段:
    - name: Public method name written into ranked result artifacts.
      name：写入 ranked result artifact 的公开方法名。
    - requires_graphs: Whether this method requires `*_graphs.json`.
      requires_graphs：该方法是否需要 `*_graphs.json`。
    - requires_graph_config: Whether this method requires graph rerank config.
      requires_graph_config：该方法是否需要 graph rerank config。
    - requires_checkpoint: Whether this method requires a trainable model checkpoint.
      requires_checkpoint：该方法是否需要可训练模型 checkpoint。
    - requires_dense_encoder: Whether this method needs dense encoder runtime args.
      requires_dense_encoder：该方法是否需要 dense encoder 运行参数。
    - seed_method: Optional flat seed method used by this method.
      seed_method：该方法使用的可选 flat seed method。
    - builder_id: Local runtime builder selected by `graph_memory.retrieval`.
      builder_id：由 `graph_memory.retrieval` 选择的本地运行时 builder。
    """

    name: str
    requires_graphs: bool
    requires_graph_config: bool
    requires_checkpoint: bool
    requires_dense_encoder: bool
    seed_method: str | None
    builder_id: str


METHOD_REGISTRY: dict[str, RetrievalMethodSpec] = {
    "bm25": RetrievalMethodSpec(
        name="bm25",
        requires_graphs=False,
        requires_graph_config=False,
        requires_checkpoint=False,
        requires_dense_encoder=False,
        seed_method=None,
        builder_id="bm25",
    ),
    "dense": RetrievalMethodSpec(
        name="dense",
        requires_graphs=False,
        requires_graph_config=False,
        requires_checkpoint=False,
        requires_dense_encoder=True,
        seed_method=None,
        builder_id="dense",
    ),
    "bm25_graph_rerank": RetrievalMethodSpec(
        name="bm25_graph_rerank",
        requires_graphs=True,
        requires_graph_config=True,
        requires_checkpoint=False,
        requires_dense_encoder=False,
        seed_method="bm25",
        builder_id="graph_rerank",
    ),
    "dense_graph_rerank": RetrievalMethodSpec(
        name="dense_graph_rerank",
        requires_graphs=True,
        requires_graph_config=True,
        requires_checkpoint=False,
        requires_dense_encoder=True,
        seed_method="dense",
        builder_id="graph_rerank",
    ),
}


def get_supported_methods() -> tuple[str, ...]:
    return tuple(METHOD_REGISTRY)


def get_graph_rerank_methods() -> tuple[str, ...]:
    return tuple(method for method, spec in METHOD_REGISTRY.items() if spec.builder_id == "graph_rerank")


def get_methods_requiring_dense_encoder() -> tuple[str, ...]:
    return tuple(method for method, spec in METHOD_REGISTRY.items() if spec.requires_dense_encoder)


def get_method_spec(method: str) -> RetrievalMethodSpec:
    try:
        return METHOD_REGISTRY[method]
    except KeyError as error:
        raise ValueError(f"Unsupported retrieval method: {method}") from error

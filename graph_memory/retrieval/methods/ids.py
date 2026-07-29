from __future__ import annotations

from graph_memory.compat import StrEnum


class RetrievalMethodId(StrEnum):
    BM25 = "bm25"
    DENSE = "dense"
    DENSE_FT = "dense_ft"
    GRAPHRAG = "graphrag"
    DENSE_RGCN_GRAPH_RETRIEVER = "dense_rgcn_graph_retriever"
    DENSE_FT_RGCN_GRAPH_RETRIEVER = "dense_ft_rgcn_graph_retriever"
    EXECUTION_PROVENANCE_RETRIEVER = "execution_provenance_retriever"
    EXECUTION_PROVENANCE_RGCN_RETRIEVER = "execution_provenance_rgcn_retriever"


__all__ = ["RetrievalMethodId"]

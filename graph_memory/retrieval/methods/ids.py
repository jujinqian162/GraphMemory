from __future__ import annotations

from typing import Literal, TypeAlias

from graph_memory.compat import StrEnum

DenseCandidateView: TypeAlias = Literal["flat", "provenance_unit"]


class RetrievalMethodId(StrEnum):
    BM25 = "bm25"
    DENSE = "dense"
    DENSE_FT = "dense_ft"
    GRAPHRAG = "graphrag"
    PROVENANCE_PATH = "provenance_path"
    PROVENANCE_RGCN = "provenance_rgcn"
    DENSE_RGCN_GRAPH_RETRIEVER = "dense_rgcn_graph_retriever"
    DENSE_FT_RGCN_GRAPH_RETRIEVER = "dense_ft_rgcn_graph_retriever"


__all__ = ["DenseCandidateView", "RetrievalMethodId"]

from __future__ import annotations

from rank_bm25 import BM25Okapi

from graph_memory.retrieval.contracts import RankedNode
from graph_memory.retrieval.requests import TextRankingRequest
from graph_memory.text.tokens import content_tokens


class BM25TaskRetriever:
    method_name = "bm25"

    def rank(self, request: TextRankingRequest) -> list[RankedNode]:
        corpus_tokens = [content_tokens(candidate.text) for candidate in request.candidates]
        query_tokens = content_tokens(request.query_text)
        bm25 = BM25Okapi(corpus_tokens)
        scores = bm25.get_scores(query_tokens)
        ranked_nodes = [
            RankedNode(node_id=candidate.item_id, score=float(score))
            for candidate, score in zip(request.candidates, scores, strict=True)
        ]
        return sorted(ranked_nodes, key=lambda ranked_node: (-ranked_node.score, ranked_node.node_id))

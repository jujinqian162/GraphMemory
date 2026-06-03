from __future__ import annotations

from rank_bm25 import BM25Okapi

from graph_memory.contracts.tasks import MemoryTaskInput
from graph_memory.text.tokens import content_tokens
from graph_memory.types import RankedNode


class BM25TaskRetriever:
    method_name = "bm25"

    def rank(self, task_input: MemoryTaskInput) -> list[RankedNode]:
        corpus_tokens = [
            content_tokens(f'{memory_item["source"]}. {memory_item["text"]}')
            for memory_item in task_input["memory_items"]
        ]
        query_tokens = content_tokens(task_input["query"])
        bm25 = BM25Okapi(corpus_tokens)
        scores = bm25.get_scores(query_tokens)
        ranked_nodes = [
            RankedNode(node_id=memory_item["id"], score=float(score))
            for memory_item, score in zip(task_input["memory_items"], scores)
        ]
        return sorted(ranked_nodes, key=lambda ranked_node: (-ranked_node.score, ranked_node.node_id))

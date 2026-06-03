from __future__ import annotations

import hashlib
import json

from graph_memory.contracts.common import JsonObject
from graph_memory.contracts.observability import RankedNodeDebugRecord, ScoreDebugRecord
from graph_memory.contracts.ranking import RankedNodeRecord
from graph_memory.retrieval.methods.graph_rerank.config import ScoreBreakdown


def config_digest(config: JsonObject) -> str:
    encoded = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]


def build_score_debug_record(
    *,
    task_id: str,
    method: str,
    top_k: int,
    ranked_nodes: list[RankedNodeRecord],
    score_breakdown: ScoreBreakdown,
    split: str | None = None,
    config: JsonObject | None = None,
) -> ScoreDebugRecord:
    debug_ranked_nodes: list[RankedNodeDebugRecord] = []
    for ranked_node in ranked_nodes[:top_k]:
        debug_node: RankedNodeDebugRecord = {"node_id": ranked_node["node_id"], "score": ranked_node["score"]}
        node_id = ranked_node["node_id"]
        if node_id in score_breakdown:
            debug_node["score_components"] = score_breakdown[node_id]
        debug_ranked_nodes.append(debug_node)

    record: ScoreDebugRecord = {
        "debug_type": "score_breakdown",
        "task_id": task_id,
        "method": method,
        "top_k": top_k,
        "ranked_nodes": debug_ranked_nodes,
    }
    if split is not None:
        record["split"] = split
    if config is not None:
        record["config_digest"] = config_digest(config)
    return record

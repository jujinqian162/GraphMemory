"""Standalone EPGM provenance retrieval runner.

This deliberately bypasses the Hydra/Prefect/registry/evaluation experiment
pipeline. The EPGM benchmark is tiny, test-only, and NOT scored automatically:
we just want to see which memory evidence each method retrieves for each query
and eyeball the differences. So this script directly reuses the low-level
retrieval components and writes one detailed JSONL per method.

Reused as-is:
* ``BM25TaskRetriever``            (flat lexical baseline)
* ``DenseTaskRetriever``           (flat dense baseline, shared frozen encoder)
* ``GraphRAGMethod``               (entity-bridge baseline; fed adapter-derived
                                    title/entity metadata)
* ``EpgmRetriever``                (no-train "ours": one implementation, run
                                    per ``--epgm-variant`` preset)

Data contract (per the EPGM raw convention):
    data/epgm_provenance/raw/<task>.json          # shared session memory
    data/epgm_provenance/raw/<task>_queries.json  # queries + reference labels

Each ``<task>_queries.json`` query becomes one ranking task over the SAME
candidate pool + provenance graph. The candidate pool is the memory document's
explicit ``candidates`` plus the ``task`` and ``agent`` nodes (with
deterministically derived text), so questions about the user task or a specific
subagent are answerable by the flat baselines too. The final ``answer`` node and
any invalidated nodes are kept as retrievable memory.

Usage
-----
    uv run python scripts/run_epgm_provenance.py \
        --raw-dir data/epgm_provenance/raw \
        --methods bm25,dense,graphrag,epgm_retriever \
        --epgm-variant typed_beam \
        --top-k 10 \
        --device cpu \
        --output-dir runs/epgm_provenance/task2

    # equivalent module form
    uv run python -m scripts.run_epgm_provenance ...
"""

from __future__ import annotations

# ruff: noqa: E402 -- path bootstrap must run before package imports

import argparse
import json
import os
import re
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

# ``python scripts/run_epgm_provenance.py`` puts ``scripts/`` on sys.path[0],
# which hides the repository root and makes ``import graph_memory`` fail.
# Mirror experiment/run.py: drop the script directory and inject the repo root.
_SCRIPT_DIRECTORY = Path(__file__).resolve().parent
_REPOSITORY_ROOT = _SCRIPT_DIRECTORY.parent
sys.path = [
    entry for entry in sys.path if Path(entry).resolve() != _SCRIPT_DIRECTORY
]
sys.path.insert(0, str(_REPOSITORY_ROOT))

from graph_memory.graphs.provenance import (
    ExecutionProvenanceEdge,
    ExecutionProvenanceGraph,
    ExecutionProvenanceNode,
    FieldBinding,
    ProvenanceEdgeType,
    ProvenanceNodeType,
)
from graph_memory.retrieval.methods.epgm import (
    EPGM_VARIANTS,
    EpgmRetriever,
    EpgmRetrieverConfig,
    EpgmVariant,
)
from graph_memory.retrieval.methods.flat.bm25 import BM25TaskRetriever
from graph_memory.retrieval.methods.flat.dense import DenseConfig, DenseTaskRetriever
from graph_memory.retrieval.methods.graphrag import (
    GraphRAGConfig,
    GraphRAGMethod,
    build_graphrag_request,
)
from graph_memory.retrieval.requests import (
    ExecutionProvenanceRankingRequest,
    TextCandidate,
    TextRankingRequest,
)

REPOSITORY_ROOT = _REPOSITORY_ROOT
DEFAULT_RAW_DIR = REPOSITORY_ROOT / "data" / "epgm_provenance" / "raw"
DEFAULT_ENCODER = "models/intfloat-e5-base-v2"
ALL_METHODS = ("bm25", "dense", "graphrag", "epgm_retriever")

# Node types promoted into the retrievable candidate pool on top of the
# document's explicit ``candidates`` list.
EXTRA_CANDIDATE_NODE_TYPES = frozenset({"task", "agent"})

_PACKAGE_PIN = re.compile(r"\b([A-Za-z][A-Za-z0-9_.\-]+==[0-9][0-9A-Za-z.\-]*)")
_ADVISORY = re.compile(r"\b(CVE-\d{4}-\d+|GHSA-[0-9a-z]{4}-[0-9a-z]{4}-[0-9a-z]{4}|PYSEC-\d{4}-\d+)")
_SESSION_REF = re.compile(r"\b([0-9a-f]{6,8}/run-\d+)\b")


# --------------------------------------------------------------------------- #
# raw loading + discovery
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class EpgmTask:
    task_id: str
    memory_path: Path
    queries_path: Path
    memory: dict[str, Any]
    queries: list[dict[str, Any]]


def discover_tasks(raw_dir: Path) -> list[EpgmTask]:
    tasks: list[EpgmTask] = []
    query_files = sorted(raw_dir.glob("*_queries.json"))
    if not query_files:
        raise SystemExit(f"No <task>_queries.json found under {raw_dir}")
    for queries_path in query_files:
        stem = queries_path.name[: -len("_queries.json")]
        memory_path = raw_dir / f"{stem}.json"
        if not memory_path.is_file():
            raise SystemExit(
                f"Query file {queries_path.name} has no companion memory "
                f"{memory_path.name}"
            )
        memory = _read_json(memory_path)
        queries_doc = _read_json(queries_path)
        queries = queries_doc.get("queries")
        if not isinstance(queries, list) or not queries:
            raise SystemExit(f"{queries_path} has no queries[]")
        mem_task_id = str(memory.get("task_id") or stem)
        q_task_id = str(queries_doc.get("task_id") or stem)
        if mem_task_id != q_task_id:
            raise SystemExit(
                f"task_id mismatch: memory={mem_task_id!r} queries={q_task_id!r}"
            )
        seen: set[str] = set()
        for query in queries:
            qid = query.get("qid")
            if not isinstance(qid, str) or not qid:
                raise SystemExit(f"{queries_path}: query missing qid")
            if qid in seen:
                raise SystemExit(f"{queries_path}: duplicate qid={qid}")
            seen.add(qid)
        tasks.append(
            EpgmTask(
                task_id=mem_task_id,
                memory_path=memory_path,
                queries_path=queries_path,
                memory=memory,
                queries=queries,
            )
        )
    return tasks


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise SystemExit(f"{path} must contain a JSON object")
    return value


# --------------------------------------------------------------------------- #
# candidate pool + provenance graph assembly
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class MemoryView:
    """Shared, query-independent memory materialization for one task."""

    task_id: str
    nodes_by_id: dict[str, dict[str, Any]]
    candidates: tuple[TextCandidate, ...]
    graph_nodes: tuple[ExecutionProvenanceNode, ...]
    graph_edges: tuple[ExecutionProvenanceEdge, ...]


def build_memory_view(task: EpgmTask) -> MemoryView:
    memory = task.memory
    graph = memory["graph"]
    nodes_by_id = {node["node_id"]: node for node in graph["nodes"]}

    explicit_ids = [candidate["item_id"] for candidate in memory["candidates"]]
    explicit_set = set(explicit_ids)

    # candidate pool = explicit candidates + task/agent nodes (derived text)
    pool_ids: list[str] = list(explicit_ids)
    for node in graph["nodes"]:
        if (
            node["node_type"] in EXTRA_CANDIDATE_NODE_TYPES
            and node["node_id"] not in explicit_set
        ):
            pool_ids.append(node["node_id"])

    candidates = tuple(
        _text_candidate(node_id, nodes_by_id, memory) for node_id in pool_ids
    )

    graph_nodes = tuple(
        ExecutionProvenanceNode(
            node_id=node["node_id"],
            node_type=ProvenanceNodeType(node["node_type"]),
            text=node["text"],
            metadata=node.get("metadata") or {},
        )
        for node in graph["nodes"]
    )
    graph_edges = tuple(
        ExecutionProvenanceEdge(
            source=edge["source"],
            target=edge["target"],
            edge_type=ProvenanceEdgeType(edge["edge_type"]),
            binding=(
                FieldBinding(**edge["binding"])
                if edge.get("binding") is not None
                else None
            ),
            weight=float(edge.get("weight", 1.0)),
            metadata=edge.get("metadata") or {},
        )
        for edge in graph["edges"]
    )
    return MemoryView(
        task_id=task.task_id,
        nodes_by_id=nodes_by_id,
        candidates=candidates,
        graph_nodes=graph_nodes,
        graph_edges=graph_edges,
    )


def _text_candidate(
    node_id: str,
    nodes_by_id: Mapping[str, dict[str, Any]],
    memory: Mapping[str, Any],
) -> TextCandidate:
    node = nodes_by_id[node_id]
    node_type = node["node_type"]
    metadata = node.get("metadata") or {}
    if node_type == "task":
        text = _task_text(memory, node)
    elif node_type == "agent":
        text = _agent_text(memory, node)
    else:
        text = node["text"]
    title, source_ref, aliases = _entity_metadata(node_type, text, metadata)
    return TextCandidate(
        item_id=node_id,
        text=text,
        metadata={
            "node_type": node_type,
            "session_id": _as_scalar(metadata.get("session_id")),
            "lifecycle_state": _as_scalar(metadata.get("lifecycle_state")),
            "kind": _as_scalar(metadata.get("kind")),
            "verdict": _as_scalar(metadata.get("verdict")),
            "title": title,
            "source_ref": source_ref,
            "aliases": aliases,
        },
    )


def _task_text(memory: Mapping[str, Any], node: Mapping[str, Any]) -> str:
    prompt = ""
    task_block = memory.get("task")
    if isinstance(task_block, dict):
        prompt = str(task_block.get("prompt") or "")
    prompt = " ".join(prompt.split())
    return prompt or node["text"]


def _agent_text(memory: Mapping[str, Any], node: Mapping[str, Any]) -> str:
    node_id = node["node_id"]
    label = node["text"]
    role = str((node.get("metadata") or {}).get("role") or "")
    parts = [f"Agent: {label}."]
    if role:
        parts.append(f"Role: {role}.")

    subgraphs = memory.get("subgraphs")
    session_ref = _agent_session_ref(memory, node_id)
    child = None
    if isinstance(subgraphs, dict):
        root = subgraphs.get("root")
        if (
            isinstance(root, dict)
            and root.get("session_ref") == session_ref
        ):
            child = root
        for candidate in subgraphs.get("children", []) or []:
            if isinstance(candidate, dict) and candidate.get("session_ref") == session_ref:
                child = candidate
                break
    if isinstance(child, dict):
        user_task = " ".join(str(child.get("user_task_text") or "").split())
        if user_task:
            parts.append(f"Task: {user_task}")
        counts = child.get("counts")
        if isinstance(counts, dict):
            parts.append(
                "Produced "
                f"{counts.get('observations', counts.get('semantic_nodes', 0))} "
                "semantic nodes "
                f"(tool_outputs={counts.get('tool_outputs', 0)}, "
                f"semantic_nodes={counts.get('semantic_nodes', 0)})."
            )
        semantic_ids = [
            str(entry.get("node_id"))
            for entry in child.get("semantic_nodes", []) or []
            if isinstance(entry, dict) and entry.get("node_id")
        ]
        if semantic_ids:
            parts.append("Nodes: " + ", ".join(semantic_ids) + ".")
    return " ".join(parts).strip()


def _agent_session_ref(memory: Mapping[str, Any], agent_node_id: str) -> str | None:
    for agent in memory.get("agents", []) or []:
        if isinstance(agent, dict) and agent.get("agent_id") == agent_node_id:
            return agent.get("session_ref")
    return None


def _entity_metadata(
    node_type: str,
    text: str,
    metadata: Mapping[str, Any],
) -> tuple[str, str, str]:
    """Deterministically derive GraphRAG title/source_ref/aliases.

    GraphRAG builds a private entity graph from candidate ``title`` /
    ``source_ref`` / ``aliases`` metadata plus capitalized body mentions. EPGM
    candidates have none of that natively, so we extract package pins, package
    names, advisory IDs, session refs and the semantic node type. This never
    reads gold labels.
    """

    pins = _dedup(_PACKAGE_PIN.findall(text))
    advisories = _dedup(_ADVISORY.findall(text))
    session_refs = _dedup(_SESSION_REF.findall(text))
    packages = _dedup(pin.split("==")[0] for pin in pins)

    title = ""
    if pins:
        title = pins[0]
    elif packages:
        title = packages[0]
    elif session_refs:
        title = session_refs[0]
    else:
        title = node_type.replace("_", " ").title()

    source_ref = packages[0] if packages else title

    alias_terms: list[str] = []
    alias_terms.extend(packages)
    alias_terms.extend(pins)
    alias_terms.extend(advisories)
    alias_terms.extend(session_refs)
    aliases = "; ".join(_dedup(term for term in alias_terms if term and term != title))
    return title, source_ref, aliases


def _dedup(values) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _as_scalar(value: Any) -> str | int | float | bool | None:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def graph_for_query(view: MemoryView, qid: str) -> ExecutionProvenanceGraph:
    # request.task_id must equal graph.task_id; rebind the shared graph per qid.
    return ExecutionProvenanceGraph(
        task_id=qid,
        nodes=view.graph_nodes,
        edges=view.graph_edges,
    )


# --------------------------------------------------------------------------- #
# runners
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Methods:
    bm25: BM25TaskRetriever | None
    dense: DenseTaskRetriever | None
    graphrag: GraphRAGMethod | None
    epgm: EpgmRetriever | None


def build_methods(
    selected: Sequence[str],
    *,
    encoder_model: str,
    device: str | None,
    epgm_variant: EpgmVariant = "typed_beam",
) -> Methods:
    needs_dense = any(m in selected for m in ("dense", "graphrag", "epgm_retriever"))
    dense_ranker = None
    if needs_dense:
        dense_ranker = DenseTaskRetriever(
            config=DenseConfig(model_name=encoder_model, device=device),
            device=device,
        )
    return Methods(
        bm25=BM25TaskRetriever() if "bm25" in selected else None,
        dense=dense_ranker if "dense" in selected else None,
        graphrag=(
            GraphRAGMethod(dense_ranker=dense_ranker, config=GraphRAGConfig())
            if "graphrag" in selected and dense_ranker is not None
            else None
        ),
        epgm=(
            EpgmRetriever(
                dense_ranker=dense_ranker,
                config=EpgmRetrieverConfig.for_variant(epgm_variant),
            )
            if "epgm_retriever" in selected and dense_ranker is not None
            else None
        ),
    )


def _evidence_row(
    rank: int,
    node_id: str,
    view: MemoryView,
    *,
    score: float,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    node = view.nodes_by_id[node_id]
    metadata = node.get("metadata") or {}
    row = {
        "rank": rank,
        "node_id": node_id,
        "node_type": node["node_type"],
        "score": round(float(score), 6),
        "lifecycle_state": _as_scalar(metadata.get("lifecycle_state")),
        "session_id": _as_scalar(metadata.get("session_id")),
        "text": node["text"] if node["node_type"] not in EXTRA_CANDIDATE_NODE_TYPES
        else _candidate_text(view, node_id),
    }
    if extra:
        row.update(extra)
    return row


def _candidate_text(view: MemoryView, node_id: str) -> str:
    for candidate in view.candidates:
        if candidate.item_id == node_id:
            return candidate.text
    return view.nodes_by_id[node_id]["text"]


def run_query(
    method_name: str,
    methods: Methods,
    view: MemoryView,
    query: Mapping[str, Any],
    *,
    top_k: int,
) -> dict[str, Any]:
    qid = query["qid"]
    query_text = query["query"]
    text_request = TextRankingRequest(
        task_id=qid, query_text=query_text, candidates=view.candidates
    )
    started = time.perf_counter()
    ranked_evidence: list[dict[str, Any]] = []
    retrieved_edges: list[dict[str, Any]] = []
    paths: list[dict[str, Any]] = []
    seed_ids: list[str] = []

    if method_name == "bm25":
        ranked = methods.bm25.rank(text_request)
        ranked_evidence = [
            _evidence_row(i, node.node_id, view, score=node.score)
            for i, node in enumerate(ranked[:top_k], start=1)
        ]
    elif method_name == "dense":
        ranked = methods.dense.rank(text_request)
        ranked_evidence = [
            _evidence_row(i, node.node_id, view, score=node.score)
            for i, node in enumerate(ranked[:top_k], start=1)
        ]
    elif method_name == "graphrag":
        gr_request = build_graphrag_request(text_request, methods.graphrag.config)
        result = methods.graphrag.rank_task(gr_request, top_k=top_k)
        ranked_evidence = [
            _evidence_row(i, node.node_id, view, score=node.score)
            for i, node in enumerate(result.ranked_nodes[:top_k], start=1)
        ]
        retrieved_edges = [
            {
                "source": edge["source"],
                "target": edge["target"],
                "edge_type": edge["edge_type"],
                "weight": edge.get("weight"),
            }
            for edge in result.trace.retrieved_edges
        ]
    elif method_name == "epgm_retriever":
        prov_request = ExecutionProvenanceRankingRequest(
            task_id=qid,
            query_text=query_text,
            candidates=view.candidates,
            graph=graph_for_query(view, qid),
        )
        result = methods.epgm.rank(prov_request)
        seed_ids = list(result.seed_ids)
        top_nodes = result.ranked_nodes[:top_k]
        top_ids = {node.node_id for node in top_nodes}
        for i, node in enumerate(top_nodes, start=1):
            extra = {
                "dense_rank": node.dense_rank,
                "dense_score": round(node.dense_score, 6),
                "graph_score": round(node.graph_score, 6),
            }
            ranked_evidence.append(
                _evidence_row(i, node.node_id, view, score=node.score, extra=extra)
            )
        for node in top_nodes:
            if node.best_path is None:
                continue
            paths.append(
                {
                    "seed_id": node.best_path.seed_id,
                    "target_id": node.best_path.target_id,
                    "node_ids": list(node.best_path.node_ids),
                    "score": round(node.best_path.score, 6),
                    "steps": [
                        {
                            "source": step.source,
                            "target": step.target,
                            "edge_type": step.edge_type,
                            "direction": step.direction,
                        }
                        for step in node.best_path.steps
                    ],
                }
            )
            for step in node.best_path.steps:
                if step.source in top_ids and step.target in top_ids:
                    retrieved_edges.append(
                        {
                            "source": step.source,
                            "target": step.target,
                            "edge_type": step.edge_type,
                            "direction": step.direction,
                        }
                    )
    else:
        raise ValueError(f"unknown method {method_name!r}")

    latency_ms = (time.perf_counter() - started) * 1000.0
    return {
        "dataset": "epgm_provenance",
        "memory_task_id": view.task_id,
        "qid": qid,
        "method": method_name,
        "variant": (
            methods.epgm.config.variant
            if method_name == "epgm_retriever" and methods.epgm is not None
            else None
        ),
        "display_name": (
            f"EPGM (non-trained, {methods.epgm.config.variant})"
            if method_name == "epgm_retriever" and methods.epgm is not None
            else method_name
        ),
        "level": query.get("level"),
        "answer_type": query.get("answer_type"),
        "query": query_text,
        "reference_answer": query.get("answer"),
        "reference_gold_evidence_node_ids": query.get("gold_evidence_node_ids", []),
        "reference_gold_edges": query.get("gold_edges", []),
        "candidate_count": len(view.candidates),
        "top_k": top_k,
        "seed_ids": seed_ids,
        "ranked_evidence": ranked_evidence,
        "retrieved_edges": _dedup_edges(retrieved_edges),
        "paths": paths,
        "latency_ms": round(latency_ms, 3),
    }


def _dedup_edges(edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple] = set()
    result: list[dict[str, Any]] = []
    for edge in edges:
        key = (edge["source"], edge["target"], edge.get("edge_type"))
        if key in seen:
            continue
        seen.add(key)
        result.append(edge)
    return result


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--task", default=None, help="Only run this task_id.")
    parser.add_argument("--methods", default=",".join(ALL_METHODS))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--encoder-model", default=DEFAULT_ENCODER)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--epgm-variant",
        default="typed_beam",
        choices=EPGM_VARIANTS,
        help="Frozen preset of the single non-trained EPGM implementation.",
    )
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=4,
        help=(
            "Cap intra-op torch threads. This benchmark is tiny (150 candidates), "
            "so the default thread-per-core pool only adds per-thread allocator "
            "arenas: on a 16-core/16 GB host that inflated peak RSS from ~1.6 GB "
            "to enough to trigger a global OOM kill. Parallelism only; retrieval "
            "results are unaffected. Pass 0 to keep the torch default."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "runs" / "epgm_provenance",
    )
    return parser.parse_args(argv)


def _cap_torch_threads(threads: int) -> None:
    """Bound the intra-op thread pool before any tensor work allocates arenas."""

    if threads <= 0:
        return
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(variable, str(threads))
    try:
        import torch
    except ImportError:
        return
    torch.set_num_threads(threads)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    _cap_torch_threads(args.torch_threads)
    selected = [m.strip() for m in args.methods.split(",") if m.strip()]
    unknown = [m for m in selected if m not in ALL_METHODS]
    if unknown:
        raise SystemExit(f"unknown methods={unknown}; choose from {ALL_METHODS}")

    tasks = discover_tasks(args.raw_dir.expanduser())
    if args.task is not None:
        tasks = [task for task in tasks if task.task_id == args.task]
        if not tasks:
            raise SystemExit(f"task_id={args.task!r} not found under {args.raw_dir}")

    methods = build_methods(
        selected,
        encoder_model=args.encoder_model,
        device=args.device,
        epgm_variant=cast(EpgmVariant, args.epgm_variant),
    )

    for task in tasks:
        view = build_memory_view(task)
        out_dir = args.output_dir.expanduser() / task.task_id
        out_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"[{task.task_id}] {len(task.queries)} queries | "
            f"{len(view.candidates)} candidates | methods={selected}"
        )
        for method_name in selected:
            suffix = (
                f"_{args.epgm_variant}" if method_name == "epgm_retriever" else ""
            )
            out_path = out_dir / f"{method_name}{suffix}.jsonl"
            with out_path.open("w", encoding="utf-8") as handle:
                for query in task.queries:
                    row = run_query(
                        method_name, methods, view, query, top_k=args.top_k
                    )
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"  wrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

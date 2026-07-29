"""Convert a recorded EPGM Pi session directory into a single raw dataset JSON.

The EPGM provenance profile records an *execution-provenance graph* while a Pi
agent solves a User Task. That graph lives, scattered, across several JSONL
files:

* the root session JSONL (``<root>.jsonl``) holds the parent agent turns,
  runtime tool outputs (``details.epgmRef`` -> ``out_*``), declared semantic
  nodes (``details.epgm`` -> ``obs_* / claim_* / verify_* / decision_* /
  answer_*``) and subagent handoffs (``details.epgmHandoff`` -> ``handoff_*``);
* per-subagent child sessions live under
  ``<root_dir>/<agent_hash>/run-N/session.jsonl`` and each owns a *private*
  child subgraph with the same node kinds.

This script flattens all of that into ONE self-contained JSON document per Task
and writes it to ``data/raw/epgm_provenance/<task>.json`` inside GraphMemory.

The output is designed to be *directly* consumable by GraphMemory's execution
provenance domain:

* ``graph`` already uses the exact node/edge vocabulary of
  ``graph_memory.graphs.provenance.contracts.ExecutionProvenanceGraph``
  (node types ``task/agent/tool_call/tool_output/observation/evidence/claim/
  verification/decision/answer`` and edge types ``contains/invokes/returns/
  feeds/grounds/precedes/supports/verifies/depends_on/contradicts/invalidates/
  affects/supersedes``). Every typed transition is legal under that contract's
  validator, so the "ours" (execution_provenance_retriever) method can load it
  with no reshaping.
* ``candidates`` is a ready ``TextCandidate`` list (``item_id`` = a graph
  node id, ``text``, ``metadata``) usable by the flat "none" text baselines
  (bm25 / dense) as well as by the provenance retriever.
* ``labels`` carries the grounded Answer and its true upstream dependency
  edges, plus the invalidated / superseded nodes, so path-level metrics work.

The document is intentionally *lossless*: alongside the normalized graph it
preserves the raw per-node records, the parent<->child handoff boundary, and
each subagent's private subgraph, so later benchmark/query authoring can pick
any granularity without re-reading the Pi sessions.

Usage
-----
    uv run python scripts/data/convert_epgm_session.py \
        --session-dir ~/.pi/EPGM/sessions/--...--task2--/<TS>_<uuid> \
        --task-name supply_chain_pin_audit

If ``--session-dir`` points at a ``sessions/--...--<task>--`` directory that
holds exactly one ``<TS>_<uuid>`` run, that run is auto-selected.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1

# --- vocabulary mirrored from graph_memory.graphs.provenance.contracts -------

NODE_TYPES = {
    "task",
    "agent",
    "tool_call",
    "tool_output",
    "observation",
    "evidence",
    "claim",
    "verification",
    "decision",
    "answer",
}

# EPGM extension dependency.edgeType -> ExecutionProvenanceGraph edge_type.
# The extension emits a compact set; we expand it to the contract vocabulary.
_DEP_EDGE_MAP = {
    "derived_from": "grounds",  # tool_output/observation -> claim/answer/decision
    "supports": "supports",
    "verifies": "verifies",
    "contradicts": "contradicts",
    "invalidates": "invalidates",
    "depends_on": "depends_on",
}

# node kind (from ref prefix) -> contract node_type
_PREFIX_NODE_TYPE = {
    "task": "task",
    "agent": "agent",
    "call": "tool_call",
    "out": "tool_output",
    "handoff": "tool_output",
    "obs": "observation",
    "claim": "claim",
    "verify": "verification",
    "decision": "decision",
    "answer": "answer",
}


# --- raw session reading -----------------------------------------------------


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [
            str(item.get("text", ""))
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        return "\n".join(part for part in parts if part).strip()
    return ""


def _short(text: str, limit: int = 1200) -> str:
    text = " ".join(text.split())
    return text[:limit]


@dataclass
class ToolCall:
    call_id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class RawNode:
    node_id: str
    node_type: str
    text: str
    metadata: dict[str, Any]
    dependencies: list[dict[str, str]] = field(default_factory=list)


@dataclass
class SessionParse:
    """Parsed content of a single Pi session JSONL (parent or child)."""

    session_id: str
    session_file: str
    session_ref: str  # portable relative label, e.g. "root" or "12ce2982/run-0"
    cwd: str
    role: str  # "root" or agent name
    agent_label: str
    user_task_text: str
    tool_calls: dict[str, ToolCall]
    # runtime tool outputs: out_* ref -> RawNode(tool_output)
    tool_outputs: dict[str, RawNode]
    # declared/verified semantic nodes: ref -> RawNode
    semantic_nodes: dict[str, RawNode]
    # handoffs recorded in this session: handoff_* ref -> RawNode(tool_output)
    handoffs: dict[str, RawNode]
    # ordered list of (ref, kind) as they were produced, for precedes edges
    order: list[tuple[str, str]]


def _parse_session(path: Path, role: str) -> SessionParse:
    session_id = ""
    cwd = ""
    user_task_text = ""
    tool_calls: dict[str, ToolCall] = {}
    tool_outputs: dict[str, RawNode] = {}
    semantic_nodes: dict[str, RawNode] = {}
    handoffs: dict[str, RawNode] = {}
    order: list[tuple[str, str]] = []
    agent_label = role

    for entry in _iter_jsonl(path):
        etype = entry.get("type")
        if etype == "session":
            session_id = str(entry.get("id") or entry.get("sessionId") or "")
            cwd = str(entry.get("cwd") or "")
            continue
        if etype != "message":
            continue
        message = entry.get("message")
        if not isinstance(message, dict):
            continue
        mrole = message.get("role")

        if mrole == "user" and not user_task_text:
            user_task_text = _content_to_text(message.get("content"))
            continue

        if mrole == "assistant":
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "toolCall":
                    cid = str(item.get("id") or "")
                    if not cid:
                        continue
                    args = item.get("arguments")
                    tool_calls[cid] = ToolCall(
                        call_id=cid,
                        name=str(item.get("name") or ""),
                        arguments=args if isinstance(args, dict) else {},
                    )
            continue

        if mrole == "toolResult":
            details = message.get("details")
            details = details if isinstance(details, dict) else {}
            tool_name = str(message.get("toolName") or "")
            call_id = str(message.get("toolCallId") or "")
            content_text = _content_to_text(message.get("content"))

            # 1) declared/verified semantic node
            epgm = details.get("epgm")
            if isinstance(epgm, dict) and isinstance(epgm.get("nodeId"), str):
                node_id = str(epgm["nodeId"])
                node_type = str(epgm.get("nodeType") or "")
                deps = epgm.get("dependencies")
                dep_list = [
                    {
                        "source": str(d.get("source")),
                        "target": str(d.get("target")),
                        "edge_type": str(d.get("edgeType") or d.get("edge_type")),
                    }
                    for d in (deps if isinstance(deps, list) else [])
                    if isinstance(d, dict)
                ]
                attrs = epgm.get("attributes")
                metadata = dict(attrs) if isinstance(attrs, dict) else {}
                metadata.update(
                    {
                        "origin": epgm.get("origin"),
                        "producer_tool_call_id": epgm.get("producerToolCallId"),
                        "created_at": epgm.get("createdAt"),
                        "source_session_id": epgm.get("sessionId"),
                    }
                )
                semantic_nodes[node_id] = RawNode(
                    node_id=node_id,
                    node_type=node_type,
                    text=_short(str(epgm.get("text") or "")),
                    metadata=metadata,
                    dependencies=dep_list,
                )
                order.append((node_id, node_type))
                continue

            # 2) subagent handoff boundary
            handoff = details.get("epgmHandoff")
            if isinstance(handoff, dict) and isinstance(handoff.get("ref"), str):
                ref = str(handoff["ref"])
                handoffs[ref] = RawNode(
                    node_id=ref,
                    node_type="tool_output",
                    text=_short(str(handoff.get("text") or "")),
                    metadata={
                        "boundary": "handoff",
                        "status": handoff.get("status"),
                        "producer_tool_call_id": handoff.get("producerToolCallId"),
                        "created_at": handoff.get("createdAt"),
                        "num_child_sessions": len(
                            handoff.get("childSessionFiles") or []
                        ),
                        "source_session_id": handoff.get("sessionId"),
                    },
                )
                order.append((ref, "tool_output"))
                continue

            # 3) ordinary runtime tool output (assigned an out_* ref)
            epgm_ref = details.get("epgmRef")
            ref = None
            if isinstance(epgm_ref, dict) and isinstance(epgm_ref.get("ref"), str):
                ref = str(epgm_ref["ref"])
            if ref is None:
                # provenance tools (record_*) do not get an out_* ref; skip.
                continue
            call = tool_calls.get(call_id)
            input_params = sorted((call.arguments or {}).keys()) if call else []
            tool_outputs[ref] = RawNode(
                node_id=ref,
                node_type="tool_output",
                text=_short(content_text or f"{tool_name} output"),
                metadata={
                    "tool_name": tool_name,
                    "call_id": call_id,
                    "input_parameters": input_params,
                    "source_session_id": (
                        epgm_ref.get("sessionId")
                        if epgm_ref is not None
                        else None
                    ),
                },
            )
            order.append((ref, "tool_output"))
            continue

    return SessionParse(
        session_id=session_id,
        session_file=str(path),
        session_ref=role,
        cwd=cwd,
        role=role,
        agent_label=agent_label,
        user_task_text=user_task_text,
        tool_calls=tool_calls,
        tool_outputs=tool_outputs,
        semantic_nodes=semantic_nodes,
        handoffs=handoffs,
        order=order,
    )


# --- graph assembly ----------------------------------------------------------


@dataclass
class GraphBuilder:
    nodes: dict[str, dict[str, Any]] = field(default_factory=dict)
    edges: list[dict[str, Any]] = field(default_factory=list)
    _edge_keys: set[tuple[str, str, str]] = field(default_factory=set)

    def add_node(
        self,
        node_id: str,
        node_type: str,
        text: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if node_type not in NODE_TYPES:
            raise ValueError(f"Unknown node_type {node_type!r} for {node_id!r}")
        if node_id in self.nodes:
            # merge metadata, keep first non-empty text
            existing = self.nodes[node_id]
            if metadata:
                merged = dict(existing.get("metadata") or {})
                merged.update({k: v for k, v in metadata.items() if v is not None})
                existing["metadata"] = merged
            if not existing.get("text") and text:
                existing["text"] = text
            return
        self.nodes[node_id] = {
            "node_id": node_id,
            "node_type": node_type,
            "text": text,
            "metadata": {k: v for k, v in (metadata or {}).items() if v is not None},
        }

    def add_edge(
        self,
        source: str,
        target: str,
        edge_type: str,
        *,
        weight: float = 1.0,
        binding: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if source == target:
            return
        key = (source, target, edge_type)
        if key in self._edge_keys:
            return
        self._edge_keys.add(key)
        self.edges.append(
            {
                "source": source,
                "target": target,
                "edge_type": edge_type,
                "binding": dict(binding) if binding else None,
                "weight": weight,
                "metadata": dict(metadata) if metadata else {},
            }
        )

    def has_node(self, node_id: str) -> bool:
        return node_id in self.nodes


def _agent_node_id(session: SessionParse, index: int) -> str:
    base = session.session_id or f"idx{index}"
    return f"agent_{session.role}_{base[:12]}"


def _add_session_nodes(
    builder: GraphBuilder,
    session: SessionParse,
    *,
    task_node_id: str,
    agent_node_id: str,
    is_root: bool,
) -> None:
    """Add one session's agent, tool calls/outputs, semantic nodes, edges."""

    builder.add_node(
        agent_node_id,
        "agent",
        session.agent_label,
        {
            "role": session.role,
            "session_id": session.session_id,
            "is_root": is_root,
        },
    )
    builder.add_edge(task_node_id, agent_node_id, "contains")

    # tool_output nodes + their originating tool_call node + invokes/returns
    for ref, node in session.tool_outputs.items():
        call_id = str(node.metadata.get("call_id") or "")
        tool_name = str(node.metadata.get("tool_name") or "")
        input_params = node.metadata.get("input_parameters") or []
        call_node_id = f"call_{ref}"  # stable, unique per output
        builder.add_node(
            call_node_id,
            "tool_call",
            f"{tool_name} call",
            {
                "tool_name": tool_name,
                "call_id": call_id,
                "input_parameters": list(input_params),
                "session_id": session.session_id,
            },
        )
        builder.add_node(
            ref,
            "tool_output",
            node.text,
            {
                "tool_name": tool_name,
                "input_parameters": list(input_params),
                "session_id": session.session_id,
            },
        )
        builder.add_edge(task_node_id, call_node_id, "contains")
        builder.add_edge(agent_node_id, call_node_id, "invokes")
        builder.add_edge(call_node_id, ref, "returns")

    # handoff nodes are tool_output boundaries owned by this (parent) session
    for ref, node in session.handoffs.items():
        builder.add_node(
            ref,
            "tool_output",
            node.text,
            {
                "boundary": "handoff",
                "status": node.metadata.get("status"),
                "session_id": session.session_id,
            },
        )
        builder.add_edge(task_node_id, ref, "contains")

    # semantic nodes: observation / claim / verification / decision / answer
    for ref, node in session.semantic_nodes.items():
        metadata = {
            k: v for k, v in node.metadata.items() if k != "session_file"
        }
        metadata["session_id"] = session.session_id
        builder.add_node(
            ref,
            node.node_type,
            node.text,
            metadata,
        )
        builder.add_edge(task_node_id, ref, "contains")


def _add_semantic_edges(
    builder: GraphBuilder,
    session: SessionParse,
) -> list[dict[str, str]]:
    """Emit contract edges from EPGM dependency records.

    Returns the list of skipped dependencies (endpoints missing from graph)
    for diagnostics.
    """

    skipped: list[dict[str, str]] = []
    for ref, node in session.semantic_nodes.items():
        for dep in node.dependencies:
            source = dep["source"]
            target = dep["target"]
            raw_edge = dep["edge_type"]
            mapped = _DEP_EDGE_MAP.get(raw_edge, raw_edge)

            if not builder.has_node(source) or not builder.has_node(target):
                skipped.append(dep)
                continue

            # "derived_from" from a tool_output/observation to a claim/answer/
            # decision is a GROUNDS edge; to another node it stays supports-like.
            if raw_edge == "derived_from":
                tgt_type = builder.nodes[target]["node_type"]
                src_type = builder.nodes[source]["node_type"]
                if (
                    src_type in {"tool_output", "observation", "evidence"}
                    and tgt_type in {"claim", "answer", "decision"}
                ):
                    # evidence grounding a conclusion.
                    mapped = "grounds"
                else:
                    # Any other derivation (tool_output -> observation, evidence
                    # -> verification, etc.) is a plain execution dependency.
                    # GROUNDS only targets claim/answer/decision, and SUPPORTS
                    # only targets claim/answer/decision, so these must use
                    # DEPENDS_ON (execution-node -> execution-node), which the
                    # provenance retriever also traverses.
                    mapped = "depends_on"

            builder.add_edge(source, target, mapped, metadata={"epgm_edge": raw_edge})
    return skipped


def _mark_invalidated(builder: GraphBuilder) -> list[str]:
    """Set lifecycle_state=invalid on nodes targeted by invalidates edges."""

    invalid_targets = {
        edge["target"]
        for edge in builder.edges
        if edge["edge_type"] in {"invalidates", "supersedes"}
    }
    for node_id in invalid_targets:
        node = builder.nodes.get(node_id)
        if node is not None:
            node["metadata"]["lifecycle_state"] = "invalidated"
            node["metadata"]["valid"] = False
    return sorted(invalid_targets)


# --- candidate + label extraction -------------------------------------------


def _build_candidates(builder: GraphBuilder) -> list[dict[str, Any]]:
    """Candidates = every retrievable evidence-bearing node.

    We expose tool_output (incl. handoffs), observation, claim, verification,
    decision and answer nodes as candidates. Flat baselines ("none") rank by
    text; the provenance retriever ("ours") ranks by typed path.
    """

    candidate_types = {
        "tool_output",
        "observation",
        "evidence",
        "claim",
        "verification",
        "decision",
        "answer",
    }
    candidates: list[dict[str, Any]] = []
    for node_id, node in builder.nodes.items():
        if node["node_type"] not in candidate_types:
            continue
        if not node["text"]:
            continue
        candidates.append(
            {
                "item_id": node_id,
                "text": node["text"],
                "metadata": {
                    "node_type": node["node_type"],
                    "session_id": node["metadata"].get("session_id"),
                    "lifecycle_state": node["metadata"].get("lifecycle_state"),
                    "kind": node["metadata"].get("kind"),
                    "verdict": node["metadata"].get("verdict"),
                },
            }
        )
    candidates.sort(key=lambda c: c["item_id"])
    return candidates


def _build_labels(
    builder: GraphBuilder,
    root: SessionParse,
    invalidated: Sequence[str],
) -> dict[str, Any]:
    """Grounded answer + its true upstream dependency chain."""

    answer_ids = [
        node_id
        for node_id, node in builder.nodes.items()
        if node["node_type"] == "answer"
    ]
    # gold dependency edges = edges pointing into the answer and, transitively,
    # into the claims/decisions/verifications the answer depends on.
    gold_edges: list[list[str]] = []
    gold_nodes: set[str] = set(answer_ids)
    frontier = list(answer_ids)
    incoming: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for edge in builder.edges:
        incoming[edge["target"]].append((edge["source"], edge["edge_type"]))

    while frontier:
        current = frontier.pop()
        for source, edge_type in incoming.get(current, []):
            if edge_type in {"contains", "invokes", "precedes"}:
                continue
            gold_edges.append([source, current])
            if source not in gold_nodes:
                gold_nodes.add(source)
                frontier.append(source)

    # dedupe gold edges
    seen: set[tuple[str, str]] = set()
    deduped: list[list[str]] = []
    for edge in gold_edges:
        key = (edge[0], edge[1])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(edge)

    answer_text = ""
    if answer_ids:
        answer_text = builder.nodes[answer_ids[0]]["text"]

    return {
        "gold_answer": answer_text,
        "answer_node_ids": sorted(answer_ids),
        "gold_evidence_node_ids": sorted(gold_nodes),
        "gold_dependency_edges": deduped,
        "invalidated_node_ids": list(invalidated),
        "metadata": {
            "dataset": "epgm_provenance",
            "source_task_prompt": _short(root.user_task_text, 1800),
        },
    }


# --- subgraph packaging (lossless child preservation) ------------------------


def _package_subgraph(session: SessionParse) -> dict[str, Any]:
    def _dump(node: RawNode) -> dict[str, Any]:
        return {
            "node_id": node.node_id,
            "node_type": node.node_type,
            "text": node.text,
            "metadata": node.metadata,
            "dependencies": node.dependencies,
        }

    return {
        "role": session.role,
        "session_ref": session.session_ref,
        "session_id": session.session_id,
        "user_task_text": _short(session.user_task_text, 1800),
        "tool_outputs": [_dump(n) for n in session.tool_outputs.values()],
        "handoffs": [_dump(n) for n in session.handoffs.values()],
        "semantic_nodes": [_dump(n) for n in session.semantic_nodes.values()],
        "counts": {
            "tool_outputs": len(session.tool_outputs),
            "handoffs": len(session.handoffs),
            "semantic_nodes": len(session.semantic_nodes),
        },
    }


# --- session discovery -------------------------------------------------------


def _resolve_run(session_dir: Path) -> Path:
    """Resolve --session-dir into the concrete <TS>_<uuid> run directory."""

    root_jsonl = session_dir.with_suffix(".jsonl")
    if session_dir.is_dir() and root_jsonl.is_file():
        return session_dir

    # a sessions/--...--<task>-- directory containing exactly one run
    runs = sorted(
        child
        for child in session_dir.iterdir()
        if child.is_dir() and child.with_suffix(".jsonl").is_file()
    )
    if len(runs) == 1:
        return runs[0]
    if not runs:
        raise SystemExit(
            f"No <TS>_<uuid> run (with sibling .jsonl) found under {session_dir}"
        )
    raise SystemExit(
        "Multiple runs found; point --session-dir at one of:\n  "
        + "\n  ".join(str(run) for run in runs)
    )


def _discover_child_sessions(run_dir: Path) -> list[Path]:
    """Find all <agent_hash>/run-N/session.jsonl child sessions."""

    children: list[Path] = []
    for agent_dir in sorted(run_dir.iterdir()):
        if not agent_dir.is_dir():
            continue
        for run_sub in sorted(agent_dir.glob("run-*")):
            session_file = run_sub / "session.jsonl"
            if session_file.is_file():
                children.append(session_file)
    return children


def _child_role(run_dir: Path, child_file: Path) -> str:
    # <run_dir>/<agent_hash>/run-N/session.jsonl -> "<agent_hash>/run-N"
    rel = child_file.relative_to(run_dir)
    return str(rel.parent).replace("\\", "/")


# --- main --------------------------------------------------------------------


def convert(session_dir: Path, task_name: str, out_root: Path) -> Path:
    run_dir = _resolve_run(session_dir)
    root_jsonl = run_dir.with_suffix(".jsonl")

    root = _parse_session(root_jsonl, role="root")
    root.agent_label = "main orchestrator"

    child_files = _discover_child_sessions(run_dir)
    children: list[SessionParse] = []
    for child_file in child_files:
        role = _child_role(run_dir, child_file)
        child = _parse_session(child_file, role=role)
        child.agent_label = f"subagent {role}"
        children.append(child)

    builder = GraphBuilder()

    task_node_id = f"task_{task_name}"
    builder.add_node(
        task_node_id,
        "task",
        _short(root.user_task_text, 1800) or task_name,
        {"role": "user_task", "task_name": task_name},
    )

    root_agent_id = _agent_node_id(root, 0)
    _add_session_nodes(
        builder,
        root,
        task_node_id=task_node_id,
        agent_node_id=root_agent_id,
        is_root=True,
    )
    skipped = list(_add_semantic_edges(builder, root))

    for index, child in enumerate(children, start=1):
        child_agent_id = _agent_node_id(child, index)
        _add_session_nodes(
            builder,
            child,
            task_node_id=task_node_id,
            agent_node_id=child_agent_id,
            is_root=False,
        )
        skipped.extend(_add_semantic_edges(builder, child))

    # link parent agent -> child agent via depends_on is not a legal transition;
    # the audit trail parent<->child is preserved through handoff nodes and the
    # subgraphs section instead.

    invalidated = _mark_invalidated(builder)
    candidates = _build_candidates(builder)
    labels = _build_labels(builder, root, invalidated)

    node_type_counts = Counter(n["node_type"] for n in builder.nodes.values())
    edge_type_counts = Counter(e["edge_type"] for e in builder.edges)

    document = {
        "schema_version": SCHEMA_VERSION,
        "task_id": task_name,
        "source": {
            "profile": "EPGM",
            "root_session_id": root.session_id,
            "child_session_refs": [c.session_ref for c in children],
            "note": (
                "Graph, candidates, labels and subgraphs are fully self-contained. "
                "session ids/refs are audit metadata only; the retriever needs none "
                "of them. Absolute session file paths are intentionally omitted for "
                "portability."
            ),
        },
        "task": {
            "task_id": task_name,
            "prompt": root.user_task_text,
        },
        "agents": [
            {
                "agent_id": root_agent_id,
                "role": "root",
                "label": root.agent_label,
                "session_id": root.session_id,
                "session_ref": root.session_ref,
            },
            *[
                {
                    "agent_id": _agent_node_id(child, index),
                    "role": child.role,
                    "label": child.agent_label,
                    "session_id": child.session_id,
                    "session_ref": child.session_ref,
                }
                for index, child in enumerate(children, start=1)
            ],
        ],
        "graph": {
            "task_id": task_name,
            "nodes": list(builder.nodes.values()),
            "edges": builder.edges,
        },
        "candidates": candidates,
        "labels": labels,
        "subgraphs": {
            "root": _package_subgraph(root),
            "children": [_package_subgraph(child) for child in children],
        },
        "stats": {
            "node_type_counts": dict(node_type_counts),
            "edge_type_counts": dict(edge_type_counts),
            "num_nodes": len(builder.nodes),
            "num_edges": len(builder.edges),
            "num_candidates": len(candidates),
            "num_children": len(children),
            "num_invalidated": len(invalidated),
            "num_skipped_dependencies": len(skipped),
        },
        "diagnostics": {
            "skipped_dependencies": skipped,
        },
    }

    out_dir = out_root / "epgm_provenance"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{task_name}.json"
    out_path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return out_path


def _default_out_root() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "raw"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--session-dir",
        required=True,
        type=Path,
        help=(
            "Either the <TS>_<uuid> run dir, or the sessions/--...--<task>-- "
            "dir containing exactly one run."
        ),
    )
    parser.add_argument(
        "--task-name",
        required=True,
        help="Dataset task id / output filename stem (e.g. supply_chain_pin_audit).",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=_default_out_root(),
        help="Root of data/raw (default: GraphMemory/data/raw).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    session_dir = args.session_dir.expanduser()
    out_path = convert(session_dir, args.task_name, args.out_root.expanduser())
    document = json.loads(out_path.read_text(encoding="utf-8"))
    stats = document["stats"]
    print(f"Wrote {out_path}")
    print(f"  nodes={stats['num_nodes']} edges={stats['num_edges']} "
          f"candidates={stats['num_candidates']} children={stats['num_children']}")
    print(f"  node_types={stats['node_type_counts']}")
    print(f"  edge_types={stats['edge_type_counts']}")
    if stats["num_skipped_dependencies"]:
        print(f"  WARNING: skipped {stats['num_skipped_dependencies']} "
              f"dependencies with missing endpoints")
    return 0


if __name__ == "__main__":
    sys.exit(main())

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize_config_dir

from graph_memory.graphs.provenance import (
    ExecutionProvenanceEdge,
    ExecutionProvenanceGraph,
    ExecutionProvenanceNode,
    FieldBinding,
    ProvenanceEdgeType,
    ProvenanceNodeType,
)
from graph_memory.experiment.config import (
    resolve_experiment_config,
    validate_composed_config,
)
from graph_memory.experiment.layout import RunLayout
from graph_memory.experiment.planning import WorkflowPlanner
from graph_memory.registry import Registry
from graph_memory.registry.methods import RequiredArtifact
from graph_memory.registry.retrieval import (
    DenseEncoderSettings,
    ExecutionProvenanceBuildPayload,
    ExecutionProvenanceRetrievalSettings,
    GraphRAGBuildPayload,
    GraphRAGRetrievalSettings,
    RetrievalMethodId,
)
from graph_memory.retrieval.methods.execution_provenance import (
    ExecutionProvenanceConfig,
)
from graph_memory.retrieval.methods.execution_provenance.search import (
    DEPENDENCY_EDGE_TYPES,
    enumerate_provenance_paths,
    invalidated_node_ids,
    score_provenance_path,
)
from graph_memory.retrieval.methods.graphrag import GraphRAGConfig
from graph_memory.retrieval.methods.graphrag.index import build_graphrag_request
from graph_memory.retrieval.requests import (
    ExecutionProvenanceRankingRequest,
    GraphRAGRequest,
    TextCandidate,
    TextRankingRequest,
)
from graph_memory.validation import ContractValidationError, validate_ranked_results


ROOT = Path(__file__).resolve().parents[1]


class RecordingEncoder:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def encode(
        self,
        texts: Sequence[str],
        batch_size: int = 64,
        normalize_embeddings: bool = True,
    ) -> object:
        _ = batch_size, normalize_embeddings
        self.calls.append(list(texts))
        rows = []
        for index, _text in enumerate(texts):
            row = np.zeros(16, dtype=np.float32)
            row[index % row.shape[0]] = 1.0
            rows.append(row)
        return np.asarray(rows, dtype=np.float32)


def _graphrag_text_request() -> TextRankingRequest:
    return TextRankingRequest(
        task_id="graph-task",
        query_text="What did Ada design?",
        candidates=(
            TextCandidate(
                "c1",
                "Ada Lovelace designed the Analytical Engine.",
                {"title": "Ada Lovelace", "aliases": "Ada"},
            ),
            TextCandidate(
                "c2",
                "Ada and the Analytical Engine influenced early computing.",
                {"title": "Analytical Engine"},
            ),
        ),
    )


def _binding(name: str) -> FieldBinding:
    return FieldBinding(name, name, f"{name}-hash", "exact")


def _alternative_path_request() -> ExecutionProvenanceRankingRequest:
    nodes = (
        ExecutionProvenanceNode("seed", ProvenanceNodeType.TOOL_OUTPUT, "seed result"),
        ExecutionProvenanceNode("call-a", ProvenanceNodeType.TOOL_CALL, "use seed"),
        ExecutionProvenanceNode("out-a", ProvenanceNodeType.TOOL_OUTPUT, "bound result"),
        ExecutionProvenanceNode("target", ProvenanceNodeType.TOOL_CALL, "final call"),
    )
    edges = (
        ExecutionProvenanceEdge("seed", "target", ProvenanceEdgeType.DEPENDS_ON),
        ExecutionProvenanceEdge(
            "seed", "call-a", ProvenanceEdgeType.FEEDS, binding=_binding("seed")
        ),
        ExecutionProvenanceEdge("call-a", "out-a", ProvenanceEdgeType.RETURNS),
        ExecutionProvenanceEdge(
            "out-a", "target", ProvenanceEdgeType.FEEDS, binding=_binding("result")
        ),
    )
    graph = ExecutionProvenanceGraph("path-task", nodes, edges)
    return ExecutionProvenanceRankingRequest(
        "path-task",
        "find result",
        (
            TextCandidate("seed", "seed result", {}),
            TextCandidate("target", "final call", {}),
        ),
        graph,
    )


def test_graphrag_builder_assembles_explicit_alias_aware_graph() -> None:
    first = build_graphrag_request(_graphrag_text_request(), GraphRAGConfig())
    second = build_graphrag_request(_graphrag_text_request(), GraphRAGConfig())

    assert isinstance(first, GraphRAGRequest)
    assert first == second
    ada = next(
        entity
        for entity in first.knowledge_graph.entities
        if entity.normalized_name == "ada lovelace"
    )
    assert "ada" in ada.normalized_aliases
    assert ada.candidate_ids == ("c1", "c2")
    assert any(
        {relation.source_entity_id, relation.target_entity_id}
        == {
            ada.entity_id,
            next(
                entity.entity_id
                for entity in first.knowledge_graph.entities
                if entity.normalized_name == "analytical engine"
            ),
        }
        for relation in first.knowledge_graph.relations
    )


def test_new_dense_methods_preserve_query_and_passage_prefixes() -> None:
    graph_encoder = RecordingEncoder()
    graph_built = Registry.retrieval.build(
        GraphRAGRetrievalSettings(
            top_k=2,
            encoder=DenseEncoderSettings("recording", "Q::", "P::", 7),
        ),
        GraphRAGBuildPayload(
            text_requests=[_graphrag_text_request()], dense_encoder=graph_encoder
        ),
    )
    graph_built.method.rank_task(
        graph_built.execution_tasks[0].method_request, top_k=2
    )

    provenance_encoder = RecordingEncoder()
    provenance_request = _alternative_path_request()
    provenance_built = Registry.retrieval.build(
        ExecutionProvenanceRetrievalSettings(
            top_k=2,
            encoder=DenseEncoderSettings("recording", "Q::", "P::", 7),
        ),
        ExecutionProvenanceBuildPayload(
            provenance_requests=[provenance_request],
            dense_encoder=provenance_encoder,
        ),
    )
    provenance_built.method.rank_task(provenance_request, top_k=2)

    for encoder in (graph_encoder, provenance_encoder):
        assert encoder.calls
        assert all(call[0].startswith("Q::") for call in encoder.calls)
        assert all(
            text.startswith("P::") for call in encoder.calls for text in call[1:]
        )


def test_provenance_rejects_untyped_support_transition() -> None:
    with pytest.raises(ValueError, match="Invalid supports transition"):
        ExecutionProvenanceGraph(
            "invalid",
            (
                ExecutionProvenanceNode(
                    "answer", ProvenanceNodeType.ANSWER, "answer"
                ),
                ExecutionProvenanceNode(
                    "call", ProvenanceNodeType.TOOL_CALL, "call"
                ),
            ),
            (
                ExecutionProvenanceEdge(
                    "answer", "call", ProvenanceEdgeType.SUPPORTS
                ),
            ),
        )


def test_provenance_scores_complete_bound_path_before_short_weak_path() -> None:
    request = _alternative_path_request()
    config = ExecutionProvenanceConfig(
        max_hops=4,
        top_paths=5,
        max_path_expansions=32,
        semantic_weight=0.1,
        dependency_weight=0.35,
        binding_weight=0.4,
        grounding_weight=0.0,
        hop_penalty=0.01,
    )
    paths = enumerate_provenance_paths(request, ("seed",), config)
    target_paths = [path for path in paths if path.node_ids[-1] == "target"]

    assert {path.node_ids for path in target_paths} == {
        ("seed", "target"),
        ("seed", "call-a", "out-a", "target"),
    }
    scores = {
        path.node_ids: score_provenance_path(
            path,
            semantic_scores={"seed": 1.0, "target": 0.9},
            invalidated_node_ids=frozenset(),
            config=config,
        ).total
        for path in target_paths
    }
    assert scores[("seed", "call-a", "out-a", "target")] > scores[
        ("seed", "target")
    ]


def test_provenance_invalidation_uses_revision_edges_and_lifecycle_metadata() -> None:
    graph = ExecutionProvenanceGraph(
        "revision-task",
        (
            ExecutionProvenanceNode(
                "verification", ProvenanceNodeType.VERIFICATION, "new check"
            ),
            ExecutionProvenanceNode(
                "edge-invalidated", ProvenanceNodeType.CLAIM, "old claim"
            ),
            ExecutionProvenanceNode(
                "metadata-invalidated",
                ProvenanceNodeType.CLAIM,
                "obsolete claim",
                {"lifecycle_state": "superseded"},
            ),
        ),
        (
            ExecutionProvenanceEdge(
                "verification",
                "edge-invalidated",
                ProvenanceEdgeType.INVALIDATES,
            ),
        ),
    )
    request = ExecutionProvenanceRankingRequest(
        "revision-task",
        "current claim",
        (
            TextCandidate("edge-invalidated", "old claim", {}),
            TextCandidate("metadata-invalidated", "obsolete claim", {}),
        ),
        graph,
    )

    assert invalidated_node_ids(request) == frozenset(
        {"edge-invalidated", "metadata-invalidated"}
    )
    assert ProvenanceEdgeType.INVALIDATES not in DEPENDENCY_EDGE_TYPES
    assert ProvenanceEdgeType.PRECEDES not in DEPENDENCY_EDGE_TYPES


def test_ranked_result_validation_rejects_malformed_native_trace() -> None:
    request = TextRankingRequest(
        "trace-task", "query", (TextCandidate("c1", "candidate", {}),)
    )
    prediction = {
        "task_id": "trace-task",
        "method": "execution_provenance_retriever",
        "ranked_nodes": [{"node_id": "c1", "score": 1.0}],
        "retrieved_subgraph": {"nodes": ["c1"], "edges": []},
        "latency_ms": 1.0,
        "input_tokens": 1,
        "metadata": {
            "native_trace": {
                "trace_kind": "execution_provenance",
                "paths": [],
                "edges": [
                    {
                        "source": "c1",
                        "edge_type": "feeds",
                        "weight": float("nan"),
                    }
                ],
            }
        },
    }

    with pytest.raises(ContractValidationError, match="native trace"):
        validate_ranked_results([prediction], [request])


def test_registry_required_artifact_query_is_authoritative() -> None:
    assert Registry.methods.requires_artifact(
        RetrievalMethodId.DENSE_RGCN_GRAPH_RETRIEVER,
        RequiredArtifact.EVIDENCE_GRAPH,
    )
    assert not Registry.methods.requires_artifact(
        RetrievalMethodId.GRAPHRAG,
        RequiredArtifact.EVIDENCE_GRAPH,
    )


@pytest.mark.parametrize(
    ("method", "expected_graph_splits"),
    [
        (RetrievalMethodId.GRAPHRAG, set()),
        (
            RetrievalMethodId.DENSE_RGCN_GRAPH_RETRIEVER,
            {"train", "dev", "test"},
        ),
    ],
)
def test_planner_derives_evidence_graphs_from_registry_artifacts(
    method: RetrievalMethodId,
    expected_graph_splits: set[str],
) -> None:
    with initialize_config_dir(
        config_dir=str(ROOT / "configs"), version_base="1.3"
    ):
        composed = compose(
            config_name="config",
            overrides=[
                f"name=registry-plan-{method.value}",
                "profile=smoke",
                "device=cpu",
                f"methods=[{method.value}]",
            ],
        )
    config = resolve_experiment_config(
        validate_composed_config(composed), repository_root=ROOT
    )
    plan = WorkflowPlanner(
        config, RunLayout(ROOT, f"registry-plan-{method.value}")
    ).build(validate_external=False)

    assert {
        invocation.split
        for invocation in plan.invocations
        if invocation.stage == "evidence_graphs"
    } == expected_graph_splits
    evaluation = next(
        invocation
        for invocation in plan.invocations
        if invocation.stage == "evaluate" and invocation.method is method
    )
    assert any(item.role == "evidence_graphs" for item in evaluation.inputs) is (
        RequiredArtifact.EVIDENCE_GRAPH
        is Registry.methods.get(method).input_spec.required_artifact
    )

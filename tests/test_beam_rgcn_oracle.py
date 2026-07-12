from __future__ import annotations

import pytest

from graph_memory.evaluation.requests import EvidenceLabel
from graph_memory.models.graph_retriever.oracle import DynamicEvidenceOracle


def _label(
    *, gold: tuple[str, ...], edges: tuple[tuple[str, str], ...]
) -> EvidenceLabel:
    return EvidenceLabel(
        task_id="oracle-task",
        gold_answer="answer",
        gold_evidence_item_ids=gold,
        gold_dependency_edges=edges,
    )


def test_dynamic_oracle_supports_chains_branches_and_multiple_roots() -> None:
    oracle = DynamicEvidenceOracle(
        label=_label(
            gold=("a", "b", "c", "d"), edges=(("a", "c"), ("b", "c"), ("c", "d"))
        ),
        candidate_node_ids=("a", "b", "c", "d", "x"),
        max_steps=5,
    )

    root = oracle.state(())
    assert root.valid_next_ids == frozenset({"a", "b"})
    assert root.masked_future_ids == frozenset({"c", "d"})
    assert root.stop_target is False
    assert oracle.state(("a", "b")).valid_next_ids == frozenset({"c"})
    assert oracle.state(("a", "b", "c", "d")).stop_target is True


def test_dynamic_oracle_keeps_recovery_after_distractor_and_unordered_fallback() -> (
    None
):
    oracle = DynamicEvidenceOracle(
        label=_label(gold=("a", "b"), edges=()),
        candidate_node_ids=("a", "b", "x"),
        max_steps=5,
    )

    state = oracle.state(("x",))
    assert state.valid_next_ids == frozenset({"a", "b"})
    assert state.masked_future_ids == frozenset()
    assert state.oracle_reachable is True


@pytest.mark.parametrize(
    ("gold", "edges", "candidates", "match"),
    [
        (("missing",), (), ("a",), "oracle-task.*missing graph nodes"),
        (("a", "b"), (("a", "b"), ("a", "b")), ("a", "b"), "duplicate dependency"),
        (("a", "b"), (("a", "b"), ("b", "a")), ("a", "b"), "cyclic"),
        (("a", "b", "c"), (), ("a", "b", "c"), "cannot complete"),
    ],
)
def test_dynamic_oracle_rejects_invalid_supervision(
    gold: tuple[str, ...],
    edges: tuple[tuple[str, str], ...],
    candidates: tuple[str, ...],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        DynamicEvidenceOracle(
            label=_label(gold=gold, edges=edges),
            candidate_node_ids=candidates,
            max_steps=2,
        )

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "prefect_semantic_parity.json"


def test_frozen_parity_matrix_covers_every_cutover_branch() -> None:
    snapshot = cast(dict[str, Any], json.loads(FIXTURE.read_text(encoding="utf-8")))
    cases = cast(list[dict[str, Any]], snapshot["cases"])

    assert {case["case"] for case in cases} == {
        "stateless_evidence",
        "dense_ft",
        "evidence_rgcn",
        "dense_ft_seeded_rgcn",
        "execution_provenance",
        "execution_provenance_rgcn",
        "rgcn_ablation",
    }
    assert len(snapshot["accepted_non_semantic_differences"]) == 4


def test_frozen_payloads_retain_behavior_fields_and_exclude_runtime_bytes() -> None:
    snapshot = cast(dict[str, Any], json.loads(FIXTURE.read_text(encoding="utf-8")))
    excluded = set(cast(list[str], snapshot["excluded_fields"]))
    metric_keys = {"Recall@10", "MRR", "Path Recall@10", "Edge Recall@10"}

    for case in cast(list[dict[str, Any]], snapshot["cases"]):
        for generation in ("legacy", "prefect"):
            payload = cast(dict[str, Any], case[generation])
            assert set(payload) == {
                "source_run",
                "prepared",
                "ranking",
                "evaluation",
            }
            assert set(payload["prepared"]) == {"task_count", "task_id"}
            assert set(payload["ranking"]) == {"ranked_count", "top_node_id"}
            assert set(payload["evaluation"]) == metric_keys
            assert cast(int, payload["prepared"]["task_count"]) > 0
            assert cast(int, payload["ranking"]["ranked_count"]) > 0
            assert excluded.isdisjoint(_all_keys(payload))


def test_method_and_variant_identity_are_singular_per_parity_case() -> None:
    snapshot = cast(dict[str, Any], json.loads(FIXTURE.read_text(encoding="utf-8")))
    for case in cast(list[dict[str, Any]], snapshot["cases"]):
        assert isinstance(case["method"], str)
        assert case["variant"] is None or isinstance(case["variant"], str)
        assert not isinstance(case["variant"], list)


def _all_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        keys = {str(key) for key in value}
        for nested in value.values():
            keys.update(_all_keys(nested))
        return keys
    if isinstance(value, list):
        keys: set[str] = set()
        for nested in value:
            keys.update(_all_keys(nested))
        return keys
    return set()

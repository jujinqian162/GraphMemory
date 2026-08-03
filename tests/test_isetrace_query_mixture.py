from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

import pytest

from graph_memory.datasets.isetrace import (
    adapt_isetrace_record,
    allocate_trajectory_grouped_splits,
    parse_isetrace_record,
    prepare_isetrace_benchmark,
)
from graph_memory.datasets.isetrace.registration import ISETRACE_REVISION
from graph_memory.query_synthesis.provenance.authoring import (
    AuthoringGold,
    AuthoringQueryRecord,
)
from graph_memory.text.chunking import TokenChunkingConfig
from tests.isetrace_fixtures import isetrace_record


class CharacterOffsetTokenizer:
    is_fast = True

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        return_offsets_mapping: bool,
        truncation: bool,
    ) -> dict[str, object]:
        del add_special_tokens, return_offsets_mapping, truncation
        return {"offset_mapping": [(index, index + 1) for index in range(len(text))]}


_CHUNKING = TokenChunkingConfig(
    tokenizer_name="test-character-tokenizer",
    max_tokens=512,
    overlap_tokens=64,
)


def _query(trajectory, *, query_id: str) -> AuthoringQueryRecord:
    call = trajectory.tool_calls[0]
    output = next(
        item for item in trajectory.tool_outputs if item.call_id == call.call_id
    )
    sections = [
        *(
            f"[I{index} | user_intent]\n{intent.text}"
            for index, intent in enumerate(trajectory.intents, start=1)
        ),
        f"[A1 | tool_call | {call.tool_name}]\n{call.raw_arguments}",
        f"[E1 | tool_output | {output.tool_name}]\n{output.content}",
    ]
    return AuthoringQueryRecord(
        id=query_id,
        text="\n\n".join(sections),
        query=f"Which recorded output answers {query_id}?",
        gold=(AuthoringGold(source="E1", quote=output.content),),
    )


def _raw_trajectory(index: int) -> dict[str, object]:
    raw = isetrace_record()
    raw["session_id"] = f"traj_fixture_{index}"
    intents = cast(list[dict[str, object]], raw["source_intents"])
    for intent in intents:
        intent["natural_language_intent"] = (
            f"{intent['natural_language_intent']} trajectory {index}"
        )
    return raw


def test_grouped_split_targets_weights_and_never_splits_one_trajectory() -> None:
    query_trajectories = {
        f"query:{trajectory_index}:{query_index}": f"trajectory:{trajectory_index}"
        for trajectory_index in range(3000)
        for query_index in range(2)
    }

    assignments, targets = allocate_trajectory_grouped_splits(
        query_trajectories,
        split_weights={"train": 3200, "dev": 800, "test": 2000},
        split_seed=41,
    )

    assert targets == {"train": 3200, "dev": 800, "test": 2000}
    assert {
        split: sum(value == split for value in assignments.values())
        for split in ("train", "dev", "test")
    } == targets
    for trajectory_index in range(3000):
        assert len(
            {
                assignments[f"query:{trajectory_index}:{query_index}"]
                for query_index in range(2)
            }
        ) == 1


def test_grouped_split_depends_only_on_split_seed() -> None:
    query_trajectories = {
        f"query:{index}": f"trajectory:{index}" for index in range(30)
    }

    seed_13_a, _ = allocate_trajectory_grouped_splits(
        query_trajectories,
        split_weights={"train": 10, "dev": 10, "test": 10},
        split_seed=13,
    )
    seed_13_b, _ = allocate_trajectory_grouped_splits(
        query_trajectories,
        split_weights={"train": 10, "dev": 10, "test": 10},
        split_seed=13,
    )
    seed_17, _ = allocate_trajectory_grouped_splits(
        query_trajectories,
        split_weights={"train": 10, "dev": 10, "test": 10},
        split_seed=17,
    )

    assert seed_13_a == seed_13_b
    assert seed_13_a != seed_17


def test_preparation_resolves_then_materializes_disjoint_grouped_splits(
    tmp_path: Path,
) -> None:
    raw_trajectories = [_raw_trajectory(index) for index in range(3)]
    trajectories = [
        adapt_isetrace_record(
            parse_isetrace_record(raw), source_revision="fixture-revision"
        )
        for raw in raw_trajectories
    ]
    trajectory_path = tmp_path / "trajectories.jsonl"
    trajectory_path.write_text(
        "".join(json.dumps(raw) + "\n" for raw in raw_trajectories),
        encoding="utf-8",
    )
    queries = [
        _query(trajectory, query_id=f"query:{trajectory_index}:{query_index}")
        for trajectory_index, trajectory in enumerate(trajectories)
        for query_index in range(2)
    ]
    query_path = tmp_path / "queries.jsonl"
    query_path.write_text(
        "".join(query.model_dump_json() + "\n" for query in queries),
        encoding="utf-8",
    )

    task_ids_by_split: dict[str, set[str]] = {}
    graph_ids_by_split: dict[str, set[str]] = {}
    for split in ("train", "dev", "test"):
        benchmark, summary = prepare_isetrace_benchmark(
            query_path,
            trajectory_path,
            source_revision="fixture-revision",
            count=None,
            seed=13,
            offset=0,
            strict=True,
            split=split,
            split_weights={"train": 2, "dev": 2, "test": 2},
            query_counts={"natural": 2, "template": 0},
            chunking=_CHUNKING,
            tokenizer=CharacterOffsetTokenizer(),
        )
        task_ids_by_split[split] = {
            ranking.task_id for ranking in benchmark.rankings
        }
        graph_ids_by_split[split] = {
            ranking.graph_id for ranking in benchmark.rankings
        }
        assert summary[f"natural_queries_capacity_{split}"] == 2
        assert summary[f"natural_queries_available_{split}"] == 2
        assert all(
            metadata.query_origin == "natural"
            for metadata in benchmark.query_metadata
        )

    assert set.union(*task_ids_by_split.values()) == {
        query.id for query in queries
    }
    assert not (graph_ids_by_split["train"] & graph_ids_by_split["dev"])
    assert not (graph_ids_by_split["train"] & graph_ids_by_split["test"])
    assert not (graph_ids_by_split["dev"] & graph_ids_by_split["test"])


def test_mixed_preparation_retains_natural_queries_and_keeps_test_natural_only(
    tmp_path: Path,
) -> None:
    raw_trajectories = [_raw_trajectory(index) for index in range(3)]
    trajectories = [
        adapt_isetrace_record(
            parse_isetrace_record(raw), source_revision="fixture-revision"
        )
        for raw in raw_trajectories
    ]
    expected_fingerprints: dict[str, str | None] = {
        trajectory.trajectory_id: None for trajectory in trajectories
    }
    trajectory_path = tmp_path / "trajectories.jsonl"
    trajectory_path.write_text(
        "".join(json.dumps(raw) + "\n" for raw in raw_trajectories),
        encoding="utf-8",
    )
    queries = [
        _query(trajectory, query_id=f"query:{trajectory_index}:{query_index}")
        for trajectory_index, trajectory in enumerate(trajectories)
        for query_index in range(2)
    ]
    query_path = tmp_path / "queries.jsonl"
    query_path.write_text(
        "".join(query.model_dump_json() + "\n" for query in queries),
        encoding="utf-8",
    )

    graphs_by_split: dict[str, set[str]] = {}
    for split in ("train", "dev", "test"):
        benchmark, summary = prepare_isetrace_benchmark(
            query_path,
            trajectory_path,
            source_revision="fixture-revision",
            count=None,
            seed=13,
            offset=0,
            strict=True,
            split=split,
            split_weights={"train": 2, "dev": 2, "test": 2},
            query_counts={
                "natural": 2,
                "template": 2 if split in {"train", "dev"} else 0,
            },
            chunking=_CHUNKING,
            tokenizer=CharacterOffsetTokenizer(),
        )
        origins = [metadata.query_origin for metadata in benchmark.query_metadata]
        assert origins.count("natural") == 2
        expected_templates = 0 if split == "test" else 2
        assert origins.count("template") == expected_templates
        assert len(benchmark.template_supervision) == expected_templates
        assert summary["natural_queries_selected"] == 2
        assert summary["template_queries_selected"] == expected_templates
        graphs_by_split[split] = {
            graph.graph_id for graph in benchmark.provenance_graphs
        }
        assert all(
            record.graph_id in graphs_by_split[split]
            for record in benchmark.template_supervision
        )
        for graph in benchmark.provenance_graphs:
            observed = expected_fingerprints[graph.graph_id]
            if observed is None:
                expected_fingerprints[graph.graph_id] = graph.fingerprint()
            else:
                assert observed == graph.fingerprint()

    assert not (graphs_by_split["train"] & graphs_by_split["dev"])
    assert not (graphs_by_split["train"] & graphs_by_split["test"])
    assert not (graphs_by_split["dev"] & graphs_by_split["test"])


def test_template_only_preparation_uses_exact_counts_from_frozen_split(
    tmp_path: Path,
) -> None:
    raw_trajectories = [_raw_trajectory(index) for index in range(3)]
    trajectories = [
        adapt_isetrace_record(
            parse_isetrace_record(raw), source_revision="fixture-revision"
        )
        for raw in raw_trajectories
    ]
    trajectory_path = tmp_path / "trajectories.jsonl"
    trajectory_path.write_text(
        "".join(json.dumps(raw) + "\n" for raw in raw_trajectories),
        encoding="utf-8",
    )
    query_path = tmp_path / "queries.jsonl"
    query_path.write_text(
        "".join(
            _query(
                trajectory,
                query_id=f"query:{trajectory_index}:{query_index}",
            ).model_dump_json()
            + "\n"
            for trajectory_index, trajectory in enumerate(trajectories)
            for query_index in range(2)
        ),
        encoding="utf-8",
    )

    benchmark, summary = prepare_isetrace_benchmark(
        query_path,
        trajectory_path,
        source_revision="fixture-revision",
        count=None,
        seed=13,
        offset=0,
        strict=True,
        split="train",
        split_weights={"train": 2, "dev": 2, "test": 2},
        query_counts={"natural": 0, "template": 2},
        chunking=_CHUNKING,
        tokenizer=CharacterOffsetTokenizer(),
    )

    assert len(benchmark.rankings) == 2
    assert len(benchmark.template_supervision) == 2
    assert {item.query_origin for item in benchmark.query_metadata} == {"template"}
    assert summary["natural_queries_requested"] == 0
    assert summary["natural_queries_selected"] == 0
    assert summary["template_queries_requested"] == 2
    assert summary["template_queries_selected"] == 2


def test_mixed_preparation_fails_on_insufficient_template_pool(
    tmp_path: Path,
) -> None:
    raw = _raw_trajectory(1)
    trajectory = adapt_isetrace_record(
        parse_isetrace_record(raw), source_revision="fixture-revision"
    )
    trajectory_path = tmp_path / "trajectories.jsonl"
    trajectory_path.write_text(json.dumps(raw) + "\n", encoding="utf-8")
    query_path = tmp_path / "queries.jsonl"
    query_path.write_text(
        _query(trajectory, query_id="query:valid").model_dump_json() + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=r"insufficient ISETrace template pool: requested=1000 available=\d+",
    ):
        prepare_isetrace_benchmark(
            query_path,
            trajectory_path,
            source_revision="fixture-revision",
            count=None,
            seed=13,
            offset=0,
            strict=True,
            split="train",
            split_weights={"train": 1, "dev": 0, "test": 0},
            query_counts={"natural": 1, "template": 1000},
            chunking=_CHUNKING,
            tokenizer=CharacterOffsetTokenizer(),
        )


def test_preparation_drops_and_reports_malformed_and_unresolvable_records(
    tmp_path: Path,
) -> None:
    raw = _raw_trajectory(1)
    trajectory = adapt_isetrace_record(
        parse_isetrace_record(raw), source_revision="fixture-revision"
    )
    trajectory_path = tmp_path / "trajectories.jsonl"
    trajectory_path.write_text(json.dumps(raw) + "\n", encoding="utf-8")
    valid = _query(trajectory, query_id="query:valid")
    unmatched = valid.model_copy(
        update={
            "id": "query:unmatched",
            "text": valid.text.replace("trajectory 1", "trajectory missing"),
        }
    )
    query_path = tmp_path / "queries.jsonl"
    query_path.write_text(
        valid.model_dump_json()
        + "\n"
        + json.dumps({"malformed": True})
        + "\n"
        + unmatched.model_dump_json()
        + "\n",
        encoding="utf-8",
    )

    benchmark, summary = prepare_isetrace_benchmark(
        query_path,
        trajectory_path,
        source_revision="fixture-revision",
        count=None,
        seed=13,
        offset=0,
        strict=False,
        split="train",
        split_weights={"train": 1, "dev": 0, "test": 0},
        query_counts={"natural": 1, "template": 0},
        chunking=_CHUNKING,
        tokenizer=CharacterOffsetTokenizer(),
    )

    assert [ranking.task_id for ranking in benchmark.rankings] == ["query:valid"]
    assert summary["queries_seen"] == 3
    assert summary["queries_parsed"] == 2
    assert summary["queries_resolved"] == 1
    assert summary["queries_dropped"] == 2
    assert summary["invalid_queries"] == 1
    assert summary["queries_unmatched"] == 1
    assert summary["natural_queries_capacity_train"] == 1


def test_authoring_revision_conflict_fails_before_query_parsing(
    tmp_path: Path,
) -> None:
    trajectory_path = tmp_path / "trajectories.jsonl"
    trajectory_bytes = b"not parsed because identity validation runs first\n"
    trajectory_path.write_bytes(trajectory_bytes)
    query_path = tmp_path / "queries.jsonl"
    query_path.write_text("not valid json\n", encoding="utf-8")
    manifest_path = query_path.with_suffix(query_path.suffix + ".run.json")
    manifest_path.write_text(
        json.dumps(
            {
                "source_revision": "wrong-revision",
                "source_files": [
                    {
                        "path": str(trajectory_path),
                        "bytes": len(trajectory_bytes),
                        "sha256": hashlib.sha256(trajectory_bytes).hexdigest(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="registered revision"):
        prepare_isetrace_benchmark(
            query_path,
            trajectory_path,
            source_revision=ISETRACE_REVISION,
            count=None,
            seed=13,
            offset=0,
            strict=True,
            split="test",
            split_weights={"train": 0, "dev": 0, "test": 1},
            query_counts={"natural": 1, "template": 0},
            chunking=_CHUNKING,
            tokenizer=CharacterOffsetTokenizer(),
        )


def test_authoring_source_digest_conflict_fails_before_graph_construction(
    tmp_path: Path,
) -> None:
    trajectory_path = tmp_path / "trajectories.jsonl"
    trajectory_bytes = b"{}\n"
    trajectory_path.write_bytes(trajectory_bytes)
    query_path = tmp_path / "queries.jsonl"
    query_path.write_text("", encoding="utf-8")
    manifest_path = query_path.with_suffix(query_path.suffix + ".run.json")
    manifest_path.write_text(
        json.dumps(
            {
                "source_revision": ISETRACE_REVISION,
                "source_files": [
                    {
                        "path": str(trajectory_path),
                        "bytes": len(trajectory_bytes),
                        "sha256": "0" * 64,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="source digest conflicts"):
        prepare_isetrace_benchmark(
            query_path,
            trajectory_path,
            source_revision=ISETRACE_REVISION,
            count=None,
            seed=13,
            offset=0,
            strict=True,
            split="test",
            split_weights={"train": 0, "dev": 0, "test": 1},
            query_counts={"natural": 1, "template": 0},
            chunking=_CHUNKING,
            tokenizer=CharacterOffsetTokenizer(),
        )

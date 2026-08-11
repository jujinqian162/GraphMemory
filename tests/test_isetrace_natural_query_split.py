from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

import pytest

from graph_memory.datasets.isetrace import (
    adapt_isetrace_record,
    allocate_trajectory_splits,
    parse_isetrace_record,
    prepare_isetrace_benchmark,
)
from graph_memory.datasets.isetrace.registration import ISETRACE_REVISION
from graph_memory.query_synthesis.provenance.authoring import (
    AuthoringGold,
    AuthoringQueryMetadataRecord,
    AuthoringQueryRecord,
    authoring_metadata_path,
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


def _trajectory_splits(
    *,
    natural: tuple[int, int, int] = (1, 1, 1),
) -> dict[str, int]:
    return {
        split: natural[index] for index, split in enumerate(("train", "dev", "test"))
    }


def test_trajectory_split_counts_are_disjoint() -> None:
    trajectory_ids = {f"trajectory:{index}" for index in range(6)}
    splits = allocate_trajectory_splits(
        trajectory_ids,
        split_counts=_trajectory_splits(natural=(2, 1, 1)),
        split_seed=41,
    )

    assert len(splits["train"]) == 2
    assert len(splits["dev"]) == 1
    assert len(splits["test"]) == 1
    assert not (splits["train"] & splits["dev"])
    assert not (splits["train"] & splits["test"])
    assert not (splits["dev"] & splits["test"])


def test_trajectory_split_depends_only_on_split_seed() -> None:
    trajectory_ids = {f"trajectory:{index}" for index in range(30)}
    counts = _trajectory_splits(natural=(10, 10, 10))

    seed_13_a = allocate_trajectory_splits(
        trajectory_ids, split_counts=counts, split_seed=13
    )
    seed_13_b = allocate_trajectory_splits(
        trajectory_ids, split_counts=counts, split_seed=13
    )
    seed_17 = allocate_trajectory_splits(
        trajectory_ids, split_counts=counts, split_seed=17
    )

    assert seed_13_a == seed_13_b
    assert seed_13_a != seed_17


def test_smaller_train_count_keeps_dev_and_test_fixed_and_train_nested() -> None:
    trajectory_ids = [f"trajectory:{index}" for index in range(40)]
    smaller = allocate_trajectory_splits(
        trajectory_ids,
        split_counts=_trajectory_splits(natural=(10, 6, 5)),
        split_seed=13,
    )
    larger = allocate_trajectory_splits(
        trajectory_ids,
        split_counts=_trajectory_splits(natural=(20, 6, 5)),
        split_seed=13,
    )

    assert smaller["test"] == larger["test"] == frozenset(trajectory_ids[:5])
    assert smaller["dev"] == larger["dev"]
    assert smaller["train"] < larger["train"]
    assert not (larger["train"] & larger["dev"])
    assert not (larger["train"] & larger["test"])
    assert not (larger["dev"] & larger["test"])


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
            trajectory_splits=_trajectory_splits(),
            chunking=_CHUNKING,
            tokenizer=CharacterOffsetTokenizer(),
        )
        task_ids_by_split[split] = {ranking.task_id for ranking in benchmark.rankings}
        graph_ids_by_split[split] = {ranking.graph_id for ranking in benchmark.rankings}
        assert summary["natural_trajectories_selected"] == 1
        assert summary["natural_queries_available"] == 2
        assert len(benchmark.query_metadata) == 2

    assert set.union(*task_ids_by_split.values()) == {query.id for query in queries}
    assert not (graph_ids_by_split["train"] & graph_ids_by_split["dev"])
    assert not (graph_ids_by_split["train"] & graph_ids_by_split["test"])
    assert not (graph_ids_by_split["dev"] & graph_ids_by_split["test"])


def test_natural_memory_modes_are_loaded_from_authoring_metadata(
    tmp_path: Path,
) -> None:
    raw = _raw_trajectory(0)
    trajectory = adapt_isetrace_record(
        parse_isetrace_record(raw), source_revision="fixture-revision"
    )
    trajectory_path = tmp_path / "trajectories.jsonl"
    trajectory_path.write_text(json.dumps(raw) + "\n", encoding="utf-8")
    queries = [
        _query(trajectory, query_id="query:direct"),
        _query(trajectory, query_id="query:linked"),
    ]
    query_path = tmp_path / "queries.jsonl"
    query_path.write_text(
        "".join(query.model_dump_json() + "\n" for query in queries),
        encoding="utf-8",
    )
    metadata_path = authoring_metadata_path(query_path)
    metadata_path.write_text(
        "".join(
            record.model_dump_json() + "\n"
            for record in (
                AuthoringQueryMetadataRecord(
                    query_id="query:direct",
                    task_key="task:direct",
                    trajectory_id=trajectory.trajectory_id,
                    memory_mode="direct_recall",
                ),
                AuthoringQueryMetadataRecord(
                    query_id="query:linked",
                    task_key="task:linked",
                    trajectory_id=trajectory.trajectory_id,
                    memory_mode="linked_recall",
                ),
            )
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
        split="test",
        trajectory_splits=_trajectory_splits(natural=(0, 0, 1)),
        chunking=_CHUNKING,
        tokenizer=CharacterOffsetTokenizer(),
        authoring_metadata_source=metadata_path,
    )

    assert {item.task_id: item.memory_mode for item in benchmark.query_metadata} == {
        "query:direct": "direct_recall",
        "query:linked": "linked_recall",
    }
    assert summary["authoring_metadata_records"] == 2

    metadata_path.write_text(
        AuthoringQueryMetadataRecord(
            query_id="query:direct",
            task_key="task:direct",
            trajectory_id=trajectory.trajectory_id,
            memory_mode="direct_recall",
        ).model_dump_json()
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(
        ValueError, match="authoring queries and metadata must align exactly"
    ):
        prepare_isetrace_benchmark(
            query_path,
            trajectory_path,
            source_revision="fixture-revision",
            count=None,
            seed=13,
            offset=0,
            strict=True,
            split="test",
            trajectory_splits=_trajectory_splits(natural=(0, 0, 1)),
            chunking=_CHUNKING,
            tokenizer=CharacterOffsetTokenizer(),
            authoring_metadata_source=metadata_path,
        )


def test_profile_cap_limits_fixed_isetrace_split_to_total_tasks(
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
            _query(trajectory, query_id=f"query:{index}:0").model_dump_json() + "\n"
            for index, trajectory in enumerate(trajectories)
        ),
        encoding="utf-8",
    )

    for split in ("train", "dev", "test"):
        benchmark, summary = prepare_isetrace_benchmark(
            query_path,
            trajectory_path,
            source_revision="fixture-revision",
            count=1,
            seed=13,
            offset=0,
            strict=True,
            split=split,
            trajectory_splits=_trajectory_splits(),
            chunking=_CHUNKING,
            tokenizer=CharacterOffsetTokenizer(),
        )
        assert len(benchmark.rankings) == 1
        assert summary["queries_selected"] == 1
        assert summary["natural_queries_selected"] == 1


def test_preparation_fails_on_insufficient_trajectory_pool(
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
        match=r"insufficient ISETrace natural trajectory pool: requested=2 available=1",
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
            trajectory_splits=_trajectory_splits(natural=(1, 1, 0)),
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
        trajectory_splits=_trajectory_splits(natural=(1, 0, 0)),
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
    assert summary["natural_trajectories_selected"] == 1


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
            trajectory_splits=_trajectory_splits(natural=(0, 0, 1)),
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
            trajectory_splits=_trajectory_splits(natural=(0, 0, 1)),
            chunking=_CHUNKING,
            tokenizer=CharacterOffsetTokenizer(),
        )

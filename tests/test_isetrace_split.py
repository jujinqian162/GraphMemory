from __future__ import annotations

from scripts.build_isetrace_split import (
    TrajectoryIdentity,
    build_split_assignments,
    normalized_intent_text_hash,
)


def _trajectory(
    index: int,
    *,
    intent_ids: tuple[str, ...] | None = None,
    texts: tuple[str, ...] | None = None,
) -> TrajectoryIdentity:
    active_intent_ids = intent_ids or (f"intent-{index}",)
    active_texts = texts or tuple(
        f"task {intent_id}" for intent_id in active_intent_ids
    )
    return TrajectoryIdentity(
        trajectory_id=f"trajectory-{index}",
        source_intent_ids=active_intent_ids,
        intent_text_hashes=tuple(
            normalized_intent_text_hash(text) for text in active_texts
        ),
        source_file="trajectories-00000.jsonl",
        source_line=index + 1,
    )


def test_split_is_deterministic_and_hits_eighty_ten_ten_targets() -> None:
    trajectories = [_trajectory(index) for index in range(100)]

    first, first_summary = build_split_assignments(trajectories, seed=13)
    second, second_summary = build_split_assignments(
        list(reversed(trajectories)), seed=13
    )

    first_by_id = {
        item.trajectory.trajectory_id: (item.group_id, item.split) for item in first
    }
    second_by_id = {
        item.trajectory.trajectory_id: (item.group_id, item.split) for item in second
    }
    assert first_by_id == second_by_id
    assert first_summary == second_summary
    assert first_summary["trajectory_counts"] == {
        "train": 80,
        "dev": 10,
        "test": 10,
    }


def test_split_keeps_shared_intent_components_together() -> None:
    trajectories = [
        _trajectory(0, intent_ids=("intent-a", "intent-b")),
        _trajectory(1, intent_ids=("intent-b", "intent-c")),
        _trajectory(2, intent_ids=("intent-d",)),
        *(_trajectory(index) for index in range(3, 20)),
    ]

    assignments, summary = build_split_assignments(trajectories, seed=13)
    by_id = {item.trajectory.trajectory_id: item for item in assignments}

    assert by_id["trajectory-0"].split == by_id["trajectory-1"].split
    assert by_id["trajectory-0"].group_id == by_id["trajectory-1"].group_id
    assert summary["shared_intent_overlap_across_splits"] == 0


def test_split_groups_exact_normalized_intent_text_duplicates() -> None:
    trajectories = [
        _trajectory(0, intent_ids=("intent-a",), texts=("Audit   the FILE",)),
        _trajectory(1, intent_ids=("intent-b",), texts=(" audit the file ",)),
        *(_trajectory(index) for index in range(2, 20)),
    ]

    assignments, summary = build_split_assignments(trajectories, seed=13)
    by_id = {item.trajectory.trajectory_id: item for item in assignments}

    assert by_id["trajectory-0"].split == by_id["trajectory-1"].split
    assert by_id["trajectory-0"].group_id == by_id["trajectory-1"].group_id
    assert summary["exact_normalized_intent_text_overlap_across_splits"] == 0

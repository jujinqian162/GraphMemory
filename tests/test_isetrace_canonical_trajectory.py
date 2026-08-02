from __future__ import annotations

import json
from pathlib import Path

import pytest

from graph_memory.datasets.isetrace import (
    ISETraceAdaptationError,
    ISETraceIngestionSummary,
    adapt_isetrace_record,
    iter_canonical_trajectories,
    parse_isetrace_record,
)
from graph_memory.trajectories import MessageEvent, ToolCallEvent
from tests.isetrace_fixtures import (
    duplicate_message,
    isetrace_record,
    set_call_arguments,
    set_tool_output_call_id,
    set_tool_output_content,
)

REVISION = "e40e04d41c04e4eb4bae181ebdd41b61c688081b"


def test_isetrace_adapter_preserves_ordered_events_and_pairing() -> None:
    raw = parse_isetrace_record(isetrace_record())
    trajectory = adapt_isetrace_record(raw, source_revision=REVISION)

    assert trajectory.trajectory_id == "traj_fixture"
    assert [intent.intent_id for intent in trajectory.intents] == [
        "intent_a",
        "intent_b",
    ]
    assert len(trajectory.tool_calls) == 5
    assert len(trajectory.tool_outputs) == 5
    calls = [event for event in trajectory.events if isinstance(event, ToolCallEvent)]
    assert [call.call_id for call in calls] == ["c1", "c2", "c_aux", "c3", "c4"]
    assert calls[1].position.message_index == calls[2].position.message_index
    assert calls[1].position.sub_index < calls[2].position.sub_index

    reasoning = [
        event.reasoning_content
        for event in trajectory.events
        if isinstance(event, MessageEvent) and event.reasoning_content
    ]
    assert "The requested path is explicit." in reasoning
    assert trajectory.fingerprint() == adapt_isetrace_record(
        raw, source_revision=REVISION
    ).fingerprint()


def test_isetrace_adapter_rejects_invalid_arguments_and_orphan_outputs() -> None:
    invalid_arguments = isetrace_record()
    set_call_arguments(
        invalid_arguments, message_index=2, call_index=0, value="{"
    )
    with pytest.raises(ISETraceAdaptationError, match="invalid_arguments_json"):
        _ = adapt_isetrace_record(
            parse_isetrace_record(invalid_arguments), source_revision=REVISION
        )

    orphaned = isetrace_record()
    set_tool_output_call_id(orphaned, message_index=3, value="unknown")
    with pytest.raises(ISETraceAdaptationError, match="pairing mismatch"):
        _ = adapt_isetrace_record(
            parse_isetrace_record(orphaned), source_revision=REVISION
        )

    duplicate_output = isetrace_record()
    duplicate_message(duplicate_output, message_index=3)
    with pytest.raises(ISETraceAdaptationError, match="duplicate tool output"):
        _ = adapt_isetrace_record(
            parse_isetrace_record(duplicate_output), source_revision=REVISION
        )


def test_reported_success_conflict_is_counted_without_relabeling() -> None:
    record = isetrace_record()
    set_tool_output_content(
        record,
        message_index=3,
        value="Traceback (most recent call last): simulated failure",
    )
    summary = ISETraceIngestionSummary()
    trajectory = adapt_isetrace_record(
        parse_isetrace_record(record),
        source_revision=REVISION,
        summary=summary,
    )

    first_output = trajectory.tool_outputs[0]
    assert first_output.source_reported_success is True
    assert summary.reported_success_error_text_conflicts == 1


def test_streaming_iteration_keeps_compact_rejection_counts(tmp_path: Path) -> None:
    source = tmp_path / "sample.jsonl"
    malformed = isetrace_record()
    set_call_arguments(
        malformed, message_index=2, call_index=0, value="not-json"
    )
    _ = source.write_text(
        "\n".join(
            [json.dumps(isetrace_record()), json.dumps(malformed), "{bad"]
        ),
        encoding="utf-8",
    )
    summary = ISETraceIngestionSummary()

    trajectories = list(
        iter_canonical_trajectories(
            source,
            source_revision=REVISION,
            strict=False,
            summary=summary,
        )
    )

    assert [trajectory.trajectory_id for trajectory in trajectories] == [
        "traj_fixture"
    ]
    assert summary.records_seen == 3
    assert summary.records_accepted == 1
    assert summary.records_rejected == 2
    assert summary.rejection_reasons == {
        "invalid_arguments_json": 1,
        "invalid_json": 1,
    }
    assert summary.tool_calls == 5
    assert summary.tool_outputs == 5


def test_streaming_iteration_accepts_raw_directory_and_shards(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    shards = raw_root / "trajectories"
    shards.mkdir(parents=True)
    first = isetrace_record()
    first["session_id"] = "traj_first"
    second = isetrace_record()
    second["session_id"] = "traj_second"
    (shards / "trajectories-00001.jsonl").write_text(
        json.dumps(second) + "\n", encoding="utf-8"
    )
    (shards / "trajectories-00000.jsonl").write_text(
        json.dumps(first) + "\n", encoding="utf-8"
    )

    trajectories = list(
        iter_canonical_trajectories(
            raw_root,
            source_revision=REVISION,
            strict=True,
        )
    )

    assert [item.trajectory_id for item in trajectories] == [
        "traj_first",
        "traj_second",
    ]

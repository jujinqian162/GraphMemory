from __future__ import annotations

import io
import json
import urllib.error
from email.message import Message
from pathlib import Path
from typing import cast

import pytest

import scripts.generate_isetrace_llm_queries as authoring
from graph_memory.datasets.isetrace import adapt_isetrace_record, parse_isetrace_record
from graph_memory.graphs.provenance import build_provenance_graph
from graph_memory.query_synthesis.provenance import AuthoringQueryRecord
from graph_memory.trajectories import SourceSpan
from scripts.generate_isetrace_llm_queries import (
    PROMPT_VERSION,
    _OUTPUT_SCHEMA,
    _SYSTEM_PROMPT,
    RuntimeSettings,
    _cached_response,
    _example,
    _generate_chunk,
    _packet,
    _plan_tasks,
    _post_response,
    _validate_evidence_quotes,
    _validate_query_contract,
    _weighted_mode_cycle,
    main,
)
from tests.isetrace_fixtures import isetrace_record

REVISION = "e40e04d41c04e4eb4bae181ebdd41b61c688081b"


def _planned_tasks():
    trajectory = adapt_isetrace_record(
        parse_isetrace_record(isetrace_record()), source_revision=REVISION
    )
    graph = build_provenance_graph(trajectory)
    return _plan_tasks(
        trajectory,
        graph,
        seed=13,
        per_trajectory=4,
        include_call_result=False,
    )


def test_authoring_packet_exposes_only_v7_task_text() -> None:
    tasks = _planned_tasks()
    packet = _packet(tasks)
    serialized = json.dumps(packet)

    assert tasks
    assert all(task.motif.motif_type != "call_result" for task in tasks)
    for task in tasks:
        assert task.motif.motif_id not in serialized
    assert "event_id" not in serialized
    assert packet["tasks"]
    payloads = packet["tasks"]
    assert isinstance(payloads, list)
    assert "episode_context" not in packet

    for task, payload in zip(tasks, payloads, strict=True):
        assert isinstance(payload, dict)
        aliases = authoring._source_aliases((task,))
        assert payload["requested_query_count"] == 1
        assert payload["text"] == authoring.render_task_text(
            task.trajectory, task.graph, aliases
        )
        assert "[I1 | user_intent]" in payload["text"]
        assert "[A1 | tool_call |" in payload["text"]
        assert "[E1 | tool_output |" in payload["text"]
        assert set(payload) == {
            "task_key",
            "authoring_brief",
            "style",
            "text",
            "requested_query_count",
        }
        assert payload["style"] == task.style
        assert isinstance(payload["authoring_brief"], str)
        assert aliases[task.target.focus_output_ids[0]] in payload["authoring_brief"]
        assert task.target.query_intent not in payload


def test_task_planning_uses_the_documented_memory_mode_weights() -> None:
    cycle = _weighted_mode_cycle()

    assert cycle.count("direct_recall") == 3
    assert cycle.count("linked_recall") == 5
    assert cycle.count("multi_fact_recall") == 2
    assert {task.memory_mode for task in _planned_tasks()} == {
        "direct_recall",
        "linked_recall",
        "multi_fact_recall",
    }


def test_llm_example_persists_only_the_four_field_authoring_schema() -> None:
    task = _planned_tasks()[0]
    aliases = authoring._source_aliases((task,))
    source = aliases[task.target.focus_output_ids[0]]
    material = authoring.source_material(task.trajectory, task.graph, aliases)[source]
    assert material.event_id is not None
    assert material.json_pointer is not None
    source_text = material.text
    quote = source_text[:48]
    example = _example(
        task=task,
        query_text="What result recorded how the selected report artifact was handled later?",
        gold=(
            authoring.ResolvedAuthoringGold(
                source=source,
                quote=quote,
                span=SourceSpan(
                    event_id=material.event_id,
                    json_pointer=material.json_pointer,
                    char_start=0,
                    char_end=len(quote),
                ),
            ),
        ),
    )

    assert isinstance(example, AuthoringQueryRecord)
    assert set(example.model_dump()) == {"id", "text", "query", "gold"}
    assert example.text == authoring.render_task_text(
        task.trajectory, task.graph, aliases
    )
    assert example.gold[0].source == source
    assert example.gold[0].quote == quote
    assert example.id.startswith("query:")
    assert PROMPT_VERSION not in example.model_dump_json()


def test_minimal_authoring_contract_rejects_missing_or_ambiguous_gold() -> None:
    base = {
        "id": "query:test",
        "text": "[I1 | user_intent]\nAudit the report.\n\n"
        "[E1 | tool_output | read]\nstatus: ready\nstatus: ready",
        "query": "What status was recorded?",
    }
    with pytest.raises(ValueError, match="exactly once"):
        _ = AuthoringQueryRecord.model_validate(
            {**base, "gold": ({"source": "E1", "quote": "ready"},)}
        )
    with pytest.raises(ValueError, match="missing source"):
        _ = AuthoringQueryRecord.model_validate(
            {**base, "gold": ({"source": "E2", "quote": "ready"},)}
        )
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        _ = AuthoringQueryRecord.model_validate(
            {
                **base,
                "gold": ({"source": "E1", "quote": "status: ready\\nstatus"},),
                "motif_id": "internal",
            }
        )


def test_output_schema_requests_only_query_and_exact_gold() -> None:
    envelope_defs = cast(dict[str, object], _OUTPUT_SCHEMA["$defs"])
    item_schema = cast(dict[str, object], envelope_defs["LlmResponseItem"])
    item_properties = cast(dict[str, object], item_schema["properties"])
    assert set(item_properties) == {
        "task_key",
        "decision",
        "queries",
        "rejection_reason",
    }
    queries = cast(dict[str, object], item_properties["queries"])
    query_schema = cast(dict[str, object], envelope_defs["LlmAuthoredQuery"])
    query_properties = cast(dict[str, object], query_schema["properties"])
    assert set(query_properties) == {"query_text", "evidence_quotes"}
    assert queries["maxItems"] == 3


def test_request_body_explicitly_disables_gateway_streaming() -> None:
    body, _digest, _cache_key = authoring._request_body(
        packet={"tasks": []},
        settings=RuntimeSettings(
            model_id="test-model",
            api_key="secret",
            base_url="https://example.test/v1",
        ),
    )

    assert body["stream"] is False
    assert body["store"] is False


def test_v7_exact_quote_maps_argument_source_to_span() -> None:
    task = _planned_tasks()[0]
    aliases = authoring._source_aliases((task,))
    argument_alias = next(alias for alias in aliases.values() if alias.startswith("A"))
    material = authoring.source_material(task.trajectory, task.graph, aliases)[
        argument_alias
    ]
    assert material.event_id is not None
    assert material.json_pointer is not None
    source_text = material.text

    spans, grounded, reason = _validate_evidence_quotes(
        (authoring.LlmEvidenceQuote(source=argument_alias, quote=source_text),),
        task=task,
        aliases=aliases,
    )

    assert reason is None
    assert grounded == (argument_alias,)
    assert spans == (
        authoring.ResolvedAuthoringGold(
            source=argument_alias,
            quote=source_text,
            span=SourceSpan(
                event_id=material.event_id,
                json_pointer="/raw_arguments",
                char_start=0,
                char_end=len(source_text),
            ),
        ),
    )


def test_v7_prompt_owns_soft_quality_while_contract_gate_blocks_leaks() -> None:
    task = _planned_tasks()[0]
    assert "DIVERSE GOOD EXAMPLES" in _SYSTEM_PROMPT
    assert "BAD EXAMPLES" in _SYSTEM_PROMPT
    assert "12 to 35 words" in _SYSTEM_PROMPT
    assert "Each group chooses its own minimal evidence" in _SYSTEM_PROMPT
    assert (
        _validate_query_contract(
            "Saved?",
            task=task,
            seen_queries=set(),
        )
        is None
    )
    assert (
        _validate_query_contract(
            "What did I1 tell us about the report review outcome?",
            task=task,
            seen_queries=set(),
        )
        == "query contains benchmark or internal metadata language"
    )
    assert (
        _validate_query_contract(
            "Which report outcome should the planning team remember?",
            task=task,
            seen_queries={"which report outcome should the planning team remember?"},
        )
        == "duplicate normalized query"
    )


def test_retryable_gateway_403_is_retried(monkeypatch, caplog) -> None:
    attempts = 0

    def fake_urlopen(request, timeout):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            headers = Message()
            headers["Retry-After"] = "0"
            raise urllib.error.HTTPError(
                request.full_url,
                403,
                "Forbidden",
                headers,
                io.BytesIO(
                    b'{"error":{"type":"bad_response_status_code",'
                    b'"message":"Please try again"}}'
                ),
            )
        return io.BytesIO(b'{"id":"resp_ok","status":"completed","output":[]}')

    monkeypatch.setattr(authoring.urllib.request, "urlopen", fake_urlopen)
    response = _post_response(
        {"model": "test-model", "input": "test"},
        settings=RuntimeSettings(
            model_id="test-model", api_key="secret", base_url="https://example.test/v1"
        ),
        timeout=1.0,
        max_retries=1,
    )

    assert response["id"] == "resp_ok"
    assert attempts == 2
    assert "Responses API HTTP 403; retrying attempt 2/2" in caplog.text


def test_response_cache_avoids_duplicate_api_calls(tmp_path: Path, monkeypatch) -> None:
    calls: list[object] = []
    response = {"id": "resp_cached", "status": "completed", "output": []}

    def fake_post(*args, **kwargs):
        calls.append((args, kwargs))
        return response

    monkeypatch.setattr(authoring, "_post_response", fake_post)
    settings = RuntimeSettings(
        model_id="test-model", api_key="secret", base_url="https://example.test/v1"
    )
    first, first_hit = _cached_response(
        {"model": "test-model"},
        request_digest="b" * 64,
        settings=settings,
        cache_dir=tmp_path,
        timeout=1.0,
        api_retries=2,
    )
    second, second_hit = _cached_response(
        {"model": "test-model"},
        request_digest="b" * 64,
        settings=settings,
        cache_dir=tmp_path,
        timeout=1.0,
        api_retries=2,
    )

    assert first == second == response
    assert first_hit is False
    assert second_hit is True
    assert len(calls) == 1


def test_exhausted_api_failure_aborts_chunk_for_safe_resume(
    tmp_path: Path, monkeypatch
) -> None:
    task = _planned_tasks()[0]

    def fail_cached_response(*args, **kwargs):
        raise RuntimeError("Responses API HTTP 426: WebSocket upgrade required")

    monkeypatch.setattr(authoring, "_cached_response", fail_cached_response)

    with pytest.raises(RuntimeError, match="authoring API failed after transport retries"):
        _generate_chunk(
            [task],
            settings=RuntimeSettings(
                model_id="test-model",
                api_key="secret",
                base_url="https://example.test/v1",
            ),
            cache_dir=tmp_path,
            timeout=1.0,
            api_retries=0,
            validation_retries=2,
            seen_queries=set(),
        )


def _model_response(item: dict[str, object], *, response_id: str) -> dict[str, object]:
    return {
        "id": response_id,
        "status": "completed",
        "model": "test-model",
        "output": [
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": json.dumps({"items": [item]}),
                    }
                ],
            }
        ],
    }


def test_model_rejection_is_terminal(tmp_path: Path, monkeypatch) -> None:
    task = _planned_tasks()[0]
    calls = 0
    response = _model_response(
        {
            "task_key": task.task_key,
            "decision": "reject",
            "queries": [],
            "rejection_reason": "the selected events are semantically unrelated",
        },
        response_id="resp_reject",
    )

    def fake_cached(*args, **kwargs):
        nonlocal calls
        calls += 1
        return response, False

    monkeypatch.setattr(authoring, "_cached_response", fake_cached)
    examples, metadata, rejected, _ = _generate_chunk(
        [task],
        settings=RuntimeSettings(
            model_id="test-model", api_key="secret", base_url="https://example.test/v1"
        ),
        cache_dir=tmp_path,
        timeout=1.0,
        api_retries=1,
        validation_retries=1,
        seen_queries=set(),
    )

    assert not examples
    assert not metadata
    assert calls == 1
    assert rejected == [
        {
            "task_key": task.task_key,
            "memory_mode": task.memory_mode,
            "reason": "the selected events are semantically unrelated",
            "attempts": 1,
            "kind": "model_rejection",
        }
    ]


def test_invalid_gold_source_is_rewritten_and_revalidated(
    tmp_path: Path, monkeypatch
) -> None:
    task = _planned_tasks()[0]
    aliases = authoring._source_aliases((task,))
    focus_alias = aliases[task.target.focus_output_ids[0]]
    material = authoring.source_material(task.trajectory, task.graph, aliases)
    other_alias = next(alias for alias in material if alias != focus_alias)
    focus_text = material[focus_alias].text
    other_text = material[other_alias].text
    first_evidence = [
        {
            "source": focus_alias,
            "quote": focus_text[: min(48, len(focus_text))],
        }
    ]
    second_evidence = [
        {
            "source": other_alias,
            "quote": other_text[: min(48, len(other_text))],
        }
    ]
    accepted_item: dict[str, object] = {
        "task_key": task.task_key,
        "decision": "accept",
        "queries": [
            {
                "query_text": "What result was recorded when the report was used for the follow-up action?",
                "evidence_quotes": first_evidence,
            },
            {
                "query_text": "During the report review, what earlier detail should the planning team remember?",
                "evidence_quotes": second_evidence,
            },
        ],
        "rejection_reason": None,
    }
    accepted_queries = cast(list[dict[str, object]], accepted_item["queries"])
    responses = iter(
        [
            _model_response(
                {
                    **accepted_item,
                    "queries": [
                        {
                            **accepted_queries[0],
                            "evidence_quotes": [
                                {"source": "D1", "quote": "invalid source quote"}
                            ],
                        },
                        accepted_queries[1],
                    ],
                },
                response_id="resp_bad_grounding",
            ),
            _model_response(accepted_item, response_id="resp_fixed"),
        ]
    )

    def fake_cached(*args, **kwargs):
        return next(responses), False

    monkeypatch.setattr(authoring, "_cached_response", fake_cached)
    examples, metadata, rejected, _ = _generate_chunk(
        [task],
        settings=RuntimeSettings(
            model_id="test-model", api_key="secret", base_url="https://example.test/v1"
        ),
        cache_dir=tmp_path,
        timeout=1.0,
        api_retries=1,
        validation_retries=1,
        seen_queries=set(),
        queries_per_task=2,
    )

    assert len(examples) == 2
    assert len(metadata) == 2
    assert not rejected
    assert examples[0].gold != examples[1].gold
    assert {item.memory_mode for item in metadata} == {task.memory_mode}
    for example in examples:
        assert isinstance(example, AuthoringQueryRecord)
        assert set(example.model_dump()) == {"id", "text", "query", "gold"}


def test_dry_run_writes_packets_without_env_or_network(tmp_path: Path) -> None:
    source = tmp_path / "sample.jsonl"
    source.write_text(json.dumps(isetrace_record()) + "\n", encoding="utf-8")
    output = tmp_path / "queries.jsonl"

    status = main(
        [
            "--source",
            str(source),
            "--source-revision",
            REVISION,
            "--output",
            str(output),
            "--limit",
            "1",
            "--dry-run",
        ]
    )

    packet_path = output.with_suffix(output.suffix + ".packets.jsonl")
    assert status == 0
    packets = [
        json.loads(line)
        for line in packet_path.read_text(encoding="utf-8").splitlines()
    ]
    assert (
        sum(
            task["requested_query_count"]
            for packet in packets
            for task in packet["tasks"]
        )
        == 2
    )
    assert {
        task["requested_query_count"] for packet in packets for task in packet["tasks"]
    } == {1}
    assert all("episode_context" not in packet for packet in packets)
    assert all(
        set(task)
        >= {
            "task_key",
            "authoring_brief",
            "style",
            "text",
            "requested_query_count",
        }
        for packet in packets
        for task in packet["tasks"]
    )
    assert not output.exists()


def test_authoring_cli_defaults_are_trajectory_based(tmp_path: Path) -> None:
    args = authoring.parse_args(["--output", str(tmp_path / "queries.jsonl")])

    assert args.source == authoring.DEFAULT_SOURCE
    assert args.limit == 100
    assert args.per_trajectory == 2
    assert args.tasks_per_call == 2
    assert args.queries_per_task == 1


def test_run_identity_allows_endpoint_changes_but_not_model_changes(
    tmp_path: Path,
) -> None:
    args = authoring.parse_args(
        [
            "--source",
            str(tmp_path),
            "--source-revision",
            REVISION,
            "--output",
            str(tmp_path / "queries.jsonl"),
        ]
    )
    corpus = authoring.ScannedCorpus(
        refs=(),
        files=({"path": "fixture", "bytes": 0, "records": 0, "sha256": "a" * 64},),
    )
    first = authoring._run_identity(
        args,
        corpus=corpus,
        settings=RuntimeSettings(
            model_id="gpt-test", api_key="first", base_url="https://first.test/v1"
        ),
    )
    second = authoring._run_identity(
        args,
        corpus=corpus,
        settings=RuntimeSettings(
            model_id="gpt-test", api_key="second", base_url="https://second.test/v1"
        ),
    )
    changed_model = authoring._run_identity(
        args,
        corpus=corpus,
        settings=RuntimeSettings(
            model_id="other-model", api_key="second", base_url="https://second.test/v1"
        ),
    )

    assert first == second
    assert "base_url" not in first
    assert first["schema_version"] == 2
    assert first != changed_model


def test_legacy_endpoint_manifest_migrates_and_records_segments(
    tmp_path: Path,
) -> None:
    output = tmp_path / "queries.jsonl"
    identity: dict[str, object] = {
        "schema_version": 2,
        "prompt_version": PROMPT_VERSION,
        "source_revision": REVISION,
        "source_files": [],
        "shuffle_seed": 13,
        "per_trajectory": 2,
        "tasks_per_call": 2,
        "queries_per_task": 1,
        "include_call_result": False,
        "model_id": "gpt-test",
    }
    legacy = {
        **identity,
        "schema_version": 1,
        "base_url": "https://first.test/v1",
    }
    manifest = output.with_suffix(output.suffix + ".run.json")
    manifest.write_text(json.dumps(legacy), encoding="utf-8")

    legacy_endpoint = authoring._initialize_run_manifest(output, identity=identity)
    authoring._record_endpoint_segment(
        output,
        model_id="gpt-test",
        base_url="https://second.test/v1/",
        selection_start=6000,
        legacy_base_url=legacy_endpoint,
    )
    authoring._record_endpoint_segment(
        output,
        model_id="gpt-test",
        base_url="https://second.test/v1",
        selection_start=7000,
    )

    assert json.loads(manifest.read_text(encoding="utf-8")) == identity
    history_path = output.with_suffix(output.suffix + ".endpoint-history.jsonl")
    assert [json.loads(line) for line in history_path.read_text().splitlines()] == [
        {
            "kind": "endpoint_segment",
            "model_id": "gpt-test",
            "base_url": "https://first.test/v1",
            "selection_start": 0,
        },
        {
            "kind": "endpoint_segment",
            "model_id": "gpt-test",
            "base_url": "https://second.test/v1",
            "selection_start": 6000,
        },
    ]


def test_raw_trajectory_shuffle_is_deterministic_and_prefix_stable(
    tmp_path: Path,
) -> None:
    source = tmp_path / "raw"
    source.mkdir()
    for shard in range(2):
        (source / f"trajectories-{shard:05d}.jsonl").write_text(
            "".join(f'{{"record": {shard * 5 + index}}}\n' for index in range(5)),
            encoding="utf-8",
        )

    first = authoring._scan_corpus(source, seed=13)
    repeated = authoring._scan_corpus(source, seed=13)
    different_seed = authoring._scan_corpus(source, seed=17)

    assert [ref.key for ref in first.refs] == [ref.key for ref in repeated.refs]
    assert [ref.key for ref in first.refs[:3]] == [ref.key for ref in repeated.refs[:3]]
    assert [ref.key for ref in first.refs] != [ref.key for ref in different_seed.refs]


def test_trajectory_limit_resumes_without_regenerating_completed_prefix(
    tmp_path: Path, monkeypatch
) -> None:
    refs = tuple(
        authoring.RawTrajectoryRef(
            path=tmp_path / "raw.jsonl",
            byte_offset=index,
            line_number=index + 1,
        )
        for index in range(3)
    )
    corpus = authoring.ScannedCorpus(
        refs=refs,
        files=(
            {
                "path": str((tmp_path / "raw.jsonl").resolve()),
                "bytes": 3,
                "records": 3,
                "sha256": "a" * 64,
            },
        ),
    )

    trajectories = []
    for index in range(3):
        record = isetrace_record()
        record["session_id"] = f"traj_{index}"
        call_ids: dict[str, str] = {}
        messages = cast(list[dict[str, object]], record["messages"])
        for message in messages:
            tool_calls = cast(list[dict[str, object]], message.get("tool_calls", []))
            for call in tool_calls:
                old_id = cast(str, call["id"])
                new_id = f"{old_id}_{index}"
                call["id"] = new_id
                call_ids[old_id] = new_id
            old_output_id = message.get("tool_call_id")
            if isinstance(old_output_id, str):
                message["tool_call_id"] = call_ids[old_output_id]
        trajectories.append(
            adapt_isetrace_record(
                parse_isetrace_record(record), source_revision=REVISION
            )
        )

    monkeypatch.setattr(authoring, "_scan_corpus", lambda *args, **kwargs: corpus)
    monkeypatch.setattr(
        authoring,
        "_read_raw_trajectory",
        lambda ref, **kwargs: trajectories[ref.line_number - 1],
    )
    monkeypatch.setattr(
        authoring,
        "_load_env",
        lambda path: RuntimeSettings(
            model_id="test-model",
            api_key="secret",
            base_url="https://example.test/v1",
        ),
    )

    generated_trajectory_ids: list[str] = []

    def fake_generate(tasks, **kwargs):
        generated_trajectory_ids.append(tasks[0].trajectory.trajectory_id)
        examples = []
        metadata = []
        for task in tasks:
            aliases = authoring._source_aliases((task,))
            source = aliases[task.target.focus_output_ids[0]]
            material = authoring.source_material(task.trajectory, task.graph, aliases)[
                source
            ]
            quote = material.text[: min(48, len(material.text))]
            assert material.event_id is not None
            assert material.json_pointer is not None
            example = _example(
                task=task,
                query_text=(
                    "What substantive result should be remembered for "
                    f"{task.trajectory.trajectory_id}?"
                ),
                gold=(
                    authoring.ResolvedAuthoringGold(
                        source=source,
                        quote=quote,
                        span=SourceSpan(
                            event_id=material.event_id,
                            json_pointer=material.json_pointer,
                            char_start=0,
                            char_end=len(quote),
                        ),
                    ),
                ),
            )
            examples.append(example)
            metadata.append(
                authoring.GeneratedQueryMetadata(
                    query_id=example.id,
                    task_key=task.task_key,
                    trajectory_id=task.trajectory.trajectory_id,
                    memory_mode=task.memory_mode,
                )
            )
        return examples, metadata, [], 0

    monkeypatch.setattr(authoring, "_generate_chunk", fake_generate)
    output = tmp_path / "queries.jsonl"
    common = ["--source", str(tmp_path), "--output", str(output)]

    assert main([*common, "--limit", "1"]) == 0
    assert generated_trajectory_ids == ["traj_0"]
    assert len(output.read_text(encoding="utf-8").splitlines()) == 2

    assert main([*common, "--limit", "3"]) == 0
    assert generated_trajectory_ids == ["traj_0", "traj_1", "traj_2"]
    assert len(output.read_text(encoding="utf-8").splitlines()) == 6

    assert main([*common, "--limit", "3"]) == 0
    assert generated_trajectory_ids == ["traj_0", "traj_1", "traj_2"]

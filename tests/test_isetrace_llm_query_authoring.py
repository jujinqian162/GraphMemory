from __future__ import annotations

import io
import json
import urllib.error
from email.message import Message
from pathlib import Path
from typing import cast

import scripts.generate_isetrace_llm_queries as authoring
from graph_memory.datasets.isetrace import adapt_isetrace_record, parse_isetrace_record
from graph_memory.graphs.provenance import build_provenance_graph
from graph_memory.query_synthesis.provenance import LlmGenerationProvenance
from scripts.generate_isetrace_llm_queries import (
    PROMPT_VERSION,
    _OUTPUT_SCHEMA,
    _SYSTEM_PROMPT,
    RuntimeSettings,
    _cached_response,
    _evidence_aliases,
    _episode_aliases,
    _example,
    _generate_chunk,
    _packet,
    _plan_tasks,
    _post_response,
    _queries_are_too_similar,
    _validate_context_anchors,
    _validate_query,
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


def test_authoring_packet_reuses_motif_labels_without_exposing_internal_ids() -> None:
    tasks = _planned_tasks()
    packet = _packet(tasks)
    serialized = json.dumps(packet)

    assert tasks
    assert all(task.motif.motif_type != "call_result" for task in tasks)
    for task in tasks:
        assert task.motif.motif_id not in serialized
        assert not any(
            output_id in serialized
            for output_id in task.target.support_output_ids
        )
    assert packet["tasks"]
    payloads = packet["tasks"]
    assert isinstance(payloads, list)
    aliases = _episode_aliases(tasks)
    episode_context = packet["episode_context"]
    assert isinstance(episode_context, dict)
    events = episode_context["events"]
    assert isinstance(events, list)
    assert len({event["alias"] for event in events}) == len(events)
    assert all("arguments=" not in event["excerpt"] for event in events)

    for task, payload in zip(tasks, payloads, strict=True):
        assert isinstance(payload, dict)
        assert payload["requested_query_count"] == 2
        assert payload["answer_event_aliases"] == [
            aliases[node_id] for node_id in task.target.answer_output_ids
        ]
        assert set(payload["support_event_aliases"]) == {
            aliases[node_id] for node_id in task.target.support_output_ids
        }
        assert set(payload["context_event_aliases"]).isdisjoint(
            payload["answer_event_aliases"]
        )
        assert all(
            edge["source"] in aliases.values() and edge["target"] in aliases.values()
            for edge in payload["dependency_edges"]
        )


def test_llm_example_keeps_gold_label_separate_from_generation_provenance() -> None:
    task = _planned_tasks()[0]
    response = {
        "id": "resp_test",
        "status": "completed",
        "model": "test-model-reported",
        "prompt_cache_key": "server-key",
        "instructions": "gateway instructions",
        "usage": {
            "input_tokens": 100,
            "input_tokens_details": {"cached_tokens": 80},
            "output_tokens": 20,
        },
    }
    example = _example(
        task=task,
        query_text="What result recorded how the selected report artifact was handled later?",
        reference_answer="The later read returned the prepared report.",
        response=response,
        settings=RuntimeSettings(
            model_id="test-model", api_key="secret", base_url="https://example.test/v1"
        ),
        request_digest="a" * 64,
        prompt_cache_key="requested-key",
        authoring_attempt=1,
    )

    assert example.label.answer_output_ids == task.target.answer_output_ids
    assert example.label.support_output_ids == task.target.support_output_ids
    assert isinstance(example.generation, LlmGenerationProvenance)
    assert example.generation.annotation_method == "llm_generated"
    assert example.generation.human_review_status == "unreviewed"
    assert example.generation.prompt_version == PROMPT_VERSION
    assert example.generation.authoring_attempt == 1
    assert example.generation.reference_answer == "The later read returned the prepared report."
    assert "template_id" not in example.label.model_dump()


def test_output_schema_uses_only_gpt_supported_array_keywords() -> None:
    properties = cast(dict[str, object], _OUTPUT_SCHEMA["properties"])
    items = cast(dict[str, object], properties["items"])
    item_schema = cast(dict[str, object], items["items"])
    item_properties = cast(dict[str, object], item_schema["properties"])
    grounding_aliases = cast(
        dict[str, object], item_properties["grounding_event_aliases"]
    )
    queries = cast(dict[str, object], item_properties["queries"])
    query_item = cast(dict[str, object], queries["items"])
    query_properties = cast(dict[str, object], query_item["properties"])
    context_aliases = cast(
        dict[str, object], query_properties["context_anchor_aliases"]
    )
    assert "uniqueItems" not in grounding_aliases
    assert "uniqueItems" not in context_aliases
    assert queries["maxItems"] == 3


def test_v4_prompt_and_local_quality_gates_reject_lexical_shortcuts() -> None:
    task = _planned_tasks()[0]
    aliases = _evidence_aliases(task)
    answer_alias = aliases[task.target.answer_output_ids[0]]

    assert "GOOD EXAMPLE" in _SYSTEM_PROMPT
    assert "BAD EXAMPLES" in _SYSTEM_PROMPT
    assert _validate_context_anchors(
        ["I1"], task=task, aliases=aliases
    ) is None
    assert _validate_context_anchors(
        [answer_alias], task=task, aliases=aliases
    ) is not None
    assert _validate_query(
        "How many bytes were written when the report was saved?",
        task=task,
        seen_queries=set(),
    ) == "query asks about a trivial tool receipt or invocation detail"
    assert _validate_query(
        "What did I1 tell us about the report review outcome?",
        task=task,
        seen_queries=set(),
    ) == "query contains benchmark or internal metadata language"
    assert _queries_are_too_similar(
        "Why did the first report fail and what correction followed?",
        "Why did the initial report fail and what correction followed?",
    )
    assert not _queries_are_too_similar(
        "Why did the first report fail and what correction followed?",
        "What outcome did the later review preserve for the planning team?",
    )


def test_retryable_gateway_403_is_retried(monkeypatch) -> None:
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
            "reference_answer": None,
            "grounding_event_aliases": [],
            "all_answer_events_necessary": None,
            "relation_is_meaningful": None,
            "rejection_reason": "the selected events are semantically unrelated",
        },
        response_id="resp_reject",
    )

    def fake_cached(*args, **kwargs):
        nonlocal calls
        calls += 1
        return response, False

    monkeypatch.setattr(authoring, "_cached_response", fake_cached)
    examples, rejected, _ = _generate_chunk(
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
    assert calls == 1
    assert rejected == [
        {
            "task_key": task.task_key,
            "reason": "the selected events are semantically unrelated",
            "attempts": 1,
            "kind": "model_rejection",
        }
    ]


def test_failed_grounding_self_check_is_rewritten_and_revalidated(
    tmp_path: Path, monkeypatch
) -> None:
    task = _planned_tasks()[0]
    aliases = _evidence_aliases(task)
    answer_aliases = [aliases[node_id] for node_id in task.target.answer_output_ids]
    accepted_item: dict[str, object] = {
        "task_key": task.task_key,
        "decision": "accept",
        "queries": [
            {
                "query_text": "What result was recorded when the report was used for the follow-up action?",
                "context_anchor_aliases": ["I1"],
            },
            {
                "query_text": "During the report review, what did the later use of the prepared artifact return?",
                "context_anchor_aliases": ["I2"],
            },
        ],
        "reference_answer": "The answer action returned the prepared report.",
        "grounding_event_aliases": answer_aliases,
        "all_answer_events_necessary": True,
        "relation_is_meaningful": True,
        "rejection_reason": None,
    }
    responses = iter(
        [
            _model_response(
                {
                    **accepted_item,
                    "grounding_event_aliases": ["D1"],
                },
                response_id="resp_bad_grounding",
            ),
            _model_response(accepted_item, response_id="resp_fixed"),
        ]
    )

    def fake_cached(*args, **kwargs):
        return next(responses), False

    monkeypatch.setattr(authoring, "_cached_response", fake_cached)
    examples, rejected, _ = _generate_chunk(
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

    assert len(examples) == 2
    assert not rejected
    for example in examples:
        assert isinstance(example.generation, LlmGenerationProvenance)
        assert example.generation.authoring_attempt == 2


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
            "5",
            "--dry-run",
        ]
    )

    packet_path = output.with_suffix(output.suffix + ".packets.jsonl")
    assert status == 0
    packets = [
        json.loads(line)
        for line in packet_path.read_text(encoding="utf-8").splitlines()
    ]
    assert sum(
        task["requested_query_count"]
        for packet in packets
        for task in packet["tasks"]
    ) == 5
    assert {
        task["requested_query_count"]
        for packet in packets
        for task in packet["tasks"]
    } == {1, 2}
    assert all("episode_context" in packet for packet in packets)
    assert not output.exists()

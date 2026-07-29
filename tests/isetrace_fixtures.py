from __future__ import annotations

import json
from typing import cast


def _tool(name: str, properties: dict[str, object]) -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"{name} tool",
            "parameters": json.dumps(
                {"type": "object", "properties": properties}
            ),
        },
    }


def _call(call_id: str, name: str, arguments: dict[str, object]) -> dict[str, object]:
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments),
        },
    }


def set_call_arguments(
    record: dict[str, object],
    *,
    message_index: int,
    call_index: int,
    value: str,
) -> None:
    messages = cast(list[dict[str, object]], record["messages"])
    tool_calls = cast(list[dict[str, object]], messages[message_index]["tool_calls"])
    function = cast(dict[str, object], tool_calls[call_index]["function"])
    function["arguments"] = value


def set_tool_output_call_id(
    record: dict[str, object], *, message_index: int, value: str
) -> None:
    messages = cast(list[dict[str, object]], record["messages"])
    messages[message_index]["tool_call_id"] = value


def set_tool_output_content(
    record: dict[str, object], *, message_index: int, value: str
) -> None:
    messages = cast(list[dict[str, object]], record["messages"])
    messages[message_index]["content"] = value


def duplicate_message(record: dict[str, object], *, message_index: int) -> None:
    messages = cast(list[dict[str, object]], record["messages"])
    messages.insert(message_index + 1, dict(messages[message_index]))


def isetrace_record() -> dict[str, object]:
    path = "/workspace/report.md"
    url = "https://example.com/source/42"
    digest = "0123456789abcdef0123456789abcdef"
    return {
        "status": "completed",
        "session_id": "traj_fixture",
        "source_intent_count": 2,
        "source_intent_ids": ["intent_a", "intent_b"],
        "source_intents": [
            {
                "intent_id": "intent_a",
                "natural_language_intent": "Create and inspect a report.",
                "task_type": "intent",
            },
            {
                "intent_id": "intent_b",
                "natural_language_intent": "Fetch and verify its source.",
                "task_type": "intent",
            },
        ],
        "session_finalized_by_intent_id": "intent_b",
        "total_steps": 5,
        "enable_thinking": True,
        "messages": [
            {"role": "system", "content": "You are a tool agent."},
            {"role": "user", "content": "Prepare the report."},
            {
                "role": "assistant",
                "content": "I will write it.",
                "reasoning_content": "The requested path is explicit.",
                "tool_calls": [_call("c1", "write", {"path": path, "content": "draft"})],
            },
            {
                "role": "tool",
                "name": "write",
                "tool_call_id": "c1",
                "content": f"Successfully wrote {path}",
                "success": True,
            },
            {
                "role": "assistant",
                "content": None,
                "reasoning_content": "Read and list the artifact.",
                "tool_calls": [
                    _call("c2", "read", {"path": path}),
                    _call("c_aux", "exec", {"command": f"ls -l {path}"}),
                ],
            },
            {
                "role": "tool",
                "name": "read",
                "tool_call_id": "c2",
                "content": json.dumps({"source_url": url, "report": "ready"}),
                "success": True,
            },
            {
                "role": "tool",
                "name": "exec",
                "tool_call_id": "c_aux",
                "content": f"-rw-r--r-- {path}",
                "success": True,
            },
            {
                "role": "assistant",
                "content": "I will fetch the source.",
                "reasoning_content": None,
                "tool_calls": [_call("c3", "web_fetch", {"url": url})],
            },
            {
                "role": "tool",
                "name": "web_fetch",
                "tool_call_id": "c3",
                "content": json.dumps({"digest": digest}),
                "success": True,
            },
            {
                "role": "assistant",
                "content": "I will verify both references.",
                "reasoning_content": None,
                "tool_calls": [
                    _call("c4", "exec", {"command": f"verify {url} {digest}"})
                ],
            },
            {
                "role": "tool",
                "name": "exec",
                "tool_call_id": "c4",
                "content": "verified",
                "success": True,
            },
            {"role": "assistant", "content": "The report is verified."},
        ],
        "tools": [
            _tool("read", {"path": {"type": "string"}}),
            _tool(
                "write",
                {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
            ),
            _tool("exec", {"command": {"type": "string"}}),
            _tool("web_fetch", {"url": {"type": "string"}}),
        ],
        "final_output": "The report is verified.",
        "intent_id": "intent_b",
        "metadata": {"persona": {"name": "Test User"}, "domains": ["test"]},
    }

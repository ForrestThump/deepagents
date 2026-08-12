"""End-to-end MVL v1 conformance tests.

These tests are the shared suite the Model View Log spec calls for: an adapter
conforms when a reader can rebuild the exact model view for any turn from the
log alone, and when crash survival, ordering, system/catalogue recovery, tool
honesty and tool-withdrawal visibility hold.

They deliberately drive the tracer through both the low-level middleware hooks
and a full ``create_deep_agent`` graph so production composition (middleware
relocation, filesystem tools, multi-turn tool loops) is covered, not just unit
stubs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from deepagents.graph import create_deep_agent
from deepagents.tracing import (
    ModelViewLogMiddleware,
    assert_log_reconstructs,
    read_events,
    reconstruct_messages,
    reconstruct_system_text,
    reconstruct_tool_catalog,
    reconstruct_turn,
    turns_present,
)
from deepagents.tracing.emitter import sha256_bytes, sha256_json
from deepagents.tracing.middleware import _tool_definition
from deepagents.tracing.reconstruct import (
    ReconstructionError,
    tool_result_content_by_call_id,
)
from tests.unit_tests.chat_model import GenericFakeChatModel


@tool
def _grep(pattern: str) -> str:
    """Search files for a pattern."""
    return f"hit:{pattern}"


@tool
def _edit_file(path: str) -> str:
    """Edit a file at path."""
    return f"edited:{path}"


@tool
def _echo(text: str) -> str:
    """Echo the given text back."""
    return text


class _FakeChatModel:
    """Minimal chat model exposing sampling knobs for direct middleware tests."""

    model_name = "fake-model"
    temperature = 0.0
    max_tokens = 1024


def _model_request(messages, tools, system_message=None):
    return ModelRequest(
        model=_FakeChatModel(),
        messages=messages,
        system_message=system_message,
        tools=tools,
    )


def _ai(content, *, tool_calls=None, usage=None, finish=None, id=None):
    return AIMessage(
        content=content,
        id=id,
        tool_calls=tool_calls or [],
        usage_metadata=usage
        or {
            "input_tokens": 10,
            "output_tokens": 4,
            "total_tokens": 14,
            "input_token_details": {"cached_tokens": 8},
            "output_token_details": {"reasoning_tokens": 1},
        },
        response_metadata={"stop_reason": finish or ("tool_use" if tool_calls else "end_turn")},
    )


def _drive_two_turn_tool_loop(tracer: ModelViewLogMiddleware) -> tuple[AIMessage, ToolMessage]:
    """Drive a full-then-delta tool loop through the middleware hooks."""
    system = SystemMessage("you are a careful coding agent")
    human = HumanMessage("find handle_request", id="h1")
    ai1 = _ai(
        "Let me search.",
        id="a1",
        tool_calls=[{"id": "c1", "name": "_grep", "args": {"pattern": "handle_request"}}],
    )
    tool_msg = ToolMessage(content="hit:handle_request", tool_call_id="c1", name="_grep", id="t1")

    tracer.wrap_model_call(
        _model_request([human], [_grep, _edit_file], system_message=system),
        lambda r: ModelResponse(result=[ai1]),
    )
    request = ToolCallRequest(
        tool=_grep,
        tool_call={"id": "c1", "name": "_grep", "args": {"pattern": "handle_request"}},
        state={},
        runtime=None,
    )
    tracer.wrap_tool_call(request, lambda r: tool_msg)
    tracer.wrap_model_call(
        _model_request([human, ai1, tool_msg], [_grep, _edit_file], system_message=system),
        lambda r: ModelResponse(result=[_ai("done", id="a2")]),
    )
    return ai1, tool_msg


class TestReconstructionConformance:
    """Spec conformance #1, #4, #5: reconstruction from the log alone."""

    def test_tool_definition_includes_real_input_schema(self) -> None:
        """Catalogue entries must carry the complete JSON schema, not `{}`."""
        definition = _tool_definition(_grep)
        assert definition["name"] == "_grep"
        assert "Search files" in (definition["description"] or "")
        schema = definition["input_schema"]
        assert schema.get("type") == "object"
        assert "pattern" in schema.get("properties", {})
        assert "pattern" in schema.get("required", [])

    def test_two_turn_loop_reconstructs_every_artifact(self, tmp_path: Path) -> None:
        """From the log alone, rebuild messages / system / catalog / params / tools."""
        path = tmp_path / "mvl.jsonl"
        tracer = ModelViewLogMiddleware(
            path=path,
            harness_name="deepagents",
            harness_version="test",
            model_id="fake-model",
            model_provider="fake",
            task_id="P3.2",
            task_text="find handle_request",
            config={"loop.doom_threshold": 3},
        )
        with tracer.run():
            _drive_two_turn_tool_loop(tracer)

        events = read_events(path)
        assert_log_reconstructs(events)
        assert turns_present(events) == [1, 2]

        turn1 = reconstruct_turn(events, 1)
        assert turn1["system_text"] == "you are a careful coding agent"
        assert turn1["tools_offered"] == ["_grep", "_edit_file"]
        assert turn1["params"] == {"temperature": 0.0, "max_tokens": 1024}
        assert [m["role"] for m in turn1["messages"]] == ["human"]
        assert turn1["messages"][0]["content"] == "find handle_request"

        catalog = turn1["tool_catalog"]
        assert [t["name"] for t in catalog] == ["_grep", "_edit_file"]
        assert catalog[0]["input_schema"]["properties"]["pattern"]["type"] == "string"
        # Digest on the prompt must match the body we recovered.
        prompt1 = next(e for e in events if e["type"] == "prompt" and e["turn"] == 1)
        assert prompt1["tool_catalog_sha256"] == sha256_json(catalog)
        assert prompt1["system"]["sha256"] == sha256_bytes(b"you are a careful coding agent")

        turn2 = reconstruct_turn(events, 2)
        assert turn2["system_text"] == "you are a careful coding agent"
        roles = [m["role"] for m in turn2["messages"]]
        assert roles == ["human", "ai", "tool"]
        ai_item = turn2["messages"][1]
        assert ai_item["content"] == "Let me search."
        assert ai_item["tool_calls"] == [
            {"id": "c1", "name": "_grep", "arguments": {"pattern": "handle_request"}}
        ]
        tool_item = turn2["messages"][2]
        assert tool_item["content"] == "hit:handle_request"
        assert tool_item["tool_call_id"] == "c1"
        assert tool_item["name"] == "_grep"

        # System text appears in full exactly once; later prompts carry the hash only.
        system_texts = [
            (e.get("system") or {}).get("text") for e in events if e["type"] == "prompt"
        ]
        assert system_texts[0] == "you are a careful coding agent"
        assert all(text is None for text in system_texts[1:])

        # Catalogue body appears exactly once for a stable tool list.
        catalogs = [e for e in events if e["type"] == "tool_catalog"]
        assert len(catalogs) == 1

    def test_delta_after_full_appends_only_new_messages(self, tmp_path: Path) -> None:
        """A reader applying full-then-delta must not double-count prior messages."""
        path = tmp_path / "mvl.jsonl"
        tracer = ModelViewLogMiddleware(path=path)
        with tracer.run():
            _drive_two_turn_tool_loop(tracer)
        events = read_events(path)
        prompts = [e for e in events if e["type"] == "prompt"]
        assert prompts[0]["messages"]["mode"] == "full"
        assert prompts[1]["messages"]["mode"] == "delta"
        delta_roles = [item["role"] for item in prompts[1]["messages"]["items"]]
        assert delta_roles == ["ai", "tool"]
        # Reconstruction at turn 2 must still yield the full three-message history.
        assert len(reconstruct_messages(events, 2)) == 3

    def test_context_changed_forces_full_and_still_reconstructs(self, tmp_path: Path) -> None:
        """After compaction the next prompt is full; reconstruction resets cleanly."""
        path = tmp_path / "mvl.jsonl"
        tracer = ModelViewLogMiddleware(path=path)
        system = SystemMessage("sys")
        human = HumanMessage("first", id="h1")
        ai = _ai(
            "",
            id="a1",
            tool_calls=[{"id": "c1", "name": "_grep", "args": {"pattern": "x"}}],
        )
        tool_msg = ToolMessage(content="hit", tool_call_id="c1", name="_grep", id="t1")
        with tracer.run():
            tracer.wrap_model_call(
                _model_request([human], [_grep], system_message=system),
                lambda r: ModelResponse(result=[ai]),
            )
            # Drop the original human message — compaction.
            tracer.wrap_model_call(
                _model_request(
                    [ai, tool_msg, HumanMessage("retry", id="h2")],
                    [_grep],
                    system_message=system,
                ),
                lambda r: ModelResponse(result=[_ai("done", id="a2")]),
            )
        events = read_events(path)
        assert_log_reconstructs(events)
        changed = next(e for e in events if e["type"] == "context_changed")
        assert changed["kind"] == "compaction"
        prompt_after = next(
            e for e in events if e["type"] == "prompt" and e["seq"] > changed["seq"]
        )
        assert prompt_after["messages"]["mode"] == "full"
        messages = reconstruct_messages(events, prompt_after["turn"])
        assert [m["role"] for m in messages] == ["ai", "tool", "human"]
        assert messages[2]["content"] == "retry"

    def test_missing_catalog_body_fails_reconstruction(self, tmp_path: Path) -> None:
        """A log that references an unseen catalogue digest is non-conforming."""
        path = tmp_path / "mvl.jsonl"
        tracer = ModelViewLogMiddleware(path=path)
        with tracer.run():
            tracer.wrap_model_call(
                _model_request([HumanMessage("hi")], [_grep]),
                lambda r: ModelResponse(result=[_ai("ok")]),
            )
        events = read_events(path)
        # Strip the catalogue body while leaving the prompt's digest reference.
        stripped = [e for e in events if e["type"] != "tool_catalog"]
        with pytest.raises(ReconstructionError, match="tool_catalog body"):
            reconstruct_tool_catalog(stripped, 1)

    def test_missing_system_text_fails_reconstruction(self, tmp_path: Path) -> None:
        """A log that hashes a system prompt it never wrote in full is non-conforming."""
        path = tmp_path / "mvl.jsonl"
        tracer = ModelViewLogMiddleware(path=path)
        with tracer.run():
            tracer.wrap_model_call(
                _model_request(
                    [HumanMessage("hi")],
                    [],
                    system_message=SystemMessage("secret system"),
                ),
                lambda r: ModelResponse(result=[_ai("ok")]),
            )
        events = read_events(path)
        # Null out the only full system text.
        for event in events:
            if event["type"] == "prompt" and (event.get("system") or {}).get("text"):
                event["system"]["text"] = None
        with pytest.raises(ReconstructionError, match="system text"):
            reconstruct_system_text(events, 1)


class TestCrashSurvivalAndOrdering:
    """Spec conformance #2 and #3."""

    def test_partial_log_is_valid_jsonl_without_run_ended(self, tmp_path: Path) -> None:
        """Kill the process mid-run: every complete line still parses and reconstructs."""
        path = tmp_path / "mvl.jsonl"
        tracer = ModelViewLogMiddleware(path=path)
        tracer.start_run()
        _drive_two_turn_tool_loop(tracer)
        # No end_run — the process "crashed". Events already flushed.
        tracer.close()
        raw = path.read_text(encoding="utf-8")
        # Append a truncated trailing line the way a killed writer would.
        with path.open("a", encoding="utf-8") as fh:
            fh.write('{"v":1,"type":"prompt","seq":99')
        events = read_events(path)
        assert events  # truncated line skipped
        assert all(isinstance(e, dict) and "seq" in e for e in events)
        assert [e["seq"] for e in events] == list(range(len(events)))
        assert_log_reconstructs(events)
        # The on-disk raw form is one JSON object per LF-terminated line (until the tail).
        complete_lines = [line for line in raw.splitlines() if line]
        assert all(json.loads(line)["v"] == 1 for line in complete_lines)

    def test_seq_monotonic_across_full_run(self, tmp_path: Path) -> None:
        """seq starts at 0 and advances by one per event with no gaps."""
        path = tmp_path / "mvl.jsonl"
        tracer = ModelViewLogMiddleware(path=path, run="fixed-run")
        with tracer.run():
            _drive_two_turn_tool_loop(tracer)
        events = read_events(path)
        assert events[0]["run"] == "fixed-run"
        assert [e["seq"] for e in events] == list(range(len(events)))


class TestToolHonestyAndWithdrawal:
    """Spec conformance #6 and #7."""

    def test_content_shown_byte_equals_tool_message(self, tmp_path: Path) -> None:
        """tool_result.content_shown is exactly the string the tool layer returned."""
        path = tmp_path / "mvl.jsonl"
        tracer = ModelViewLogMiddleware(path=path)
        shown = "crates/acp-bridge/src/main.rs:412: handle_request"
        with tracer.run():
            ai = _ai(
                "",
                tool_calls=[{"id": "c1", "name": "_grep", "args": {"pattern": "x"}}],
            )
            tracer.wrap_model_call(
                _model_request([HumanMessage("go")], [_grep]),
                lambda r: ModelResponse(result=[ai]),
            )
            request = ToolCallRequest(
                tool=_grep,
                tool_call={"id": "c1", "name": "_grep", "args": {"pattern": "x"}},
                state={},
                runtime=None,
            )
            tracer.wrap_tool_call(
                request,
                lambda r: ToolMessage(content=shown, tool_call_id="c1", name="_grep"),
            )
        events = read_events(path)
        by_id = tool_result_content_by_call_id(events)
        assert by_id["c1"] == shown
        # The same string reappears as the tool message content in the next prompt
        # when the harness threads it back — here we assert the tool_result itself.
        result = next(e for e in events if e["type"] == "tool_result")
        assert result["ok"] is True
        assert result["call_id"] == "c1"
        assert result["turn"] == 1

    def test_tools_changed_on_withdrawal(self, tmp_path: Path) -> None:
        """A withdrawn tool is explicit: tools_changed, and tools_offered shrinks."""
        path = tmp_path / "mvl.jsonl"
        tracer = ModelViewLogMiddleware(path=path)
        ai = _ai(
            "",
            tool_calls=[{"id": "c1", "name": "_grep", "args": {"pattern": "x"}}],
        )
        with tracer.run():
            tracer.wrap_model_call(
                _model_request([HumanMessage("go")], [_grep, _edit_file]),
                lambda r: ModelResponse(result=[ai]),
            )
            tracer.wrap_model_call(
                _model_request(
                    [
                        HumanMessage("go"),
                        ai,
                        ToolMessage(content="hit", tool_call_id="c1", name="_grep"),
                    ],
                    [_grep],  # _edit_file withdrawn
                ),
                lambda r: ModelResponse(result=[_ai("done")]),
            )
        events = read_events(path)
        assert_log_reconstructs(events)
        changed = next(e for e in events if e["type"] == "tools_changed")
        assert changed["removed"] == ["_edit_file"]
        assert changed["added"] == []
        turn2 = reconstruct_turn(events, 2)
        assert turn2["tools_offered"] == ["_grep"]


class TestCreateDeepAgentIntegration:
    """End-to-end: MVL through create_deep_agent's full middleware stack."""

    def test_create_deep_agent_multi_turn_reconstructs(self, tmp_path: Path) -> None:
        """A real agent loop produces a log that reconstructs every turn.

        The tracer is listed *first* in the caller middleware so relocation to
        the innermost slot is exercised. The fake model runs a tool call then a
        final answer; the log must recover system text, the tool catalogue
        (including schemas), full/delta messages with tool_calls, and the tool
        result shown back to the model.
        """
        path = tmp_path / "mvl.jsonl"
        tracer = ModelViewLogMiddleware(
            path=path,
            harness_name="deepagents",
            harness_version="test",
            model_id="fake-model",
            model_provider="fake",
            task_id="e2e",
            task_text="echo hi",
        )
        model = GenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="calling echo",
                        tool_calls=[
                            {
                                "name": "_echo",
                                "args": {"text": "hi"},
                                "id": "call_echo_1",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(content="echo returned hi"),
                ]
            )
        )
        # Intentionally first so create_deep_agent must relocate it innermost.
        agent = create_deep_agent(
            model=model,
            tools=[_echo],
            system_prompt="You are a concise assistant.",
            middleware=[tracer],
        )
        with tracer.run(task_id="e2e", task_text="echo hi"):
            result = agent.invoke({"messages": [HumanMessage(content="say hi")]})

        assert "messages" in result
        events = read_events(path)
        types = [e["type"] for e in events]
        assert types[0] == "run_started"
        assert types[-1] == "run_ended"
        assert events[-1]["outcome"] == "succeeded"
        assert "tool_catalog" in types
        assert "prompt" in types
        assert "completion" in types
        assert "tool_result" in types

        # Envelope + reconstruction for every turn the producer recorded.
        assert_log_reconstructs(events)
        present = turns_present(events)
        assert present  # at least one model turn
        assert present == sorted(present)

        for turn in present:
            view = reconstruct_turn(events, turn)
            # System prompt is recoverable and matches what we configured.
            assert view["system_text"] is not None
            assert "concise assistant" in view["system_text"]
            # Catalogue is complete ordered definitions with schemas.
            assert isinstance(view["tool_catalog"], list)
            assert view["tool_catalog"], "tool catalogue must not be empty"
            for entry in view["tool_catalog"]:
                assert "name" in entry
                assert "description" in entry
                assert isinstance(entry.get("input_schema"), dict)
            echo_defs = [t for t in view["tool_catalog"] if t["name"] == "_echo"]
            if echo_defs:
                props = echo_defs[0]["input_schema"].get("properties", {})
                assert "text" in props
            # Messages rebuild to a non-empty list with roles.
            assert view["messages"]
            assert all("role" in m and "content" in m for m in view["messages"])
            # tools_offered is request-time and non-empty for this agent.
            assert view["tools_offered"]

        # Tool honesty: content_shown equals what the model later saw in messages.
        by_id = tool_result_content_by_call_id(events)
        assert "call_echo_1" in by_id
        assert by_id["call_echo_1"] == "hi"
        # The tool message content is also present in some reconstructed turn.
        found_tool_content = False
        for turn in present:
            for item in reconstruct_messages(events, turn):
                if item.get("role") == "tool" and item.get("content") == "hi":
                    assert item.get("tool_call_id") == "call_echo_1"
                    found_tool_content = True
        assert found_tool_content, "reconstructed messages must include the tool result"

        # Completions carry usage.cached_input when the provider reports it
        # (GenericFakeChatModel may leave usage empty — require the field shape
        # when a completion has usage at all).
        for completion in (e for e in events if e["type"] == "completion"):
            assert "text" in completion
            assert "tool_calls" in completion
            assert "finish_reason" in completion
            assert "usage" in completion
            usage = completion["usage"]
            assert {"input", "cached_input", "output", "reasoning"} <= set(usage)

        # First prompt is full; a subsequent prompt after the tool turn is delta
        # (or full if some middleware rewrote history — either is reconstructible).
        prompts = [e for e in events if e["type"] == "prompt"]
        assert prompts[0]["messages"]["mode"] == "full"
        if len(prompts) > 1:
            assert prompts[1]["messages"]["mode"] in {"full", "delta"}

    def test_create_deep_agent_aborted_on_exception(self, tmp_path: Path) -> None:
        """An exception inside run() marks the log aborted, not failed."""
        path = tmp_path / "mvl.jsonl"
        tracer = ModelViewLogMiddleware(path=path, harness_name="deepagents")
        model = GenericFakeChatModel(messages=iter([AIMessage(content="ok")]))
        agent = create_deep_agent(model=model, tools=[_echo], middleware=[tracer])
        with pytest.raises(RuntimeError, match="boom"), tracer.run():
            agent.invoke({"messages": [HumanMessage(content="hi")]})
            msg = "boom"
            raise RuntimeError(msg)
        events = read_events(path)
        assert events[-1]["type"] == "run_ended"
        assert events[-1]["outcome"] == "aborted"
        assert events[-1]["reason"] == "boom"
        # Everything up to the abort still reconstructs.
        if turns_present(events):
            assert_log_reconstructs(events)

    def test_join_integrity_tool_result_turn_matches_completion(self, tmp_path: Path) -> None:
        """tool_result.turn / call_id join to the completion that issued the call."""
        path = tmp_path / "mvl.jsonl"
        tracer = ModelViewLogMiddleware(path=path)
        model = GenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "_echo",
                                "args": {"text": "z"},
                                "id": "call_z",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(content="done"),
                ]
            )
        )
        agent = create_deep_agent(model=model, tools=[_echo], middleware=[tracer])
        with tracer.run():
            agent.invoke({"messages": [HumanMessage(content="z")]})
        events = read_events(path)
        completions = [e for e in events if e["type"] == "completion"]
        results = [e for e in events if e["type"] == "tool_result"]
        assert results
        for result in results:
            call_id = result["call_id"]
            # Exactly one completion issued this call_id.
            issuers = [
                c
                for c in completions
                if any(tc.get("id") == call_id for tc in (c.get("tool_calls") or []))
            ]
            assert len(issuers) == 1, f"call_id {call_id} must join to one completion"
            assert result["turn"] == issuers[0]["turn"]

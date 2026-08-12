"""Unit tests for the Model View Log tracer middleware."""

import json

import pytest
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from deepagents.tracing import ModelViewLogMiddleware, read_events
from deepagents.tracing.emitter import sha256_bytes


class _FakeChatModel:
    """Minimal stand-in for a chat model, exposing sampling knobs."""

    model_name = "fake-model"
    temperature = 0.0
    max_tokens = 1024


@tool
def _grep(pattern: str) -> str:
    """Search files for a pattern."""
    return "hit:" + pattern


@tool
def _edit_file(path: str) -> str:
    """Edit a file."""
    return "edited:" + path


def _read(path):
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line:
            events.append(json.loads(line))
    return events


def _model_request(messages, tools, system_message=None):
    return ModelRequest(
        model=_FakeChatModel(),
        messages=messages,
        system_message=system_message,
        tools=tools,
    )


def _ai(content, *, tool_calls=None, usage=None, finish=None):
    return AIMessage(
        content=content,
        tool_calls=tool_calls or [],
        usage_metadata=usage or {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        response_metadata={"stop_reason": finish or ("tool_use" if tool_calls else "end_turn")},
    )


class TestModelViewLogMiddleware:
    """Test the per-turn MVL events emitted by the middleware."""

    def test_run_started_carries_metadata(self, tmp_path) -> None:
        """run_started reports harness, model, task, repo and resolved config."""
        tracer = ModelViewLogMiddleware(
            path=tmp_path / "log.jsonl",
            harness_name="deepagents",
            harness_version="1.0.0",
            model_id="fake-model",
            model_provider="fake",
            task_id="T1",
            task_text="do it",
            repo_commit="abc123",
            repo_dirty=False,
            config={"loop.doom_threshold": 3},
        )
        tracer.start_run()
        tracer.close()
        started = read_events(tmp_path / "log.jsonl")[0]
        assert started["harness"] == {"name": "deepagents", "version": "1.0.0"}
        assert started["model"] == {"id": "fake-model", "provider": "fake"}
        assert started["task"] == {"id": "T1", "text": "do it"}
        assert started["repo"] == {"commit": "abc123", "dirty": False}
        assert started["config"] == {"loop.doom_threshold": 3}

    def test_first_prompt_is_full_and_second_is_delta(self, tmp_path) -> None:
        """A full prompt is followed by a delta of only the appended messages."""
        tracer = ModelViewLogMiddleware(path=tmp_path / "log.jsonl")
        tracer.start_run()
        ai1 = _ai("", tool_calls=[{"id": "c1", "name": "_grep", "args": {"pattern": "x"}}])
        tracer.wrap_model_call(
            _model_request([HumanMessage("go")], [_grep]),
            lambda r: ModelResponse(result=[ai1]),
        )
        tracer.wrap_model_call(
            _model_request(
                [HumanMessage("go"), ai1, ToolMessage(content="hit:x", tool_call_id="c1", name="_grep")],
                [_grep],
            ),
            lambda r: ModelResponse(result=[_ai("done")]),
        )
        tracer.close()
        prompts = [e for e in read_events(tmp_path / "log.jsonl") if e["type"] == "prompt"]
        assert [p["messages"]["mode"] for p in prompts] == ["full", "delta"]
        delta_roles = [item["role"] for item in prompts[1]["messages"]["items"]]
        assert delta_roles == ["ai", "tool"]

    def test_system_text_emitted_once_per_hash(self, tmp_path) -> None:
        """The full system text appears once; later prompts carry only the hash."""
        from langchain_core.messages import SystemMessage

        tracer = ModelViewLogMiddleware(path=tmp_path / "log.jsonl")
        tracer.start_run()
        sys = SystemMessage("you are helpful")
        tracer.wrap_model_call(
            _model_request([HumanMessage("hi")], [], system_message=sys),
            lambda r: ModelResponse(result=[_ai("a")]),
        )
        tracer.wrap_model_call(
            _model_request([HumanMessage("hi"), _ai("a")], [], system_message=sys),
            lambda r: ModelResponse(result=[_ai("b")]),
        )
        tracer.close()
        prompts = [e for e in read_events(tmp_path / "log.jsonl") if e["type"] == "prompt"]
        assert prompts[0]["system"]["text"] == "you are helpful"
        assert prompts[1]["system"]["text"] is None
        assert prompts[1]["system"]["sha256"] == prompts[0]["system"]["sha256"]

    def test_prompt_carries_catalog_hash_and_sampling_params(self, tmp_path) -> None:
        """Prompt records the tool catalogue hash, offered tools and params."""
        tracer = ModelViewLogMiddleware(path=tmp_path / "log.jsonl")
        tracer.start_run()
        tracer.wrap_model_call(
            _model_request([HumanMessage("hi")], [_grep, _edit_file]),
            lambda r: ModelResponse(result=[_ai("ok")]),
        )
        tracer.close()
        prompt = next(e for e in read_events(tmp_path / "log.jsonl") if e["type"] == "prompt")
        assert prompt["tool_catalog_sha256"]
        assert prompt["tools_offered"] == ["_grep", "_edit_file"]
        assert prompt["params"] == {"temperature": 0.0, "max_tokens": 1024}

    def test_tool_catalog_emitted_once_per_distinct_digest(self, tmp_path) -> None:
        """A stable tool list emits a single tool_catalog, referenced by digest."""
        tracer = ModelViewLogMiddleware(path=tmp_path / "log.jsonl")
        tracer.start_run()
        ai1 = _ai("", tool_calls=[{"id": "c1", "name": "_grep", "args": {"pattern": "x"}}])
        tracer.wrap_model_call(
            _model_request([HumanMessage("go")], [_grep, _edit_file]),
            lambda r: ModelResponse(result=[ai1]),
        )
        tracer.wrap_model_call(
            _model_request([HumanMessage("go"), ai1, ToolMessage(content="hit", tool_call_id="c1", name="_grep")], [_grep, _edit_file]),
            lambda r: ModelResponse(result=[_ai("done")]),
        )
        tracer.close()
        catalogs = [e for e in read_events(tmp_path / "log.jsonl") if e["type"] == "tool_catalog"]
        assert len(catalogs) == 1
        assert [t["name"] for t in catalogs[0]["tools"]] == ["_grep", "_edit_file"]
        assert all(e["tool_catalog_sha256"] == catalogs[0]["sha256"] for e in read_events(tmp_path / "log.jsonl") if e["type"] == "prompt")

    def test_completion_records_text_tool_calls_finish_and_usage(self, tmp_path) -> None:
        """Completion captures what the model produced, including cached_input."""
        tracer = ModelViewLogMiddleware(path=tmp_path / "log.jsonl")
        tracer.start_run()
        ai = _ai(
            "check",
            tool_calls=[{"id": "c1", "name": "_grep", "args": {"pattern": "x"}}],
            usage={
                "input_tokens": 100,
                "output_tokens": 20,
                "total_tokens": 120,
                "input_token_details": {"cached_tokens": 90},
                "output_token_details": {"reasoning_tokens": 5},
            },
            finish="tool_use",
        )
        tracer.wrap_model_call(_model_request([HumanMessage("go")], [_grep]), lambda r: ModelResponse(result=[ai]))
        tracer.close()
        completion = next(e for e in read_events(tmp_path / "log.jsonl") if e["type"] == "completion")
        assert completion["text"] == "check"
        assert completion["tool_calls"] == [{"id": "c1", "name": "_grep", "arguments": {"pattern": "x"}}]
        assert completion["finish_reason"] == "tool_use"
        assert completion["usage"] == {
            "input": 100,
            "cached_input": 90,
            "output": 20,
            "reasoning": 5,
        }

    def test_tool_result_reports_ok_and_error(self, tmp_path) -> None:
        """tool_result carries the shown content, ok flag, call_id and initiating turn."""
        tracer = ModelViewLogMiddleware(path=tmp_path / "log.jsonl")
        tracer.start_run()
        ai = _ai("", tool_calls=[{"id": "c1", "name": "_grep", "args": {"pattern": "x"}}])
        tracer.wrap_model_call(_model_request([HumanMessage("go")], [_grep]), lambda r: ModelResponse(result=[ai]))
        request = ToolCallRequest(
            tool=_grep,
            tool_call={"id": "c1", "name": "_grep", "args": {"pattern": "x"}},
            state={},
            runtime=None,
        )
        tracer.wrap_tool_call(request, lambda r: ToolMessage(content="Error: no match", tool_call_id="c1", name="_grep"))
        tracer.close()
        result = next(e for e in read_events(tmp_path / "log.jsonl") if e["type"] == "tool_result")
        assert result["call_id"] == "c1"
        assert result["name"] == "_grep"
        assert result["ok"] is False
        assert result["content_shown"] == "Error: no match"
        assert result["turn"] == 1
        assert result["duration_ms"] >= 0

    def test_tools_changed_emitted_on_withdrawal(self, tmp_path) -> None:
        """A changed offered tool set is surfaced as tools_changed with removed/added."""
        tracer = ModelViewLogMiddleware(path=tmp_path / "log.jsonl")
        tracer.start_run()
        ai = _ai("", tool_calls=[{"id": "c1", "name": "_grep", "args": {"pattern": "x"}}])
        tracer.wrap_model_call(_model_request([HumanMessage("go")], [_grep, _edit_file]), lambda r: ModelResponse(result=[ai]))
        tracer.wrap_model_call(
            _model_request(
                [HumanMessage("go"), ai, ToolMessage(content="hit", tool_call_id="c1", name="_grep")],
                [_grep],
            ),
            lambda r: ModelResponse(result=[_ai("done")]),
        )
        tracer.close()
        changed = next(e for e in read_events(tmp_path / "log.jsonl") if e["type"] == "tools_changed")
        assert changed["removed"] == ["_edit_file"]
        assert changed["added"] == []

    def test_context_changed_compaction_forces_full_prompt(self, tmp_path) -> None:
        """When messages are removed, context_changed fires and the next prompt is full."""
        tracer = ModelViewLogMiddleware(path=tmp_path / "log.jsonl")
        tracer.start_run()
        ai = _ai("", tool_calls=[{"id": "c1", "name": "_grep", "args": {"pattern": "x"}}])
        tracer.wrap_model_call(_model_request([HumanMessage("first")], [_grep]), lambda r: ModelResponse(result=[ai]))
        # The "first" human message is dropped; the rest are new -> compaction.
        tracer.wrap_model_call(
            _model_request([ai, ToolMessage(content="hit", tool_call_id="c1", name="_grep"), HumanMessage("retry")], [_grep]),
            lambda r: ModelResponse(result=[_ai("done")]),
        )
        tracer.close()
        events = read_events(tmp_path / "log.jsonl")
        idx = next(i for i, e in enumerate(events) if e["type"] == "context_changed")
        assert events[idx]["kind"] == "compaction"
        assert events[idx + 1]["type"] == "prompt"
        assert events[idx + 1]["messages"]["mode"] == "full"

    def test_run_context_manager_emits_aborted_on_exception(self, tmp_path) -> None:
        """run() emits run_ended aborted when the body raises, else succeeded."""
        tracer = ModelViewLogMiddleware(path=tmp_path / "log.jsonl")
        with pytest.raises(RuntimeError), tracer.run(task_id="T1"):
            msg = "boom"
            raise RuntimeError(msg)
        events = read_events(tmp_path / "log.jsonl")
        assert events[0]["type"] == "run_started"
        assert events[-1]["type"] == "run_ended"
        assert events[-1]["outcome"] == "aborted"
        assert events[-1]["reason"] == "boom"

    def test_run_context_manager_aborts_on_base_exception(self, tmp_path) -> None:
        """Even a KeyboardInterrupt emits run_ended aborted before propagating."""
        tracer = ModelViewLogMiddleware(path=tmp_path / "log.jsonl")
        with pytest.raises(KeyboardInterrupt), tracer.run():
            raise KeyboardInterrupt
        events = read_events(tmp_path / "log.jsonl")
        assert events[-1]["type"] == "run_ended"
        assert events[-1]["outcome"] == "aborted"

    def test_tool_result_offload_ref_hashes_raw_bytes(self, tmp_path) -> None:
        """full_content digest is over the actual file bytes, with the true size."""
        evicted = tmp_path / "evicted.txt"
        evicted.write_bytes(b"x" * 1000)
        tracer = ModelViewLogMiddleware(path=tmp_path / "log.jsonl")
        tracer.start_run()
        ai = _ai("", tool_calls=[{"id": "c1", "name": "_grep", "args": {"pattern": "x"}}])
        tracer.wrap_model_call(_model_request([HumanMessage("go")], [_grep]), lambda r: ModelResponse(result=[ai]))
        tm = ToolMessage(content="evicted", tool_call_id="c1", name="_grep")
        tm.additional_kwargs["lc_evicted_to"] = str(evicted)
        request = ToolCallRequest(
            tool=_grep,
            tool_call={"id": "c1", "name": "_grep", "args": {"pattern": "x"}},
            state={},
            runtime=None,
        )
        tracer.wrap_tool_call(request, lambda r: tm)
        tracer.close()
        result = next(e for e in read_events(tmp_path / "log.jsonl") if e["type"] == "tool_result")
        assert result["offloaded"] is True
        assert result["full_content"]["ref"] == "sha256:" + sha256_bytes(b"x" * 1000)
        assert result["full_content"]["bytes"] == 1000
        assert result["full_content"]["path"] == str(evicted)

    def test_run_context_manager_succeeded(self, tmp_path) -> None:
        """run() emits run_ended succeeded when the body completes."""
        tracer = ModelViewLogMiddleware(path=tmp_path / "log.jsonl")
        with tracer.run(task_id="T1"):
            pass
        events = read_events(tmp_path / "log.jsonl")
        assert events[-1]["outcome"] == "succeeded"

    def test_missing_run_ended_is_reader_inference(self, tmp_path) -> None:
        """A log that ends without run_ended still has parseable, ordered events."""
        tracer = ModelViewLogMiddleware(path=tmp_path / "log.jsonl")
        tracer.start_run()
        tracer.wrap_model_call(_model_request([HumanMessage("hi")], [_grep]), lambda r: ModelResponse(result=[_ai("ok")]))
        # No end_run call; the process "crashes" here.
        tracer.close()
        events = read_events(tmp_path / "log.jsonl")
        assert [e["seq"] for e in events] == list(range(len(events)))
        assert events[-1]["type"] != "run_ended"


class TestModelViewLogMiddlewareAsync:
    """Test the async hooks mirror the sync behaviour."""

    async def test_awrap_model_call_and_awrap_tool_call(self, tmp_path) -> None:
        """The async variants emit the same event set as their sync counterparts."""
        tracer = ModelViewLogMiddleware(path=tmp_path / "log.jsonl")
        tracer.start_run()
        ai = _ai("", tool_calls=[{"id": "c1", "name": "_grep", "args": {"pattern": "x"}}])

        async def amodel(r):
            return ModelResponse(result=[ai])

        await tracer.awrap_model_call(_model_request([HumanMessage("go")], [_grep]), amodel)

        request = ToolCallRequest(
            tool=_grep,
            tool_call={"id": "c1", "name": "_grep", "args": {"pattern": "x"}},
            state={},
            runtime=None,
        )

        async def atool(r):
            return ToolMessage(content="hit", tool_call_id="c1", name="_grep")

        await tracer.awrap_tool_call(request, atool)
        tracer.close()
        types = [e["type"] for e in read_events(tmp_path / "log.jsonl")]
        assert "prompt" in types
        assert "completion" in types
        assert "tool_result" in types

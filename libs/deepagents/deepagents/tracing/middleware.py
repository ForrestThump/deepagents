"""Model View Log (MVL) tracer as an ``AgentMiddleware``.

The tracer records, at the executor / provider boundary, exactly what the model
was sent and what it produced, following the MVL v1 spec: the prompt (with a
full/delta message stream), the ordered tool catalogue, the completion, the
tool results shown back, and the two "drift" events (`tools_changed`,
`context_changed`) that explain why the offered tool set or the visible history
changed between turns.

The middleware is written to run *innermost* in the model-call stack so it sees
the final request after every other middleware (filesystem filtering, memory
injection, prompt caching, summarization) has transformed it, and the raw
completion the provider returned. ``create_deep_agent`` relocates instances of
this class to the innermost slot automatically.

## Framing

Per-turn events (`prompt`, `completion`, `tool_result`, `tool_catalog`,
`tools_changed`, `context_changed`) are emitted automatically. The run-level
events are framed explicitly so their `harness` / `task` / `repo` / `config`
metadata stays accurate:

```python
from deepagents.tracing import ModelViewLogMiddleware

tracer = ModelViewLogMiddleware(path="model_view.jsonl", harness_name="deepagents", harness_version="1.0.0")
agent = create_deep_agent(model=..., middleware=[tracer])

with tracer.run(task_id="P3.2", task_text="...") as run:
    result = agent.invoke({"messages": [{"role": "user", "content": "hi"}]})
```

`run()` emits `run_started`, yields, and emits `run_ended` (or `aborted` if the
body raises). A run that is not framed this way still emits a `run_started` on
its first model call; if it then crashes, the missing `run_ended` tells a reader
to treat it as `aborted`, which is exactly the correct inference.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Self

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ContextT,
    ExtendedModelResponse,
    ModelRequest,
    ModelResponse,
    ResponseT,
)
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from deepagents.tracing.emitter import (
    ModelViewLogger,
    sha256_bytes,
    sha256_json,
)

if TYPE_CHECKING:
    from langchain.tools.tool_node import ToolCallRequest
    from langchain_core.tools import BaseTool
    from langgraph.types import Command

    from deepagents.tracing.emitter import ModelViewLogger as _ModelViewLogger


@dataclass
class ModelViewLogConfig:
    """Resolved tracing knobs for a :class:`ModelViewLogMiddleware`.

    Args:
        path: Filesystem path of the JSONL log.
        offload_dir: Optional directory for content-addressed ``full_content``
            refs.
        run: Optional explicit run id. A random one is generated otherwise.
        harness_name: Name reported in ``run_started.harness``.
        harness_version: Version reported in ``run_started.harness``.
        model_id: Override the model id reported in ``run_started.model``.
        model_provider: Override the provider reported in ``run_started.model``.
        task_id: Id reported in ``run_started.task``.
        task_text: Text reported in ``run_started.task``.
        repo_commit: Commit reported in ``run_started.repo``.
        repo_dirty: Dirty flag reported in ``run_started.repo``.
        config: Resolved knob values reported in ``run_started.config``.
    """

    path: str | Path
    offload_dir: str | Path | None = None
    run: str | None = None
    harness_name: str | None = None
    harness_version: str | None = None
    model_id: str | None = None
    model_provider: str | None = None
    task_id: str | None = None
    task_text: str | None = None
    repo_commit: str | None = None
    repo_dirty: bool | None = None
    config: dict[str, Any] | None = None


@dataclass
class _StreamState:
    """Mutable per-run bookkeeping used to emit full/delta and drift events."""

    turn: int = 0
    started: bool = False
    last_keys: list[str] | None = None
    last_offered: list[str | None] | None = None
    catalog_digests: set[str] = field(default_factory=set)
    seen_system: set[str] = field(default_factory=set)
    call_to_turn: dict[str, int] = field(default_factory=dict)


def _content_to_str(content: object) -> str:
    """Render a message ``content`` (string or block list) to its text form."""
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, Mapping):
                if block.get("text") is not None:
                    parts.append(str(block["text"]))
            elif getattr(block, "text", None):
                parts.append(str(block.text))
        return "\n".join(parts)
    return str(content)


def _message_key(msg: BaseMessage) -> str:
    """Return a stable identity for a message used to diff full/delta streams."""
    msg_id = getattr(msg, "id", None)
    if msg_id:
        return f"id:{msg_id}"
    tool_call_id = getattr(msg, "tool_call_id", None) or ""
    content = _content_to_str(getattr(msg, "content", ""))
    return f"{msg.type}:{tool_call_id}:{sha256_bytes(content.encode('utf-8'))}"


def _serialize_content(content: object) -> Any:
    """Serialize message content for the log without collapsing structure.

    Strings stay strings. Block lists stay lists of JSON-friendly dicts so a
    reader can rebuild the exact payload the provider received (text, image
    placeholders, etc.), not a lossy concatenation of text fragments.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        blocks: list[Any] = []
        for block in content:
            if isinstance(block, str):
                blocks.append(block)
            elif isinstance(block, Mapping):
                blocks.append(dict(block))
            else:
                dump = getattr(block, "model_dump", None)
                if callable(dump):
                    try:
                        blocks.append(dump(mode="json"))
                        continue
                    except TypeError:
                        pass
                text = getattr(block, "text", None)
                if text is not None:
                    blocks.append({"type": "text", "text": str(text)})
                else:
                    blocks.append(str(block))
        return blocks
    return str(content)


def _message_item(msg: BaseMessage) -> dict[str, Any]:
    """Serialize a message for the prompt stream.

    The reconstruction checklist rebuilds the message list from ``prompt``
    events alone, so each item must carry every field that changes model
    behaviour: role, content, tool calls (on AI messages), and the
    ``tool_call_id`` / name pair (on tool messages).
    """
    item: dict[str, Any] = {
        "role": msg.type,
        "content": _serialize_content(getattr(msg, "content", "")),
    }
    msg_id = getattr(msg, "id", None)
    if msg_id:
        item["id"] = msg_id
    tool_calls = getattr(msg, "tool_calls", None) or None
    if tool_calls:
        item["tool_calls"] = [
            {
                "id": call.get("id"),
                "name": call.get("name"),
                "arguments": call.get("args", {}),
            }
            for call in tool_calls
        ]
    tool_call_id = getattr(msg, "tool_call_id", None)
    if tool_call_id:
        item["tool_call_id"] = tool_call_id
    name = getattr(msg, "name", None)
    if name:
        item["name"] = name
    return item


def _tool_name(tool: BaseTool | dict[str, Any]) -> str | None:
    """Extract a tool's name from a `BaseTool` or dict tool."""
    if isinstance(tool, dict):
        name = tool.get("name")
        return name if isinstance(name, str) else None
    name = getattr(tool, "name", None)
    return name if isinstance(name, str) else None


def _json_schema_from_tool(tool: BaseTool) -> dict[str, Any]:
    """Return the JSON Schema the provider receives for ``tool``.

    Prefers ``tool_call_schema.model_json_schema()`` — the same shape LangChain
    binds into the request — and falls back to ``get_input_schema()`` when the
    tool does not expose ``tool_call_schema``. Never calls ``model_dump`` on a
    Pydantic *class* (that raises and used to leave ``input_schema`` empty).
    """
    tcs = getattr(tool, "tool_call_schema", None)
    if tcs is not None:
        schema_fn = getattr(tcs, "model_json_schema", None)
        if callable(schema_fn):
            result = schema_fn()
            if isinstance(result, dict):
                return result
    resolved = tool.get_input_schema()
    schema_fn = getattr(resolved, "model_json_schema", None)
    if callable(schema_fn):
        result = schema_fn()
        if isinstance(result, dict):
            return result
    # Instance path only — a class's model_dump is unbound and TypeErrors.
    if not isinstance(resolved, type):
        dump = getattr(resolved, "model_dump", None)
        if callable(dump):
            result = dump(mode="json")
            if isinstance(result, dict):
                return result
    return {}


def _tool_definition(tool: BaseTool | dict[str, Any]) -> dict[str, Any]:
    """Return the complete tool definition offered to the provider."""
    if isinstance(tool, dict):
        schema = tool.get("input_schema") or tool.get("parameters") or tool.get("args_schema") or {}
        if not isinstance(schema, dict):
            # args_schema may be a Pydantic model class when tools are pre-dictified.
            schema_fn = getattr(schema, "model_json_schema", None)
            schema = schema_fn() if callable(schema_fn) else {}
        return {
            "name": tool.get("name"),
            "description": tool.get("description"),
            "input_schema": schema if isinstance(schema, dict) else {},
        }
    try:
        schema = _json_schema_from_tool(tool)
    except Exception:  # noqa: BLE001  # a malformed schema must not break tracing
        schema = {}
    return {
        "name": getattr(tool, "name", None),
        "description": getattr(tool, "description", None),
        "input_schema": schema,
    }


class ModelViewLogMiddleware(AgentMiddleware[Any, ContextT, Any]):
    """Middleware that records the model view to an MVL JSONL log.

    Args:
        path: Filesystem path of the JSONL log.
        **kwargs: Additional :class:`ModelViewLogConfig` knobs (e.g.
            ``harness_name``, ``task_id``). See that class for the full set.
    """

    def __init__(self, path: str | Path, **kwargs: Any) -> None:
        """Initialize the tracer, opening its JSONL log lazily.

        Args:
            path: Filesystem path of the JSONL log.
            **kwargs: Additional :class:`ModelViewLogConfig` knobs.
        """
        self._config = ModelViewLogConfig(path=path, **kwargs)
        self._logger = ModelViewLogger(
            self._config.path,
            offload_dir=self._config.offload_dir,
        )
        if self._config.run:
            self._logger.replace_run(self._config.run)
        self._state = _StreamState()

    @property
    def logger(self) -> _ModelViewLogger:
        """The underlying :class:`ModelViewLogger` writing to disk."""
        return self._logger

    def close(self) -> None:
        """Close the underlying log file handle, if open."""
        self._logger.close()

    # ------------------------------------------------------------------ framing

    def start_run(self, **overrides: Any) -> None:
        """Emit `run_started` and reset the per-run stream state.

        Args:
            **overrides: Keyword overrides for `model_id`, `model_provider`,
                `task_id`, `task_text`, and `config`, mirroring the
                :class:`ModelViewLogConfig` fields.
        """
        config = self._config
        harness: dict[str, str] = {}
        if config.harness_name:
            harness["name"] = config.harness_name
        if config.harness_version:
            harness["version"] = config.harness_version
        model: dict[str, Any] = {
            "id": overrides.get("model_id", config.model_id),
            "provider": overrides.get("model_provider", config.model_provider),
        }
        task: dict[str, Any] | None = None
        task_id = overrides.get("task_id", config.task_id)
        task_text = overrides.get("task_text", config.task_text)
        if task_id or task_text:
            task = {"id": task_id, "text": task_text}
        repo: dict[str, Any] | None = None
        if config.repo_commit is not None:
            repo = {"commit": config.repo_commit, "dirty": config.repo_dirty}
        self._logger.emit(
            "run_started",
            harness=harness or None,
            model=model,
            task=task,
            repo=repo,
            config=overrides.get("config", config.config),
        )
        self._state = _StreamState()
        self._state.started = True

    def end_run(
        self,
        *,
        outcome: Literal["succeeded", "failed", "cancelled", "aborted"] = "succeeded",
        reason: str | None = None,
        gates: list[dict[str, Any]] | None = None,
    ) -> None:
        """Emit `run_ended` with the given outcome.

        Args:
            outcome: One of `succeeded`, `failed`, `cancelled`, `aborted`.
            reason: Free-form reason, for example a failure message.
            gates: Optional list of gate results, e.g.
                ``[{"name": "cargo test", "passed": false}]``.
        """
        self._logger.emit("run_ended", outcome=outcome, reason=reason, gates=gates)

    @contextlib.contextmanager
    def run(self, **start_overrides: Any) -> Iterator[Self]:
        """Frame a run with `run_started`/`run_ended`.

        A raised body marks the run `aborted` with the exception message, which
        is the correct distinction between a crash and a decision.

        Args:
            **start_overrides: Overrides forwarded to :meth:`start_run`.

        Yields:
            The middleware itself.
        """
        self.start_run(**start_overrides)
        try:
            yield self
        except BaseException as exc:  # any crash, incl. KeyboardInterrupt, aborts the run
            self.end_run(outcome="aborted", reason=str(exc))
            raise
        else:
            self.end_run(outcome="succeeded")
        finally:
            self.close()

    # ------------------------------------------------------------ model events

    def _ensure_started(self, request: ModelRequest[ContextT]) -> None:
        if self._state.started:
            return
        model_id = self._config.model_id or self._model_id(request.model)
        provider = self._config.model_provider or self._provider(request.model)
        harness: dict[str, str] = {}
        if self._config.harness_name:
            harness["name"] = self._config.harness_name
        if self._config.harness_version:
            harness["version"] = self._config.harness_version
        task: dict[str, Any] | None = None
        if self._config.task_id or self._config.task_text:
            task = {"id": self._config.task_id, "text": self._config.task_text}
        repo: dict[str, Any] | None = None
        if self._config.repo_commit is not None:
            repo = {"commit": self._config.repo_commit, "dirty": self._config.repo_dirty}
        self._logger.emit(
            "run_started",
            harness=harness or None,
            model={"id": model_id, "provider": provider},
            task=task,
            repo=repo,
            config=self._config.config,
        )
        self._state.started = True

    def _emit_catalog(self, request: ModelRequest[ContextT]) -> str:
        tools = list(request.tools or [])
        definitions = [_tool_definition(tool) for tool in tools]
        digest = sha256_json(definitions)
        if digest not in self._state.catalog_digests:
            self._state.catalog_digests.add(digest)
            self._logger.emit("tool_catalog", sha256=digest, tools=definitions)
        return digest

    def _check_tools_changed(self, request: ModelRequest[ContextT]) -> None:
        offered = [_tool_name(tool) for tool in (request.tools or [])]
        previous = self._state.last_offered
        self._state.last_offered = offered
        if previous is None:
            return
        prev_set = {name for name in previous if name is not None}
        cur_set = {name for name in offered if name is not None}
        if prev_set == cur_set:
            return
        self._logger.emit(
            "tools_changed",
            turn=self._state.turn,
            removed=sorted(prev_set - cur_set),
            added=sorted(cur_set - prev_set),
            reason="offer",
        )

    def _classify_removed(self, current_messages: Sequence[BaseMessage]) -> str:
        for message in current_messages:
            if isinstance(message, ToolMessage) and (message.additional_kwargs or {}).get("lc_evicted_to"):
                return "eviction"
        return "compaction"

    def _prompt_mode(
        self,
        request: ModelRequest[ContextT],
    ) -> tuple[str, set[str] | None]:
        """Decide full vs delta and return (mode, previous message keys).

        ``last_keys`` is updated to the current turn, but the caller needs the
        *previous* key set to compute which messages are appended in delta mode,
        so it is returned here.
        """
        keys = [_message_key(message) for message in request.messages]
        previous = self._state.last_keys
        previous_set = set(previous) if previous is not None else None
        self._state.last_keys = keys
        if previous is None:
            return "full", None
        current_set = set(keys)
        removed = [key for key in previous if key not in current_set]
        if removed:
            kind = self._classify_removed(request.messages)
            self._logger.emit(
                "context_changed",
                turn=self._state.turn,
                kind=kind,
                removed_messages=len(removed),
                summary=None,
                details={},
            )
            return "full", None
        return "delta", previous_set

    def _emit_prompt(
        self,
        request: ModelRequest[ContextT],
        mode: str,
        catalog_digest: str,
        turn: int,
        previous_keys: set[str] | None = None,
    ) -> None:
        system_text = ""
        if request.system_message is not None:
            system_text = _content_to_str(request.system_message.content)
        sys_sha = sha256_bytes(system_text.encode("utf-8")) if system_text else None
        text: str | None = None
        if sys_sha is not None and sys_sha not in self._state.seen_system:
            self._state.seen_system.add(sys_sha)
            text = system_text
        if mode == "full":
            items = [_message_item(message) for message in request.messages]
        else:
            previous_set = previous_keys or set()
            items = [_message_item(message) for message in request.messages if _message_key(message) not in previous_set]
        offered = [_tool_name(tool) for tool in (request.tools or [])]
        self._logger.emit(
            "prompt",
            turn=turn,
            messages={"mode": mode, "items": items},
            system={"sha256": sys_sha, "text": text},
            tool_catalog_sha256=catalog_digest,
            tools_offered=offered,
            params=self._sampling_params(request),
        )

    @staticmethod
    def _sampling_params(request: ModelRequest[ContextT]) -> dict[str, Any]:
        params: dict[str, Any] = {}
        settings = request.model_settings or {}
        model = request.model
        temperature = getattr(model, "temperature", None)
        max_tokens = getattr(model, "max_tokens", None)
        if "temperature" in settings:
            params["temperature"] = settings["temperature"]
        elif temperature is not None:
            params["temperature"] = temperature
        if "max_tokens" in settings:
            params["max_tokens"] = settings["max_tokens"]
        elif max_tokens is not None:
            params["max_tokens"] = max_tokens
        return params

    @staticmethod
    def _coerce_messages(response: object) -> list[BaseMessage]:
        while hasattr(response, "model_response"):
            response = getattr(response, "model_response")  # noqa: B009  # attr is dynamic across response shapes
        if hasattr(response, "result"):
            return list(getattr(response, "result"))  # noqa: B009  # attr is dynamic across response shapes
        if isinstance(response, BaseMessage):
            return [response]
        return []

    def _emit_completion(self, response: object, turn: int) -> None:
        messages = self._coerce_messages(response)
        ai = next((m for m in messages if isinstance(m, AIMessage)), None)
        if ai is None:
            return
        tool_calls = [{"id": call.get("id"), "name": call.get("name"), "arguments": call.get("args", {})} for call in (ai.tool_calls or [])]
        for call in ai.tool_calls or []:
            call_id = call.get("id")
            if call_id:
                self._state.call_to_turn[call_id] = turn
        self._logger.emit(
            "completion",
            turn=turn,
            text=_content_to_str(ai.content),
            tool_calls=tool_calls,
            finish_reason=self._finish_reason(ai),
            usage=self._usage(ai),
        )

    @staticmethod
    def _finish_reason(ai: AIMessage) -> str:
        meta = getattr(ai, "response_metadata", {}) or {}
        reason = meta.get("stop_reason") or meta.get("finish_reason")
        if reason is not None:
            return str(reason)
        if ai.tool_calls:
            return "tool_calls"
        return "stop"

    @staticmethod
    def _usage(ai: AIMessage) -> dict[str, int]:
        metadata = ai.usage_metadata or {}
        input_details = metadata.get("input_token_details") or {}
        output_details = metadata.get("output_token_details") or {}
        cached = input_details.get("cached_tokens") or input_details.get("cache_read_input_tokens") or 0
        reasoning = output_details.get("reasoning_tokens") or 0
        return {
            "input": metadata.get("input_tokens", 0),
            "cached_input": cached,
            "output": metadata.get("output_tokens", 0),
            "reasoning": reasoning,
        }

    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT] | ExtendedModelResponse:
        """Record the prompt, then the completion, around the provider call.

        Args:
            request: The (final, post-transform) model request.
            handler: The inner model call.

        Returns:
            The model response returned by ``handler``.
        """
        self._ensure_started(request)
        self._state.turn += 1
        turn = self._state.turn
        catalog_digest = self._emit_catalog(request)
        self._check_tools_changed(request)
        mode, previous_keys = self._prompt_mode(request)
        self._emit_prompt(request, mode, catalog_digest, turn, previous_keys)
        response = handler(request)
        self._emit_completion(response, turn)
        return response

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]],
    ) -> ModelResponse[ResponseT] | ExtendedModelResponse:
        """Async variant of :meth:`wrap_model_call`."""
        self._ensure_started(request)
        self._state.turn += 1
        turn = self._state.turn
        catalog_digest = self._emit_catalog(request)
        self._check_tools_changed(request)
        mode, previous_keys = self._prompt_mode(request)
        self._emit_prompt(request, mode, catalog_digest, turn, previous_keys)
        response = await handler(request)
        self._emit_completion(response, turn)
        return response

    # --------------------------------------------------------------- tool events

    def _emit_tool_result(
        self,
        result: ToolMessage | Command[Any],
        call_id: str | None,
        name: str | None,
        turn: int,
        duration_ms: int,
    ) -> None:
        ok = True
        content_shown = ""
        full_content: dict[str, Any] | None = None
        offloaded = False
        if isinstance(result, ToolMessage):
            content_shown = _content_to_str(result.content)
            ok = self._tool_ok(result)
            evicted = (result.additional_kwargs or {}).get("lc_evicted_to")
            if isinstance(evicted, (str, Path)):
                full_content = self._offload_reference(Path(evicted))
                offloaded = full_content is not None
        self._logger.emit(
            "tool_result",
            turn=turn,
            call_id=call_id,
            name=name,
            ok=ok,
            content_shown=content_shown,
            full_content=full_content,
            truncated=False,
            offloaded=offloaded,
            duration_ms=duration_ms,
        )

    @staticmethod
    def _tool_ok(result: ToolMessage) -> bool:
        status_error = getattr(result, "status", None) == "error"
        content_error = _content_to_str(result.content).startswith("Error:")
        return not (status_error or content_error)

    @staticmethod
    def _offload_reference(path: Path) -> dict[str, Any] | None:
        """Build a content reference for an evicted tool result file.

        The digest is over the raw file bytes so two runs that offload identical
        content produce the same ref regardless of the file's path.

        Args:
            path: The file the harness offloaded the full content to.

        Returns:
            A ``{"ref", "bytes", "path"}`` dict, or ``None`` if the file is
            missing or unreadable.
        """
        try:
            data = path.read_bytes()
        except OSError:
            return None
        return {
            "ref": f"sha256:{sha256_bytes(data)}",
            "bytes": len(data),
            "path": str(path),
        }

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        """Record the tool result shown back to the model.

        Args:
            request: The tool call request being executed.
            handler: The inner tool execution.

        Returns:
            The tool result returned by ``handler``.
        """
        call = request.tool_call
        call_id = call.get("id")
        name = call.get("name")
        turn = self._state.call_to_turn.get(call_id or "", self._state.turn)
        started = time.perf_counter()
        result = handler(request)
        duration_ms = round((time.perf_counter() - started) * 1000)
        self._emit_tool_result(result, call_id, name, turn, duration_ms)
        return result

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        """Async variant of :meth:`wrap_tool_call`."""
        call = request.tool_call
        call_id = call.get("id")
        name = call.get("name")
        turn = self._state.call_to_turn.get(call_id or "", self._state.turn)
        started = time.perf_counter()
        result = await handler(request)
        duration_ms = round((time.perf_counter() - started) * 1000)
        self._emit_tool_result(result, call_id, name, turn, duration_ms)
        return result

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _model_id(model: object) -> str:
        for attr in ("model_name", "model", "deployment_name"):
            value = getattr(model, attr, None)
            if isinstance(value, str) and value:
                return value
        return type(model).__name__

    def _provider(self, model: object) -> str | None:
        if self._config.model_provider:
            return self._config.model_provider
        module = type(model).__module__
        for key in ("anthropic", "openai", "google_genai", "bedrock", "fireworks", "groq", "ollama", "together"):
            if key in module:
                return key
        return None


__all__ = ["ModelViewLogConfig", "ModelViewLogMiddleware"]

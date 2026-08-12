"""Reconstruct the model view at turn N from an MVL v1 JSONL log.

Implements the normative reconstruction checklist from the Model View Log
spec: from the log alone a reader must recover, for any turn present in the
log, the system text, ordered tool definitions, message list, sampling
parameters and tools offered on that request.

This module is the shared reader used by the conformance suite. Producers do
not call it — they emit. Readers and tests call it to prove the log is
"what the model sees", not a summary of it.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any


class ReconstructionError(ValueError):
    """Raised when a log cannot rebuild the model view for a requested turn."""


def _by_type(events: Sequence[Mapping[str, Any]], type_: str) -> list[Mapping[str, Any]]:
    return [event for event in events if event.get("type") == type_]


def turns_present(events: Sequence[Mapping[str, Any]]) -> list[int]:
    """Return the sorted list of turn numbers that have a ``prompt`` event."""
    turns = {int(event["turn"]) for event in _by_type(events, "prompt") if "turn" in event}
    return sorted(turns)


def reconstruct_system_text(events: Sequence[Mapping[str, Any]], turn: int) -> str | None:
    """Recover the system text in force on ``turn``.

    Uses the latest non-null ``prompt.system.text`` whose ``sha256`` matches
    the hash on turn ``N``'s prompt (or the text embedded when first seen).

    Args:
        events: Parsed MVL events in file order.
        turn: The turn to reconstruct.

    Returns:
        The system text string, or ``None`` when the prompt carried no system
        hash (empty / absent system prompt).

    Raises:
        ReconstructionError: If the prompt is missing or its hash has no
            matching full text anywhere in the log.
    """
    prompts = [event for event in _by_type(events, "prompt") if event.get("turn") == turn]
    if not prompts:
        msg = f"no prompt event for turn {turn}"
        raise ReconstructionError(msg)
    target = prompts[-1]
    system = target.get("system") or {}
    digest = system.get("sha256")
    if digest is None:
        return None
    # Walk in order so the first non-null text for this digest wins; later
    # prompts for the same digest must carry null text per the producer rules.
    for event in _by_type(events, "prompt"):
        sys = event.get("system") or {}
        if sys.get("sha256") == digest and sys.get("text") is not None:
            return sys["text"]
    msg = f"system text for sha256={digest} on turn {turn} never appears in full"
    raise ReconstructionError(msg)


def reconstruct_tool_catalog(
    events: Sequence[Mapping[str, Any]],
    turn: int,
) -> list[dict[str, Any]]:
    """Recover the ordered tool definitions offered on ``turn``.

    Args:
        events: Parsed MVL events in file order.
        turn: The turn to reconstruct.

    Returns:
        The ordered list of complete tool definitions (name, description,
        input_schema).

    Raises:
        ReconstructionError: If the prompt or its catalogue body is missing.
    """
    prompts = [event for event in _by_type(events, "prompt") if event.get("turn") == turn]
    if not prompts:
        msg = f"no prompt event for turn {turn}"
        raise ReconstructionError(msg)
    digest = prompts[-1].get("tool_catalog_sha256")
    if not digest:
        msg = f"prompt on turn {turn} is missing tool_catalog_sha256"
        raise ReconstructionError(msg)
    for event in _by_type(events, "tool_catalog"):
        if event.get("sha256") == digest:
            tools = event.get("tools")
            if not isinstance(tools, list):
                msg = f"tool_catalog sha256={digest} has no tools list"
                raise ReconstructionError(msg)
            return list(tools)
    msg = f"tool_catalog body for sha256={digest} never appears in the log"
    raise ReconstructionError(msg)


def reconstruct_messages(
    events: Sequence[Mapping[str, Any]],
    turn: int,
) -> list[dict[str, Any]]:
    """Rebuild the message list the model saw on ``turn``.

    Applies every ``prompt`` with ``messages.mode=full`` as a reset, then
    appends each subsequent ``delta`` until turn ``N`` inclusive.

    Args:
        events: Parsed MVL events in file order.
        turn: The turn to reconstruct.

    Returns:
        The ordered list of message items (role/content plus any extra fields
        the producer recorded, such as ``tool_calls`` or ``tool_call_id``).

    Raises:
        ReconstructionError: If a required prompt is missing, a delta arrives
            before any full, or turn ``N`` has no prompt.
    """
    messages: list[dict[str, Any]] = []
    saw_full = False
    matched_turn = False
    for event in _by_type(events, "prompt"):
        event_turn = event.get("turn")
        if event_turn is None:
            continue
        if int(event_turn) > turn:
            break
        body = event.get("messages") or {}
        mode = body.get("mode")
        items = body.get("items") or []
        if mode == "full":
            messages = [dict(item) for item in items]
            saw_full = True
        elif mode == "delta":
            if not saw_full:
                msg = f"delta prompt on turn {event_turn} arrived before any full"
                raise ReconstructionError(msg)
            messages.extend(dict(item) for item in items)
        else:
            msg = f"prompt on turn {event_turn} has unknown messages.mode={mode!r}"
            raise ReconstructionError(msg)
        if int(event_turn) == turn:
            matched_turn = True
    if not matched_turn:
        msg = f"no prompt event for turn {turn}"
        raise ReconstructionError(msg)
    return messages


def reconstruct_params(events: Sequence[Mapping[str, Any]], turn: int) -> dict[str, Any]:
    """Recover the sampling parameters on ``turn``.

    Args:
        events: Parsed MVL events in file order.
        turn: The turn to reconstruct.

    Returns:
        The ``prompt.params`` dict (temperature, max_tokens, …).

    Raises:
        ReconstructionError: If the prompt is missing.
    """
    prompts = [event for event in _by_type(events, "prompt") if event.get("turn") == turn]
    if not prompts:
        msg = f"no prompt event for turn {turn}"
        raise ReconstructionError(msg)
    params = prompts[-1].get("params") or {}
    if not isinstance(params, dict):
        msg = f"prompt on turn {turn} has non-object params"
        raise ReconstructionError(msg)
    return dict(params)


def reconstruct_tools_offered(
    events: Sequence[Mapping[str, Any]],
    turn: int,
) -> list[str | None]:
    """Recover the tools offered on the request for ``turn``.

    Args:
        events: Parsed MVL events in file order.
        turn: The turn to reconstruct.

    Returns:
        The ``prompt.tools_offered`` list.

    Raises:
        ReconstructionError: If the prompt is missing.
    """
    prompts = [event for event in _by_type(events, "prompt") if event.get("turn") == turn]
    if not prompts:
        msg = f"no prompt event for turn {turn}"
        raise ReconstructionError(msg)
    offered = prompts[-1].get("tools_offered")
    if not isinstance(offered, list):
        msg = f"prompt on turn {turn} is missing tools_offered"
        raise ReconstructionError(msg)
    return list(offered)


def reconstruct_turn(events: Sequence[Mapping[str, Any]], turn: int) -> dict[str, Any]:
    """Rebuild every artifact the reconstruction checklist requires for ``turn``.

    Args:
        events: Parsed MVL events in file order.
        turn: The turn to reconstruct.

    Returns:
        A dict with keys ``system_text``, ``tool_catalog``, ``messages``,
        ``params`` and ``tools_offered``.

    Raises:
        ReconstructionError: If any artifact is missing or contradictory.
    """
    return {
        "system_text": reconstruct_system_text(events, turn),
        "tool_catalog": reconstruct_tool_catalog(events, turn),
        "messages": reconstruct_messages(events, turn),
        "params": reconstruct_params(events, turn),
        "tools_offered": reconstruct_tools_offered(events, turn),
    }


def assert_log_reconstructs(events: Sequence[Mapping[str, Any]]) -> None:
    """Assert every turn in ``events`` reconstructs without contradiction.

    Also enforces the envelope invariants the conformance suite cares about:
    gap-free monotonic ``seq``, stable ``run``, and that every prompt's
    system/catalogue digests resolve to a full body somewhere in the log.

    Args:
        events: Parsed MVL events in file order.

    Raises:
        ReconstructionError: On the first failure.
    """
    if not events:
        msg = "log is empty"
        raise ReconstructionError(msg)

    seqs = [event.get("seq") for event in events]
    if seqs != list(range(len(events))):
        msg = f"seq is not gap-free monotonic: {seqs}"
        raise ReconstructionError(msg)

    runs = {event.get("run") for event in events}
    if len(runs) != 1 or None in runs:
        msg = f"run id is not stable across the log: {runs}"
        raise ReconstructionError(msg)

    for event in events:
        if event.get("v") != 1:
            msg = f"event seq={event.get('seq')} has unexpected v={event.get('v')}"
            raise ReconstructionError(msg)
        if not event.get("type") or not event.get("ts"):
            msg = f"event seq={event.get('seq')} is missing type or ts"
            raise ReconstructionError(msg)

    for turn in turns_present(events):
        reconstruct_turn(events, turn)


def tool_result_content_by_call_id(
    events: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    """Index ``tool_result.content_shown`` by ``call_id`` for honesty checks."""
    out: dict[str, str] = {}
    for event in _by_type(events, "tool_result"):
        call_id = event.get("call_id")
        if isinstance(call_id, str):
            out[call_id] = event.get("content_shown") or ""
    return out


def iter_completions(events: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Return completion events in file order."""
    return list(_by_type(list(events), "completion"))


__all__ = [
    "ReconstructionError",
    "assert_log_reconstructs",
    "iter_completions",
    "reconstruct_messages",
    "reconstruct_params",
    "reconstruct_system_text",
    "reconstruct_tool_catalog",
    "reconstruct_tools_offered",
    "reconstruct_turn",
    "tool_result_content_by_call_id",
    "turns_present",
]

"""Event-type constants for the Model View Log (MVL) v1 schema."""

from typing import Final

MVL_VERSION: Final = 1
"""The MVL spec version stamped on every envelope (`v` field)."""

EVENT_TYPES: Final = frozenset(
    {
        "run_started",
        "tool_catalog",
        "prompt",
        "completion",
        "tool_result",
        "context_changed",
        "tools_changed",
        "run_ended",
    }
)
"""The set of event types defined by MVL v1.

Unknown ``type`` values must be skipped rather than treated as errors so a v1
reader survives a v1.1 producer.
"""

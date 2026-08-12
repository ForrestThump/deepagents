"""Model View Log (MVL) tracing for Deep Agents.

The :class:`ModelViewLogMiddleware` records, at the provider boundary, exactly
what the model saw and what it produced — the prompt message stream, ordered
tool catalogue, sampling parameters, completions, tool results, and the tool /
context drift events — as append-and-flush JSONL conforming to the MVL v1 spec.

Example:
```python
from deepagents import create_deep_agent
from deepagents.tracing import ModelViewLogMiddleware

tracer = ModelViewLogMiddleware(path="model_view.jsonl", harness_name="deepagents")
agent = create_deep_agent(model=..., middleware=[tracer])

with tracer.run(task_id="P3.2"):
    agent.invoke({"messages": [{"role": "user", "content": "hi"}]})
```
"""

from deepagents.tracing.emitter import (
    ModelViewLogger,
    canonical_json,
    now_rfc3339,
    read_events,
    sha256_bytes,
    sha256_json,
)
from deepagents.tracing.middleware import (
    ModelViewLogConfig,
    ModelViewLogMiddleware,
)
from deepagents.tracing.reconstruct import (
    ReconstructionError,
    assert_log_reconstructs,
    reconstruct_messages,
    reconstruct_params,
    reconstruct_system_text,
    reconstruct_tool_catalog,
    reconstruct_tools_offered,
    reconstruct_turn,
    turns_present,
)

__all__ = [
    "ModelViewLogConfig",
    "ModelViewLogMiddleware",
    "ModelViewLogger",
    "ReconstructionError",
    "assert_log_reconstructs",
    "canonical_json",
    "now_rfc3339",
    "read_events",
    "reconstruct_messages",
    "reconstruct_params",
    "reconstruct_system_text",
    "reconstruct_tool_catalog",
    "reconstruct_tools_offered",
    "reconstruct_turn",
    "sha256_bytes",
    "sha256_json",
    "turns_present",
]

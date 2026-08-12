"""Append-and-flush JSONL sink for Model View Log (MVL) v1 events.

Every event is a single JSON object on its own line (UTF-8, LF). Lines are
written and flushed as they are produced so a run that crashes still yields a
parseable, valid log — the most valuable log is the one from the run that
crashed, and a log assembled at exit is exactly the log you lose.

Each line carries the MVL envelope:

- `v` — spec version (`1`).
- `type` — the event type.
- `ts` — RFC3339 UTC with millisecond precision.
- `run` — run id, stable for the whole log.
- `seq` — monotonic from `0`; the ordering authority (two events may share `ts`).

This module is deliberately free of any knowledge of ``AgentMiddleware``; the
Middleware in :mod:`deepagents.tracing.middleware` is what decides *which*
events to emit. Keeping the sink generic keeps it unit-testable in isolation.
"""

import contextlib
import hashlib
import json
import threading
import uuid
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self

from deepagents.tracing._events import EVENT_TYPES, MVL_VERSION

_JSON_SEPARATORS = (",", ":")


def now_rfc3339() -> str:
    """Return the current UTC time as an RFC3339 string with millisecond precision."""
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def canonical_json(value: object) -> str:
    """Serialize ``value`` to canonical JSON for content digesting.

    The canonical form used for hashing is deterministic and key-sorted so that
    structurally identical values (e.g. the same ordered tool catalogue) always
    produce the same digest, independent of key insertion order.

    Args:
        value: The value to serialize.

    Returns:
        The canonical JSON string.
    """
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=_JSON_SEPARATORS)


def sha256_bytes(data: bytes) -> str:
    """Return the lowercase hex SHA-256 digest of ``data``."""
    return hashlib.sha256(data).hexdigest()


def sha256_json(value: object) -> str:
    """Return the SHA-256 digest of ``value`` under canonical JSON serialization."""
    return sha256_bytes(canonical_json(value).encode("utf-8"))


class ModelViewLogger:
    """Append-and-flush JSONL writer emitting MVL v1 envelopes.

    Writes are serialized with a lock so a single log can safely be shared by a
    multithreaded producer without interleaved (and therefore malformed) lines.
    Any exception raised by the wrapped ``status_callback`` is re-raised after
    the enclosing call, matching the append-as-you-go contract — the event is
    already on disk by the time the exception surfaces.

    Args:
        path: Filesystem path of the JSONL log. Its parent directory is created
            on first use.
        run: Optional explicit run id stamped on every event. A random hex id is
            generated otherwise.
        offload_dir: Optional directory where large payloads are stored
            content-addressed. When set, :meth:`offload` enables
            ``full_content`` refs for oversized tool results.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        run: str | None = None,
        offload_dir: str | Path | None = None,
    ) -> None:
        """Initialize the logger, leaving the file handle opened lazily."""
        self._path = Path(path)
        self._offload_dir = Path(offload_dir) if offload_dir is not None else None
        self._run = run or uuid.uuid4().hex
        self._seq = 0
        self._lock = threading.Lock()
        self._fh: Any = None

    @property
    def run(self) -> str:
        """The run id stamped on every event."""
        return self._run

    def replace_run(self, run: str) -> None:
        """Stably set the run id used on subsequently emitted events.

        Args:
            run: The run id to use.
        """
        with self._lock:
            self._run = run

    def _open(self) -> None:
        if self._fh is not None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self._path.open("a", encoding="utf-8", newline="\n")

    def emit(self, type_: str, **fields: Any) -> None:
        """Serialize one event with the envelope appended and flush it.

        Args:
            type_: The MVL event type. Must be a known type (see
                :data:`deepagents.tracing._events.EVENT_TYPES`) but producers
                up-versioning the message set are preserved as-is.
            **fields: Event-specific fields.

        Raises:
            OSError: If the underlying path cannot be opened or written.
        """
        if type_ not in EVENT_TYPES:
            # A newer producer may emit types this reader does not know yet;
            # they must be skipped, not treated as errors. We still persist them.
            pass
        with self._lock:
            self._open()
            # Build and increment `seq` under the lock so concurrent producers
            # never observe the same sequence number, keeping it gap-free and
            # monotonic. `run` is read here too so a concurrent `replace_run`
            # cannot split a single event across two run ids.
            payload = {
                "v": MVL_VERSION,
                "type": type_,
                "ts": now_rfc3339(),
                "run": self._run,
                "seq": self._seq,
                **fields,
            }
            line = json.dumps(payload, ensure_ascii=False, separators=_JSON_SEPARATORS)
            self._fh.write(line + "\n")
            self._fh.flush()
            self._seq += 1

    def offload(self, data: bytes, *, extension: str = "txt") -> dict[str, Any]:
        """Store ``data`` content-addressed and return its content reference.

        The reference embeds the SHA-256 digest of the raw bytes, so two runs
        that read the same payload produce the same ``ref`` and cross-run
        comparison is cheap.

        Args:
            data: Raw bytes to store.
            extension: File extension for the stored blob.

        Returns:
            A content-reference dict of ``{"ref", "bytes", "path"}``, or a
            ``{"ref", "bytes"}`` dict when no ``offload_dir`` is configured.
        """
        digest = sha256_bytes(data)
        location = None
        if self._offload_dir is not None:
            location = self._offload_dir / f"{digest}.{extension}"
            with self._lock:
                # Serialize the existence check and rename so two concurrent
                # offloads of the same digest cannot race on a shared temp path.
                if not location.exists():
                    location.parent.mkdir(parents=True, exist_ok=True)
                    tmp = location.with_suffix(location.suffix + ".tmp")
                    tmp.write_bytes(data)
                    tmp.replace(location)
        ref: dict[str, Any] = {"ref": f"sha256:{digest}", "bytes": len(data)}
        if location is not None:
            ref["path"] = str(location)
        return ref

    def close(self) -> None:
        """Flush and close the underlying file handle, if any is open."""
        with self._lock:
            if self._fh is not None:
                self._fh.close()
                self._fh = None

    def __enter__(self) -> Self:
        """Enter the context manager, returning the logger."""
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Exit the context manager, closing the underlying file."""
        self.close()

    def __del__(self) -> None:
        """Best-effort close when the logger is garbage collected.

        Ensures the underlying file handle is not left dangling (which would
        surface as a ``ResourceWarning`` and a file-finalizer unraisable
        exception). Errors are swallowed: the handle may already be closed or
        the interpreter may be mid-shutdown.
        """
        with contextlib.suppress(Exception):  # never raise from a destructor
            self.close()


def read_events(path: str | Path) -> list[dict[str, Any]]:
    """Parse a JSONL file into a list of event dicts, skipping malformed lines.

    Args:
        path: Path to the JSONL log.

    Returns:
        The list of parsed events in file order.
    """
    events: list[dict[str, Any]] = []
    for _, raw in enumerate(_iter_lines(Path(path))):
        if not raw:
            continue
        try:
            events.append(json.loads(raw))
        except json.JSONDecodeError:
            # Crash survival requirement: a truncated trailing line must not
            # invalidate the events that came before it.
            continue
    return events


def _iter_lines(path: Path) -> Iterable[str]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            yield line.rstrip("\n")

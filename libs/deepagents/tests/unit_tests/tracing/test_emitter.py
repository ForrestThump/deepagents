"""Unit tests for the MVL JSONL emitter and hashing helpers."""

import json
import re
import threading

from deepagents.tracing import (
    ModelViewLogger,
    canonical_json,
    now_rfc3339,
    read_events,
    sha256_bytes,
    sha256_json,
)


class TestModelViewLogger:
    """Test the low-level append-and-flush MVL sink."""

    def test_envelope_shape_and_monotonic_seq(self, tmp_path) -> None:
        """Each emitted line is a valid JSON envelope with the full envelope fields."""
        path = tmp_path / "log.jsonl"
        logger = ModelViewLogger(path, run="run-1")
        logger.emit("run_started", harness={"name": "deepagents"})
        logger.emit("prompt", turn=1, messages={"mode": "full", "items": []})
        logger.emit("run_ended", outcome="succeeded")
        logger.close()

        events = read_events(path)
        assert [event["seq"] for event in events] == [0, 1, 2]
        assert all(event["v"] == 1 for event in events)
        assert all(event["run"] == "run-1" for event in events)
        assert all(re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z", event["ts"]) for event in events)
        assert events[0]["type"] == "run_started"
        assert events[-1]["type"] == "run_ended"

    def test_append_and_flush_writes_each_event_immediately(self, tmp_path) -> None:
        """Events are persisted before the next one is emitted, not at close."""
        path = tmp_path / "log.jsonl"
        logger = ModelViewLogger(path)
        logger.emit("run_started")
        logger.emit("completion", turn=1, text="a")
        # Data must already be on disk without calling close().
        assert len(read_events(path)) == 2

    def test_generated_run_id_is_stable(self, tmp_path) -> None:
        """A fresh logger generates a run id that is stable across its events."""
        path = tmp_path / "log.jsonl"
        logger = ModelViewLogger(path)
        logger.emit("run_started")
        logger.emit("run_ended")
        logger.close()
        runs = {event["run"] for event in read_events(path)}
        assert len(runs) == 1
        assert len(next(iter(runs))) == 32  # hex digest length

    def test_crash_survival_ignores_truncated_trailing_line(self, tmp_path) -> None:
        """A malformed final line does not invalidate the events before it."""
        path = tmp_path / "log.jsonl"
        logger = ModelViewLogger(path)
        logger.emit("run_started")
        logger.emit("run_ended")
        logger.close()
        with path.open("a", encoding="utf-8") as fh:
            fh.write('{"v":1,"type":"prompt","seq":2')  # truncated mid-run on disk
        assert len(read_events(path)) == 2

    def test_offload_is_content_addressed_and_deduplicated(self, tmp_path) -> None:
        """Offloaded payloads get the same ref and path for identical bytes."""
        offload = tmp_path / "offload"
        logger = ModelViewLogger(tmp_path / "log.jsonl", offload_dir=offload)
        ref1 = logger.offload(b"same bytes")
        ref2 = logger.offload(b"same bytes")
        assert ref1 == ref2
        assert ref1["ref"].startswith("sha256:")
        assert ref1["bytes"] == 10
        assert ref2["path"] == ref1["path"]
        assert (offload / f"{ref1['ref'].split(':')[1]}.txt").exists()
        assert len(list(offload.iterdir())) == 1  # deduplicated on disk

    def test_offload_without_dir_omits_path(self, tmp_path) -> None:
        """When no offload dir is configured the ref still carries digest and bytes."""
        logger = ModelViewLogger(tmp_path / "log.jsonl")
        ref = logger.offload(b"data")
        assert ref == {"ref": "sha256:" + sha256_bytes(b"data"), "bytes": 4}
        assert "path" not in ref

    def test_offload_dedups_concurrent_calls(self, tmp_path) -> None:
        """Concurrent offloads of the same bytes settle on one on-disk blob."""
        offload = tmp_path / "offload"
        logger = ModelViewLogger(tmp_path / "log.jsonl", offload_dir=offload)
        barrier = threading.Barrier(4)

        def worker() -> None:
            barrier.wait()
            ref = logger.offload(b"shared payload")
            assert ref["ref"] == "sha256:" + sha256_bytes(b"shared payload")

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert len(list(offload.iterdir())) == 1  # no partial temp files left

    def test_seq_is_gap_free_under_concurrent_emit(self, tmp_path) -> None:
        """Concurrent producers still yield monotonic, gap-free sequence numbers."""
        path = tmp_path / "log.jsonl"
        logger = ModelViewLogger(path)
        n_threads, per_thread = 8, 50
        barrier = threading.Barrier(n_threads)

        def worker() -> None:
            barrier.wait()
            for _ in range(per_thread):
                logger.emit("prompt", turn=1)

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        logger.close()
        seqs = [event["seq"] for event in read_events(path)]
        assert len(seqs) == n_threads * per_thread
        assert seqs == list(range(len(seqs)))


class TestHashing:
    """Test canonical JSON and content digests."""

    def test_canonical_json_is_key_sorted(self) -> None:
        """Structurally identical dicts hash the same regardless of key order."""
        a = {"b": 1, "a": 2}
        b = {"a": 2, "b": 1}
        assert canonical_json(a) == canonical_json(b)
        assert sha256_json(a) == sha256_json(b)

    def test_sha256_helpers(self) -> None:
        """Digests match hashlib on the expected inputs."""
        # sha256_json digests the canonical (quoted) JSON serialization of a string.
        assert sha256_json("abc") == sha256_bytes(b'"abc"')
        assert sha256_json("abc") == sha256_json("abc")

    def test_canonical_json_roundtrips(self) -> None:
        """Canonical JSON can be parsed back to the original object."""
        value = {"tools": [{"name": "grep", "description": "Search"}]}
        assert json.loads(canonical_json(value)) == value


class TestTimestamp:
    """Test the RFC3339 timestamp helper."""

    def test_rfc3339_utc_milliseconds(self) -> None:
        """Timestamps are UTC with milliseconds and a trailing Z."""
        ts = now_rfc3339()
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z", ts)
        assert ts.endswith("Z")

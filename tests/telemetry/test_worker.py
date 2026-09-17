"""Unit tests for the persistent telemetry worker protocol."""

import json
import threading
from multiprocessing import Pipe
from multiprocessing.connection import Connection

from tt_blackhole_benchmark.telemetry.worker import (
    serve_worker_requests,
)


class FakeTelemetrySource:
    def __init__(self) -> None:
        self.update_count = 0

    def update_telem(self) -> None:
        self.update_count += 1

    def get_logs_json(self) -> str:
        return json.dumps(
            {
                "device_info": [],
                "sequence": self.update_count,
            }
        )


class FailingTelemetrySource:
    def update_telem(self) -> None:
        raise RuntimeError("telemetry read failed")

    def get_logs_json(self) -> str:
        raise AssertionError("must not be called")


def start_worker_thread(
    connection: Connection,
    source: FakeTelemetrySource | FailingTelemetrySource,
) -> threading.Thread:
    thread = threading.Thread(
        target=serve_worker_requests,
        args=(connection, source),
    )
    thread.start()
    return thread


def test_worker_serves_multiple_acquisitions() -> None:
    parent_connection, worker_connection = Pipe()
    source = FakeTelemetrySource()
    thread = start_worker_thread(worker_connection, source)

    try:
        parent_connection.send("acquire")
        first = parent_connection.recv()

        parent_connection.send("acquire")
        second = parent_connection.recv()

        assert first["kind"] == "sample"
        assert second["kind"] == "sample"
        assert json.loads(first["payload"])["sequence"] == 1
        assert json.loads(second["payload"])["sequence"] == 2
        assert source.update_count == 2

        parent_connection.send("close")
        closed = parent_connection.recv()

        assert closed == {"kind": "closed"}
    finally:
        parent_connection.close()
        worker_connection.close()
        thread.join(timeout=1.0)

    assert thread.is_alive() is False


def test_worker_reports_acquisition_error_and_continues() -> None:
    parent_connection, worker_connection = Pipe()
    source = FailingTelemetrySource()
    thread = start_worker_thread(worker_connection, source)

    try:
        parent_connection.send("acquire")
        response = parent_connection.recv()

        assert response == {
            "kind": "error",
            "message": "telemetry read failed",
        }

        parent_connection.send("close")
        assert parent_connection.recv() == {"kind": "closed"}
    finally:
        parent_connection.close()
        worker_connection.close()
        thread.join(timeout=1.0)

    assert thread.is_alive() is False


def test_worker_rejects_unknown_command() -> None:
    parent_connection, worker_connection = Pipe()
    source = FakeTelemetrySource()
    thread = start_worker_thread(worker_connection, source)

    try:
        parent_connection.send("invalid-command")
        response = parent_connection.recv()

        assert response["kind"] == "error"
        assert "Unknown worker command" in response["message"]

        parent_connection.send("close")
        assert parent_connection.recv() == {"kind": "closed"}
    finally:
        parent_connection.close()
        worker_connection.close()
        thread.join(timeout=1.0)

    assert thread.is_alive() is False


def test_worker_stops_when_parent_pipe_closes() -> None:
    parent_connection, worker_connection = Pipe()
    source = FakeTelemetrySource()
    thread = start_worker_thread(worker_connection, source)

    parent_connection.close()
    thread.join(timeout=1.0)
    worker_connection.close()

    assert thread.is_alive() is False

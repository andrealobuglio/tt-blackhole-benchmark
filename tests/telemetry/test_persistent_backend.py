"""Unit tests for the persistent telemetry backend."""

import json
import time
from multiprocessing.connection import Connection

import pytest

from tt_blackhole_benchmark.telemetry.backend import (
    TelemetryBackendError,
)
from tt_blackhole_benchmark.telemetry.persistent_backend import (
    PersistentTtSmiBackend,
)
from tt_blackhole_benchmark.telemetry.worker import (
    serve_worker_requests,
)


class FakeTelemetrySource:
    def __init__(self) -> None:
        self.sequence = 0

    def update_telem(self) -> None:
        self.sequence += 1

    def get_logs_json(self) -> str:
        return json.dumps(
            {
                "device_info": [
                    {
                        "board_info": {
                            "bus_id": "0000:01:00.0",
                            "board_type": "p150a",
                        },
                        "telemetry": {
                            "voltage": "0.74",
                            "current": "48.0",
                            "power": str(35 + self.sequence),
                            "aiclk": "800",
                            "asic_temperature": "31.0",
                            "fan_speed": "38",
                            "heartbeat": str(self.sequence),
                        },
                    }
                ]
            }
        )


def fake_worker(connection: Connection) -> None:
    connection.send({"kind": "ready"})
    serve_worker_requests(
        connection,
        FakeTelemetrySource(),
    )
    connection.close()


def fatal_worker(connection: Connection) -> None:
    connection.send(
        {
            "kind": "fatal",
            "message": "initialization failed",
        }
    )
    connection.close()


def silent_worker(connection: Connection) -> None:
    time.sleep(5)
    connection.close()


def test_persistent_backend_serves_multiple_samples() -> None:
    backend = PersistentTtSmiBackend(
        worker_target=fake_worker,
    )

    with backend:
        first = backend.acquire()
        second = backend.acquire()

        assert backend.is_running is True
        assert first.snapshot.samples[0].power_w == 36.0
        assert second.snapshot.samples[0].power_w == 37.0
        assert first.command == ("persistent-tt-smi-worker",)

    assert backend.is_running is False


def test_acquire_before_start_is_rejected() -> None:
    backend = PersistentTtSmiBackend(
        worker_target=fake_worker,
    )

    with pytest.raises(
        TelemetryBackendError,
        match="is not running",
    ):
        backend.acquire()


def test_second_start_is_rejected() -> None:
    backend = PersistentTtSmiBackend(
        worker_target=fake_worker,
    )
    backend.start()

    try:
        with pytest.raises(
            TelemetryBackendError,
            match="already running",
        ):
            backend.start()
    finally:
        backend.close()


def test_worker_initialization_failure_is_reported() -> None:
    backend = PersistentTtSmiBackend(
        worker_target=fatal_worker,
    )

    with pytest.raises(
        TelemetryBackendError,
        match="initialization failed",
    ):
        backend.start()

    assert backend.is_running is False


def test_worker_startup_timeout_is_reported() -> None:
    backend = PersistentTtSmiBackend(
        worker_target=silent_worker,
        startup_timeout_seconds=0.05,
    )

    with pytest.raises(
        TelemetryBackendError,
        match="Timed out",
    ):
        backend.start()

    assert backend.is_running is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("startup_timeout_seconds", 0.0),
        ("request_timeout_seconds", 0.0),
    ],
)
def test_invalid_timeouts_are_rejected(
    field: str,
    value: float,
) -> None:
    arguments = {field: value}

    with pytest.raises(
        ValueError,
        match="must be positive",
    ):
        PersistentTtSmiBackend(**arguments)

"""Tests for the tt-smi subprocess backend."""

import json
import sys

import pytest

from tt_blackhole_benchmark.telemetry.backend import (
    TelemetryBackendError,
    TtSmiBackend,
)


def valid_document() -> dict[str, object]:
    return {
        "device_info": [
            {
                "board_info": {
                    "bus_id": "0000:01:00.0",
                    "board_type": "p150a",
                },
                "telemetry": {
                    "voltage": "0.74",
                    "current": "48.0",
                    "power": "35.0",
                    "aiclk": "800",
                    "asic_temperature": "30.7",
                    "fan_speed": "38",
                    "heartbeat": "112416",
                },
            }
        ]
    }


def python_command(source: str) -> tuple[str, ...]:
    """Build a portable subprocess command for tests."""

    return (sys.executable, "-c", source)


def test_backend_accepts_clean_process_exit() -> None:
    payload = json.dumps(valid_document())
    backend = TtSmiBackend(python_command(f"print({payload!r})"))

    acquisition = backend.acquire()

    assert acquisition.exit_code == 0
    assert acquisition.recovered_from_nonzero_exit is False
    assert acquisition.snapshot.samples[0].power_w == 35.0
    assert acquisition.duration_seconds >= 0


def test_backend_recovers_json_from_abnormal_exit() -> None:
    payload = json.dumps(valid_document())
    source = (
        "import sys; "
        f"print({payload!r}); "
        "print('double free or corruption', file=sys.stderr); "
        "sys.exit(134)"
    )
    backend = TtSmiBackend(python_command(source))

    acquisition = backend.acquire()

    assert acquisition.exit_code == 134
    assert acquisition.recovered_from_nonzero_exit is True
    assert acquisition.snapshot.samples[0].power_w == 35.0
    assert "double free" in acquisition.stderr


def test_backend_rejects_nonzero_exit_without_json() -> None:
    backend = TtSmiBackend(
        python_command("import sys; print('failure', file=sys.stderr); sys.exit(1)")
    )

    with pytest.raises(
        TelemetryBackendError,
        match=r"no valid telemetry \(exit code 1\)",
    ):
        backend.acquire()


def test_backend_rejects_missing_command() -> None:
    backend = TtSmiBackend(
        ("/definitely/missing/tt-smi",),
    )

    with pytest.raises(
        TelemetryBackendError,
        match="Telemetry command not found",
    ):
        backend.acquire()


def test_backend_enforces_timeout() -> None:
    backend = TtSmiBackend(
        python_command("import time; time.sleep(2)"),
        timeout_seconds=0.05,
    )

    with pytest.raises(
        TelemetryBackendError,
        match="Telemetry command timed out",
    ):
        backend.acquire()


@pytest.mark.parametrize(
    "timeout_seconds",
    [0.0, -1.0],
)
def test_backend_rejects_invalid_timeout(
    timeout_seconds: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="timeout_seconds must be positive",
    ):
        TtSmiBackend(
            timeout_seconds=timeout_seconds,
        )


def test_backend_rejects_empty_command() -> None:
    with pytest.raises(
        ValueError,
        match="command must not be empty",
    ):
        TtSmiBackend(())

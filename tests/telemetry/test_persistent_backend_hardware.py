"""Hardware tests for persistent tt-smi telemetry."""

import pytest

from tt_blackhole_benchmark.telemetry.persistent_backend import (
    PersistentTtSmiBackend,
)


@pytest.mark.hardware
def test_persistent_backend_acquires_multiple_samples() -> None:
    with PersistentTtSmiBackend(
        startup_timeout_seconds=30.0,
        request_timeout_seconds=5.0,
    ) as backend:
        acquisitions = [backend.acquire() for _ in range(5)]

        assert backend.is_running is True

    assert backend.is_running is False
    assert len(acquisitions) == 5

    for acquisition in acquisitions:
        sample = acquisition.snapshot.samples[0]

        assert acquisition.exit_code == 0
        assert acquisition.command == ("persistent-tt-smi-worker",)
        assert sample.board_type == "p150a"
        assert sample.bus_id == "0000:01:00.0"
        assert sample.power_w > 0
        assert sample.voltage_v > 0
        assert sample.current_a > 0
        assert sample.aiclk_mhz > 0

    monotonic_times = [acquisition.snapshot.monotonic_ns for acquisition in acquisitions]

    assert monotonic_times == sorted(monotonic_times)
    assert len(set(monotonic_times)) == len(monotonic_times)

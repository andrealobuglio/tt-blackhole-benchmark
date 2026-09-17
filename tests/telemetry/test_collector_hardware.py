"""Hardware integration tests for periodic telemetry collection."""

import time

import pytest

from tt_blackhole_benchmark.telemetry.collector import (
    TelemetryCollector,
)
from tt_blackhole_benchmark.telemetry.persistent_backend import (
    PersistentTtSmiBackend,
)


@pytest.mark.hardware
def test_collector_samples_persistent_backend() -> None:
    with PersistentTtSmiBackend(
        startup_timeout_seconds=30.0,
        request_timeout_seconds=5.0,
    ) as backend:
        collector = TelemetryCollector(
            backend,
            interval_seconds=0.5,
        )

        collector.start()

        try:
            time.sleep(3.2)
        finally:
            collector.stop(timeout_seconds=5.0)

    acquisitions = collector.acquisitions

    assert collector.is_running is False
    assert collector.failures == ()
    assert len(acquisitions) >= 5

    timestamps = [acquisition.snapshot.monotonic_ns for acquisition in acquisitions]

    assert timestamps == sorted(timestamps)
    assert len(set(timestamps)) == len(timestamps)

    for acquisition in acquisitions:
        sample = acquisition.snapshot.samples[0]

        assert sample.board_type == "p150a"
        assert sample.power_w > 0

"""Unit tests for periodic telemetry collection."""

import threading
from datetime import UTC, datetime

import pytest

from tt_blackhole_benchmark.telemetry.backend import (
    TelemetryBackendError,
)
from tt_blackhole_benchmark.telemetry.collector import (
    TelemetryCollector,
    TelemetryCollectorError,
)
from tt_blackhole_benchmark.telemetry.models import (
    TelemetryAcquisition,
    TelemetrySample,
    TelemetrySnapshot,
)


def make_acquisition() -> TelemetryAcquisition:
    timestamp = datetime.now(UTC)
    monotonic_ns = 1_000_000

    sample = TelemetrySample(
        captured_at=timestamp,
        monotonic_ns=monotonic_ns,
        device_index=0,
        bus_id="0000:01:00.0",
        board_type="p150a",
        power_w=35.0,
        voltage_v=0.74,
        current_a=48.0,
        asic_temperature_c=30.7,
        aiclk_mhz=800.0,
        fan_speed_percent=38.0,
        heartbeat=112_416,
    )
    snapshot = TelemetrySnapshot(
        captured_at=timestamp,
        monotonic_ns=monotonic_ns,
        samples=(sample,),
    )

    return TelemetryAcquisition(
        snapshot=snapshot,
        command=("fake-tt-smi", "-s"),
        exit_code=0,
        duration_seconds=0.001,
    )


class CountingBackend:
    def __init__(self, target_count: int) -> None:
        self.call_count = 0
        self.target_count = target_count
        self.target_reached = threading.Event()

    def acquire(self) -> TelemetryAcquisition:
        self.call_count += 1

        if self.call_count >= self.target_count:
            self.target_reached.set()

        return make_acquisition()


class FailingOnceBackend:
    def __init__(self) -> None:
        self.call_count = 0
        self.success_reached = threading.Event()

    def acquire(self) -> TelemetryAcquisition:
        self.call_count += 1

        if self.call_count == 1:
            raise TelemetryBackendError("temporary failure")

        self.success_reached.set()
        return make_acquisition()


def test_collector_acquires_periodic_samples() -> None:
    backend = CountingBackend(target_count=3)
    collector = TelemetryCollector(
        backend,
        interval_seconds=0.01,
    )

    collector.start()
    reached = backend.target_reached.wait(timeout=1.0)
    collector.stop(timeout_seconds=1.0)

    assert reached is True
    assert collector.is_running is False
    assert len(collector.acquisitions) >= 3
    assert collector.failures == ()


def test_collector_continues_after_recoverable_failure() -> None:
    backend = FailingOnceBackend()
    collector = TelemetryCollector(
        backend,
        interval_seconds=0.01,
    )

    collector.start()
    reached = backend.success_reached.wait(timeout=1.0)
    collector.stop(timeout_seconds=1.0)

    assert reached is True
    assert len(collector.failures) == 1
    assert collector.failures[0].message == "temporary failure"
    assert len(collector.acquisitions) >= 1


def test_collector_context_manager_stops_thread() -> None:
    backend = CountingBackend(target_count=1)

    with TelemetryCollector(
        backend,
        interval_seconds=0.01,
    ) as collector:
        reached = backend.target_reached.wait(timeout=1.0)
        assert reached is True
        assert collector.is_running is True

    assert collector.is_running is False


def test_collector_rejects_second_start() -> None:
    backend = CountingBackend(target_count=1)
    collector = TelemetryCollector(
        backend,
        interval_seconds=0.01,
    )

    collector.start()

    try:
        with pytest.raises(
            TelemetryCollectorError,
            match="already running",
        ):
            collector.start()
    finally:
        collector.stop(timeout_seconds=1.0)


@pytest.mark.parametrize("interval_seconds", [0.0, -1.0])
def test_collector_rejects_invalid_interval(
    interval_seconds: float,
) -> None:
    backend = CountingBackend(target_count=1)

    with pytest.raises(
        ValueError,
        match="interval_seconds must be positive",
    ):
        TelemetryCollector(
            backend,
            interval_seconds=interval_seconds,
        )


def test_stop_before_start_is_safe() -> None:
    backend = CountingBackend(target_count=1)
    collector = TelemetryCollector(
        backend,
        interval_seconds=0.01,
    )

    collector.stop()

    assert collector.is_running is False

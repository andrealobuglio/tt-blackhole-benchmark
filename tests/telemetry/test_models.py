"""Unit tests for telemetry domain models."""

from datetime import UTC, datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from tt_blackhole_benchmark.telemetry.models import (
    TelemetrySample,
    TelemetrySnapshot,
)


def make_sample(
    *,
    captured_at: datetime | None = None,
    monotonic_ns: int = 1_000_000_000,
    device_index: int = 0,
) -> TelemetrySample:
    """Create a valid telemetry sample for tests."""

    return TelemetrySample(
        captured_at=captured_at or datetime.now(UTC),
        monotonic_ns=monotonic_ns,
        device_index=device_index,
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


def test_sample_accepts_valid_values() -> None:
    timestamp = datetime.now(UTC)
    sample = make_sample(captured_at=timestamp)

    assert sample.captured_at == timestamp
    assert sample.device_index == 0
    assert sample.power_w == 35.0
    assert sample.source == "tt-smi"


def test_sample_normalizes_timestamp_to_utc() -> None:
    local_timezone = timezone(timedelta(hours=2))
    local_timestamp = datetime(
        2026,
        9,
        16,
        12,
        30,
        tzinfo=local_timezone,
    )

    sample = make_sample(captured_at=local_timestamp)

    assert sample.captured_at.tzinfo == UTC
    assert sample.captured_at.hour == 10


def test_sample_rejects_naive_timestamp() -> None:
    naive_timestamp = datetime(2026, 9, 16, 10, 30)

    with pytest.raises(
        ValidationError,
        match="captured_at must be timezone-aware",
    ):
        make_sample(captured_at=naive_timestamp)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("power_w", -1.0),
        ("voltage_v", -1.0),
        ("current_a", -1.0),
        ("aiclk_mhz", -1.0),
        ("fan_speed_percent", 101.0),
        ("heartbeat", -1),
    ],
)
def test_sample_rejects_invalid_measurements(
    field: str,
    value: float,
) -> None:
    values = make_sample().model_dump()
    values[field] = value

    with pytest.raises(ValidationError):
        TelemetrySample.model_validate(values)


def test_snapshot_accepts_multiple_devices() -> None:
    timestamp = datetime.now(UTC)
    monotonic_ns = 2_000_000_000

    first = make_sample(
        captured_at=timestamp,
        monotonic_ns=monotonic_ns,
        device_index=0,
    )
    second = make_sample(
        captured_at=timestamp,
        monotonic_ns=monotonic_ns,
        device_index=1,
    )

    snapshot = TelemetrySnapshot(
        captured_at=timestamp,
        monotonic_ns=monotonic_ns,
        samples=(first, second),
    )

    assert len(snapshot.samples) == 2


def test_snapshot_rejects_duplicate_device_indices() -> None:
    timestamp = datetime.now(UTC)
    first = make_sample(captured_at=timestamp, device_index=0)
    second = make_sample(captured_at=timestamp, device_index=0)

    with pytest.raises(
        ValidationError,
        match="duplicate device indices",
    ):
        TelemetrySnapshot(
            captured_at=timestamp,
            monotonic_ns=1_000_000_000,
            samples=(first, second),
        )


def test_snapshot_rejects_inconsistent_timestamp() -> None:
    snapshot_timestamp = datetime.now(UTC)
    sample_timestamp = snapshot_timestamp + timedelta(milliseconds=1)

    sample = make_sample(captured_at=sample_timestamp)

    with pytest.raises(
        ValidationError,
        match="Sample captured_at must match",
    ):
        TelemetrySnapshot(
            captured_at=snapshot_timestamp,
            monotonic_ns=1_000_000_000,
            samples=(sample,),
        )


def test_snapshot_rejects_inconsistent_monotonic_time() -> None:
    timestamp = datetime.now(UTC)
    sample = make_sample(
        captured_at=timestamp,
        monotonic_ns=2_000_000_000,
    )

    with pytest.raises(
        ValidationError,
        match="Sample monotonic_ns must match",
    ):
        TelemetrySnapshot(
            captured_at=timestamp,
            monotonic_ns=1_000_000_000,
            samples=(sample,),
        )

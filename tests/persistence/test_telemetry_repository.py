"""Tests for raw telemetry persistence."""

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tt_blackhole_benchmark.models import Workload
from tt_blackhole_benchmark.persistence.campaign import (
    create_campaign_workspace,
)
from tt_blackhole_benchmark.persistence.run_repository import (
    create_run_workspace,
)
from tt_blackhole_benchmark.persistence.telemetry_repository import (
    record_telemetry_acquisition,
    record_telemetry_failure,
)
from tt_blackhole_benchmark.telemetry.models import (
    TelemetryAcquisition,
    TelemetryCollectionFailure,
    TelemetrySample,
    TelemetrySnapshot,
)


def make_campaign_and_run(tmp_path: Path):
    campaign = create_campaign_workspace(
        results_directory=tmp_path,
        name="telemetry-tests",
        configuration={},
        environment={},
        created_at=datetime(2026, 9, 17, tzinfo=UTC),
        campaign_id="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
    )

    run = create_run_workspace(
        campaign=campaign,
        model="test-model",
        workload=Workload(
            input_tokens=64,
            output_tokens=8,
            batch_size=1,
            request_count=1,
        ),
        repetition_index=0,
        configuration={},
        prompts=[],
        run_id="11111111-2222-3333-4444-555555555555",
    )

    return campaign, run


def make_sample(
    *,
    captured_at: datetime,
    monotonic_ns: int,
    device_index: int,
    power_w: float,
) -> TelemetrySample:
    return TelemetrySample(
        captured_at=captured_at,
        monotonic_ns=monotonic_ns,
        device_index=device_index,
        bus_id=f"0000:0{device_index + 1}:00.0",
        board_type="p150a",
        power_w=power_w,
        voltage_v=0.74,
        current_a=48.0,
        asic_temperature_c=31.5,
        aiclk_mhz=800.0,
        fan_speed_percent=38.0,
        heartbeat=112_416,
        source="tt-smi",
    )


def test_record_telemetry_acquisition_for_multiple_devices(
    tmp_path: Path,
) -> None:
    campaign, run = make_campaign_and_run(tmp_path)
    captured_at = datetime(2026, 9, 17, 10, 30, tzinfo=UTC)

    samples = (
        make_sample(
            captured_at=captured_at,
            monotonic_ns=10_000,
            device_index=0,
            power_w=35.0,
        ),
        make_sample(
            captured_at=captured_at,
            monotonic_ns=10_000,
            device_index=1,
            power_w=37.0,
        ),
    )

    acquisition = TelemetryAcquisition(
        snapshot=TelemetrySnapshot(
            captured_at=captured_at,
            monotonic_ns=10_000,
            samples=samples,
        ),
        command=("persistent-tt-smi-worker",),
        exit_code=0,
        stderr="backend diagnostic",
    )

    record_telemetry_acquisition(
        campaign=campaign,
        run=run,
        acquisition=acquisition,
    )

    with sqlite3.connect(campaign.database_path) as connection:
        rows = connection.execute(
            """
            SELECT
                run_id,
                captured_at,
                monotonic_ns,
                device_index,
                bus_id,
                board_type,
                power_w,
                voltage_v,
                current_a,
                asic_temperature_c,
                aiclk_mhz,
                fan_speed_percent,
                heartbeat,
                source,
                backend_command,
                backend_exit_code,
                backend_stderr
            FROM telemetry_samples
            ORDER BY device_index
            """
        ).fetchall()

    assert len(rows) == 2
    assert rows[0] == (
        run.run_id,
        "2026-09-17T10:30:00+00:00",
        10_000,
        0,
        "0000:01:00.0",
        "p150a",
        35.0,
        0.74,
        48.0,
        31.5,
        800.0,
        38.0,
        112_416,
        "tt-smi",
        json.dumps(["persistent-tt-smi-worker"]),
        0,
        "backend diagnostic",
    )
    assert rows[1][3] == 1
    assert rows[1][6] == 37.0


def test_record_telemetry_failure(
    tmp_path: Path,
) -> None:
    campaign, run = make_campaign_and_run(tmp_path)

    failure = TelemetryCollectionFailure(
        captured_at=datetime(2026, 9, 17, 10, 31, tzinfo=UTC),
        monotonic_ns=20_000,
        message="telemetry request timed out",
    )

    record_telemetry_failure(
        campaign=campaign,
        run=run,
        failure=failure,
    )

    with sqlite3.connect(campaign.database_path) as connection:
        row = connection.execute(
            """
            SELECT
                run_id,
                captured_at,
                monotonic_ns,
                message
            FROM telemetry_failures
            """
        ).fetchone()

    assert row == (
        run.run_id,
        "2026-09-17T10:31:00+00:00",
        20_000,
        "telemetry request timed out",
    )


def test_duplicate_sample_is_rejected_atomically(
    tmp_path: Path,
) -> None:
    campaign, run = make_campaign_and_run(tmp_path)
    captured_at = datetime(2026, 9, 17, tzinfo=UTC)
    sample = make_sample(
        captured_at=captured_at,
        monotonic_ns=30_000,
        device_index=0,
        power_w=35.0,
    )
    acquisition = TelemetryAcquisition(
        snapshot=TelemetrySnapshot(
            captured_at=captured_at,
            monotonic_ns=30_000,
            samples=(sample,),
        ),
        command=("worker",),
        exit_code=0,
    )

    record_telemetry_acquisition(
        campaign=campaign,
        run=run,
        acquisition=acquisition,
    )

    from tt_blackhole_benchmark.persistence.telemetry_repository import (
        TelemetryRepositoryError,
    )

    with pytest.raises(TelemetryRepositoryError):
        record_telemetry_acquisition(
            campaign=campaign,
            run=run,
            acquisition=acquisition,
        )

    with sqlite3.connect(campaign.database_path) as connection:
        count = connection.execute("SELECT COUNT(*) FROM telemetry_samples").fetchone()[0]

    assert count == 1

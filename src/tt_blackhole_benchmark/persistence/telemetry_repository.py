"""Persistence of raw telemetry observations."""

from __future__ import annotations

import json

from tt_blackhole_benchmark.persistence.campaign import CampaignWorkspace
from tt_blackhole_benchmark.persistence.database import connect_database
from tt_blackhole_benchmark.persistence.run_repository import RunWorkspace
from tt_blackhole_benchmark.telemetry.models import (
    TelemetryAcquisition,
    TelemetryCollectionFailure,
)


class TelemetryRepositoryError(RuntimeError):
    """Raised when raw telemetry cannot be persisted."""


def _validate_run_campaign(
    campaign: CampaignWorkspace,
    run: RunWorkspace,
) -> None:
    if run.campaign_id != campaign.campaign_id:
        raise ValueError("Run does not belong to the supplied campaign")


def record_telemetry_acquisition(
    *,
    campaign: CampaignWorkspace,
    run: RunWorkspace,
    acquisition: TelemetryAcquisition,
) -> None:
    """Persist every device sample from one backend acquisition."""

    _validate_run_campaign(campaign, run)

    backend_command = json.dumps(
        list(acquisition.command),
        ensure_ascii=False,
    )

    rows = [
        (
            run.run_id,
            sample.captured_at.isoformat(),
            sample.monotonic_ns,
            sample.device_index,
            sample.bus_id,
            sample.board_type,
            sample.power_w,
            sample.voltage_v,
            sample.current_a,
            sample.asic_temperature_c,
            sample.aiclk_mhz,
            sample.fan_speed_percent,
            sample.heartbeat,
            sample.source,
            backend_command,
            acquisition.exit_code,
            acquisition.stderr,
        )
        for sample in acquisition.snapshot.samples
    ]

    try:
        with connect_database(campaign.database_path) as connection:
            connection.executemany(
                """
                INSERT INTO telemetry_samples (
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
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            connection.commit()
    except Exception as error:
        raise TelemetryRepositoryError(
            f"Unable to persist telemetry for run {run.run_id}"
        ) from error


def record_telemetry_failure(
    *,
    campaign: CampaignWorkspace,
    run: RunWorkspace,
    failure: TelemetryCollectionFailure,
) -> None:
    """Persist one raw telemetry collection failure."""

    _validate_run_campaign(campaign, run)

    try:
        with connect_database(campaign.database_path) as connection:
            connection.execute(
                """
                INSERT INTO telemetry_failures (
                    run_id,
                    captured_at,
                    monotonic_ns,
                    message
                )
                VALUES (?, ?, ?, ?)
                """,
                (
                    run.run_id,
                    failure.captured_at.isoformat(),
                    failure.monotonic_ns,
                    failure.message,
                ),
            )
            connection.commit()
    except Exception as error:
        raise TelemetryRepositoryError(
            f"Unable to persist telemetry failure for run {run.run_id}"
        ) from error

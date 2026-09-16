"""Unit tests for tt-smi telemetry parsing."""

import json
from datetime import UTC, datetime

import pytest

from tt_blackhole_benchmark.telemetry.parser import (
    TelemetryParseError,
    parse_tt_smi_output,
)


def valid_document() -> dict[str, object]:
    return {
        "time": "2026-09-16T10:42:58.085188",
        "device_info": [
            {
                "board_info": {
                    "bus_id": "0000:01:00.0",
                    "board_type": "p150a",
                },
                "telemetry": {
                    "voltage": "0.74",
                    "current": " 48.0",
                    "power": " 35.0",
                    "aiclk": " 800",
                    "asic_temperature": "30.7",
                    "fan_speed": " 38",
                    "heartbeat": "112416",
                },
            }
        ],
    }


def test_parse_valid_tt_smi_document() -> None:
    captured_at = datetime(2026, 9, 16, 10, 42, tzinfo=UTC)
    output = json.dumps(valid_document())

    snapshot = parse_tt_smi_output(
        output,
        captured_at=captured_at,
        monotonic_ns=10_000,
    )

    sample = snapshot.samples[0]

    assert snapshot.captured_at == captured_at
    assert sample.bus_id == "0000:01:00.0"
    assert sample.board_type == "p150a"
    assert sample.power_w == 35.0
    assert sample.voltage_v == 0.74
    assert sample.current_a == 48.0
    assert sample.asic_temperature_c == 30.7
    assert sample.aiclk_mhz == 800.0
    assert sample.fan_speed_percent == 38.0
    assert sample.heartbeat == 112_416


def test_parse_json_surrounded_by_process_noise() -> None:
    output = (
        "UMD topology discovery log\n"
        + json.dumps(valid_document())
        + "\ndouble free or corruption (!prev)\nAborted"
    )

    snapshot = parse_tt_smi_output(
        output,
        captured_at=datetime.now(UTC),
        monotonic_ns=20_000,
    )

    assert len(snapshot.samples) == 1
    assert snapshot.samples[0].power_w == 35.0


@pytest.mark.parametrize(
    "output",
    [
        "",
        "no JSON here",
        "{invalid JSON",
        '{"unrelated": true}',
    ],
)
def test_parse_rejects_output_without_tt_smi_document(
    output: str,
) -> None:
    with pytest.raises(
        TelemetryParseError,
        match="No valid tt-smi JSON document",
    ):
        parse_tt_smi_output(
            output,
            captured_at=datetime.now(UTC),
            monotonic_ns=30_000,
        )


def test_parse_rejects_empty_device_list() -> None:
    output = json.dumps({"device_info": []})

    with pytest.raises(
        TelemetryParseError,
        match="device_info must be a non-empty list",
    ):
        parse_tt_smi_output(
            output,
            captured_at=datetime.now(UTC),
            monotonic_ns=40_000,
        )


def test_parse_rejects_missing_measurement() -> None:
    document = valid_document()
    devices = document["device_info"]
    assert isinstance(devices, list)

    device = devices[0]
    assert isinstance(device, dict)

    telemetry = device["telemetry"]
    assert isinstance(telemetry, dict)
    del telemetry["power"]

    with pytest.raises(
        TelemetryParseError,
        match=r"device_info\[0\]\.telemetry\.power must be numeric",
    ):
        parse_tt_smi_output(
            json.dumps(document),
            captured_at=datetime.now(UTC),
            monotonic_ns=50_000,
        )

"""Parsing of JSON telemetry produced by tt-smi."""

import json
from collections.abc import Mapping
from datetime import datetime
from typing import Any

from tt_blackhole_benchmark.telemetry.models import (
    TelemetrySample,
    TelemetrySnapshot,
)


class TelemetryParseError(ValueError):
    """Raised when tt-smi output cannot be parsed."""


def _extract_json_document(output: str) -> Mapping[str, Any]:
    decoder = json.JSONDecoder()

    for position, character in enumerate(output):
        if character != "{":
            continue

        try:
            document, _ = decoder.raw_decode(output[position:])
        except json.JSONDecodeError:
            continue

        if isinstance(document, dict) and "device_info" in document:
            return document

    raise TelemetryParseError("No valid tt-smi JSON document found in output")


def _require_mapping(
    value: object,
    field_path: str,
) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise TelemetryParseError(f"{field_path} must be a mapping")
    return value


def _require_string(
    mapping: Mapping[str, Any],
    key: str,
    field_path: str,
) -> str:
    value = mapping.get(key)

    if not isinstance(value, str) or not value.strip():
        raise TelemetryParseError(f"{field_path}.{key} must be a non-empty string")

    return value.strip()


def _require_float(
    mapping: Mapping[str, Any],
    key: str,
    field_path: str,
) -> float:
    value = mapping.get(key)

    if isinstance(value, bool) or not isinstance(
        value,
        (str, int, float),
    ):
        raise TelemetryParseError(f"{field_path}.{key} must be numeric")

    try:
        return float(value)
    except ValueError as error:
        raise TelemetryParseError(f"{field_path}.{key} must be numeric") from error


def _require_int(
    mapping: Mapping[str, Any],
    key: str,
    field_path: str,
) -> int:
    value = mapping.get(key)

    if isinstance(value, bool) or not isinstance(
        value,
        (str, int),
    ):
        raise TelemetryParseError(f"{field_path}.{key} must be an integer")

    try:
        return int(value)
    except ValueError as error:
        raise TelemetryParseError(f"{field_path}.{key} must be an integer") from error


def parse_tt_smi_output(
    output: str,
    *,
    captured_at: datetime,
    monotonic_ns: int,
) -> TelemetrySnapshot:
    """Parse one tt-smi invocation into a telemetry snapshot."""

    document = _extract_json_document(output)
    device_info = document.get("device_info")

    if not isinstance(device_info, list) or not device_info:
        raise TelemetryParseError("device_info must be a non-empty list")

    samples: list[TelemetrySample] = []

    for device_index, raw_device in enumerate(device_info):
        device_path = f"device_info[{device_index}]"
        device = _require_mapping(raw_device, device_path)
        board = _require_mapping(
            device.get("board_info"),
            f"{device_path}.board_info",
        )
        telemetry = _require_mapping(
            device.get("telemetry"),
            f"{device_path}.telemetry",
        )

        samples.append(
            TelemetrySample(
                captured_at=captured_at,
                monotonic_ns=monotonic_ns,
                device_index=device_index,
                bus_id=_require_string(
                    board,
                    "bus_id",
                    f"{device_path}.board_info",
                ),
                board_type=_require_string(
                    board,
                    "board_type",
                    f"{device_path}.board_info",
                ),
                power_w=_require_float(
                    telemetry,
                    "power",
                    f"{device_path}.telemetry",
                ),
                voltage_v=_require_float(
                    telemetry,
                    "voltage",
                    f"{device_path}.telemetry",
                ),
                current_a=_require_float(
                    telemetry,
                    "current",
                    f"{device_path}.telemetry",
                ),
                asic_temperature_c=_require_float(
                    telemetry,
                    "asic_temperature",
                    f"{device_path}.telemetry",
                ),
                aiclk_mhz=_require_float(
                    telemetry,
                    "aiclk",
                    f"{device_path}.telemetry",
                ),
                fan_speed_percent=_require_float(
                    telemetry,
                    "fan_speed",
                    f"{device_path}.telemetry",
                ),
                heartbeat=_require_int(
                    telemetry,
                    "heartbeat",
                    f"{device_path}.telemetry",
                ),
            )
        )

    return TelemetrySnapshot(
        captured_at=captured_at,
        monotonic_ns=monotonic_ns,
        samples=tuple(samples),
    )

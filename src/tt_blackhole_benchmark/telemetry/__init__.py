"""Telemetry acquisition for Tenstorrent accelerators."""

from tt_blackhole_benchmark.telemetry.models import (
    TelemetrySample,
    TelemetrySnapshot,
)
from tt_blackhole_benchmark.telemetry.parser import (
    TelemetryParseError,
    parse_tt_smi_output,
)

__all__ = [
    "TelemetryParseError",
    "TelemetrySample",
    "TelemetrySnapshot",
    "parse_tt_smi_output",
]

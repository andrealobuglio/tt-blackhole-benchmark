"""Subprocess backend for tt-smi telemetry acquisition."""

import subprocess
import time
from collections.abc import Sequence
from datetime import UTC, datetime

from tt_blackhole_benchmark.telemetry.models import (
    TelemetryAcquisition,
)
from tt_blackhole_benchmark.telemetry.parser import (
    TelemetryParseError,
    parse_tt_smi_output,
)


class TelemetryBackendError(RuntimeError):
    """Raised when telemetry acquisition fails."""


class TtSmiBackend:
    """Acquire Tenstorrent telemetry through tt-smi."""

    def __init__(
        self,
        command: Sequence[str] = ("tt-smi", "-s"),
        *,
        timeout_seconds: float = 10.0,
    ) -> None:
        if not command:
            raise ValueError("command must not be empty")

        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

        self._command = tuple(command)
        self._timeout_seconds = timeout_seconds

    @property
    def command(self) -> tuple[str, ...]:
        """Return the configured command."""

        return self._command

    def acquire(self) -> TelemetryAcquisition:
        """Execute tt-smi and return parsed telemetry with diagnostics."""

        started_at = datetime.now(UTC)
        started_monotonic_ns = time.monotonic_ns()

        try:
            process: subprocess.CompletedProcess[str] = subprocess.run(
                self._command,
                capture_output=True,
                text=True,
                timeout=self._timeout_seconds,
                check=False,
            )
        except FileNotFoundError as error:
            raise TelemetryBackendError(
                f"Telemetry command not found: {self._command[0]}"
            ) from error
        except subprocess.TimeoutExpired as error:
            raise TelemetryBackendError(
                f"Telemetry command timed out after {self._timeout_seconds:.3f} seconds"
            ) from error
        except OSError as error:
            raise TelemetryBackendError(f"Cannot execute telemetry command: {error}") from error

        finished_at = datetime.now(UTC)
        finished_monotonic_ns = time.monotonic_ns()

        duration_ns = finished_monotonic_ns - started_monotonic_ns
        midpoint_monotonic_ns = started_monotonic_ns + duration_ns // 2
        midpoint_timestamp = started_at + (finished_at - started_at) / 2

        combined_output = "\n".join(part for part in (process.stdout, process.stderr) if part)

        try:
            snapshot = parse_tt_smi_output(
                combined_output,
                captured_at=midpoint_timestamp,
                monotonic_ns=midpoint_monotonic_ns,
            )
        except TelemetryParseError as error:
            raise TelemetryBackendError(
                f"Telemetry command produced no valid telemetry (exit code {process.returncode})"
            ) from error

        return TelemetryAcquisition(
            snapshot=snapshot,
            command=self._command,
            exit_code=process.returncode,
            duration_seconds=duration_ns / 1_000_000_000,
            stderr=process.stderr,
        )

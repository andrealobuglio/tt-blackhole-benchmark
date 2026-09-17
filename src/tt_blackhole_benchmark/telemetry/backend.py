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

        try:
            process: subprocess.CompletedProcess[str] = subprocess.run(
                self._command,
                capture_output=True,
                text=True,
                timeout=self._timeout_seconds,
                check=False,
            )
            captured_at = datetime.now(UTC)
            monotonic_ns = time.monotonic_ns()
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

        combined_output = "\n".join(part for part in (process.stdout, process.stderr) if part)

        try:
            snapshot = parse_tt_smi_output(
                combined_output,
                captured_at=captured_at,
                monotonic_ns=monotonic_ns,
            )
        except TelemetryParseError as error:
            raise TelemetryBackendError(
                f"Telemetry command produced no valid telemetry (exit code {process.returncode})"
            ) from error

        return TelemetryAcquisition(
            snapshot=snapshot,
            command=self._command,
            exit_code=process.returncode,
            stderr=process.stderr,
        )

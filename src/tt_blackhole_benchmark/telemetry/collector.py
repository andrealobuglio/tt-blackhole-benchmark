"""Periodic background collection of accelerator telemetry."""

import threading
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Protocol

from tt_blackhole_benchmark.telemetry.backend import (
    TelemetryBackendError,
)
from tt_blackhole_benchmark.telemetry.models import (
    TelemetryAcquisition,
    TelemetryCollectionFailure,
)


class TelemetryBackend(Protocol):
    """Backend interface required by the collector."""

    def acquire(self) -> TelemetryAcquisition:
        """Acquire one telemetry snapshot."""
        ...


class TelemetryCollectorError(RuntimeError):
    """Raised for invalid collector lifecycle operations."""


class TelemetryCollector:
    """Collect telemetry periodically in a background thread."""

    def __init__(
        self,
        backend: TelemetryBackend,
        *,
        interval_seconds: float,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")

        self._backend = backend
        self._interval_ns = int(interval_seconds * 1_000_000_000)

        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

        self._acquisitions: list[TelemetryAcquisition] = []
        self._failures: list[TelemetryCollectionFailure] = []

    @property
    def is_running(self) -> bool:
        """Whether the collection thread is currently alive."""

        thread = self._thread
        return thread is not None and thread.is_alive()

    @property
    def acquisitions(self) -> Sequence[TelemetryAcquisition]:
        """Return an immutable copy of collected acquisitions."""

        with self._lock:
            return tuple(self._acquisitions)

    @property
    def failures(self) -> Sequence[TelemetryCollectionFailure]:
        """Return an immutable copy of recoverable failures."""

        with self._lock:
            return tuple(self._failures)

    def start(self) -> None:
        """Start background telemetry collection."""

        if self.is_running:
            raise TelemetryCollectorError("Telemetry collector is already running")

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="tt-telemetry-collector",
            daemon=True,
        )
        self._thread.start()

    def stop(
        self,
        *,
        timeout_seconds: float | None = None,
    ) -> None:
        """Request collection shutdown and wait for the thread."""

        if timeout_seconds is not None and timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

        thread = self._thread

        if thread is None:
            return

        self._stop_event.set()
        thread.join(timeout=timeout_seconds)

        if thread.is_alive():
            raise TelemetryCollectorError("Telemetry collector did not stop before timeout")

        self._thread = None

    def _record_acquisition(
        self,
        acquisition: TelemetryAcquisition,
    ) -> None:
        with self._lock:
            self._acquisitions.append(acquisition)

    def _record_failure(self, error: Exception) -> None:
        failure = TelemetryCollectionFailure(
            captured_at=datetime.now(UTC),
            monotonic_ns=time.monotonic_ns(),
            message=str(error),
        )

        with self._lock:
            self._failures.append(failure)

    def _run(self) -> None:
        next_deadline_ns = time.monotonic_ns()

        while not self._stop_event.is_set():
            now_ns = time.monotonic_ns()
            remaining_ns = next_deadline_ns - now_ns

            if remaining_ns > 0:
                should_stop = self._stop_event.wait(remaining_ns / 1_000_000_000)
                if should_stop:
                    break

            try:
                acquisition = self._backend.acquire()
            except TelemetryBackendError as error:
                self._record_failure(error)
            else:
                self._record_acquisition(acquisition)

            now_ns = time.monotonic_ns()
            next_deadline_ns += self._interval_ns

            if next_deadline_ns <= now_ns:
                missed_intervals = (now_ns - next_deadline_ns) // self._interval_ns + 1
                next_deadline_ns += missed_intervals * self._interval_ns

    def __enter__(self) -> "TelemetryCollector":
        self.start()
        return self

    def __exit__(
        self,
        exception_type: object,
        exception: object,
        traceback: object,
    ) -> None:
        self.stop()

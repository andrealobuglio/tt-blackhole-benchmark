"""Persistent process-isolated backend for tt-smi telemetry."""

from collections.abc import Callable, Mapping
from datetime import datetime
from multiprocessing import get_context
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess

from tt_blackhole_benchmark.telemetry.backend import (
    TelemetryBackendError,
)
from tt_blackhole_benchmark.telemetry.models import (
    TelemetryAcquisition,
)
from tt_blackhole_benchmark.telemetry.parser import (
    TelemetryParseError,
    parse_tt_smi_output,
)
from tt_blackhole_benchmark.telemetry.worker import (
    run_tt_smi_worker,
)

WorkerTarget = Callable[[Connection], object]


class PersistentTtSmiBackend:
    """Access tt-smi through an isolated persistent worker."""

    def __init__(
        self,
        *,
        startup_timeout_seconds: float = 10.0,
        request_timeout_seconds: float = 5.0,
        worker_target: WorkerTarget = run_tt_smi_worker,
    ) -> None:
        if startup_timeout_seconds <= 0:
            raise ValueError("startup_timeout_seconds must be positive")

        if request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be positive")

        self._startup_timeout_seconds = startup_timeout_seconds
        self._request_timeout_seconds = request_timeout_seconds
        self._worker_target = worker_target

        self._process: BaseProcess | None = None
        self._connection: Connection | None = None

    @property
    def is_running(self) -> bool:
        """Whether the persistent worker is alive."""

        return self._process is not None and self._process.is_alive()

    def start(self) -> None:
        """Start the worker and wait for initialization."""

        if self.is_running:
            raise TelemetryBackendError("Persistent telemetry backend is already running")

        context = get_context("spawn")
        parent_connection, worker_connection = context.Pipe(duplex=True)
        process = context.Process(
            target=self._worker_target,
            args=(worker_connection,),
            name="tt-smi-telemetry-worker",
            daemon=True,
        )

        self._connection = parent_connection
        self._process = process

        try:
            process.start()
            worker_connection.close()

            response = self._receive_message(timeout_seconds=self._startup_timeout_seconds)

            if response["kind"] != "ready":
                message = response.get(
                    "message",
                    "unexpected worker response",
                )
                raise TelemetryBackendError(f"Telemetry worker failed to start: {message}")

        except Exception:
            worker_connection.close()
            self._terminate_worker()
            raise

    def acquire(self) -> TelemetryAcquisition:
        """Request one telemetry sample from the worker."""

        connection = self._require_connection()

        try:
            connection.send("acquire")
        except (BrokenPipeError, EOFError, OSError) as error:
            raise TelemetryBackendError("Cannot send request to telemetry worker") from error

        response = self._receive_message(timeout_seconds=self._request_timeout_seconds)

        kind = response["kind"]

        if kind == "error":
            message = response.get(
                "message",
                "unknown telemetry worker error",
            )
            raise TelemetryBackendError(f"Telemetry worker acquisition failed: {message}")

        if kind != "sample":
            raise TelemetryBackendError(f"Unexpected telemetry worker response: {kind}")

        payload = response.get("payload")

        if not isinstance(payload, str):
            raise TelemetryBackendError("Telemetry worker returned an invalid payload")

        captured_at_raw = response.get("captured_at")
        monotonic_ns = response.get("monotonic_ns")

        if not isinstance(captured_at_raw, str):
            raise TelemetryBackendError("Telemetry worker returned an invalid timestamp")

        try:
            captured_at = datetime.fromisoformat(captured_at_raw)
        except ValueError as error:
            raise TelemetryBackendError("Telemetry worker returned an invalid timestamp") from error

        if captured_at.tzinfo is None or captured_at.utcoffset() is None:
            raise TelemetryBackendError("Telemetry worker returned a naive timestamp")

        if isinstance(monotonic_ns, bool) or not isinstance(monotonic_ns, int) or monotonic_ns < 0:
            raise TelemetryBackendError("Telemetry worker returned an invalid monotonic timestamp")

        try:
            snapshot = parse_tt_smi_output(
                payload,
                captured_at=captured_at,
                monotonic_ns=monotonic_ns,
            )
        except TelemetryParseError as error:
            raise TelemetryBackendError("Telemetry worker returned invalid tt-smi JSON") from error

        return TelemetryAcquisition(
            snapshot=snapshot,
            command=("persistent-tt-smi-worker",),
            exit_code=0,
        )

    def close(self) -> None:
        """Stop and release the telemetry worker."""

        connection = self._connection
        process = self._process

        if connection is None or process is None:
            return

        if process.is_alive():
            try:
                connection.send("close")

                response = self._receive_message(timeout_seconds=self._request_timeout_seconds)

                if response["kind"] != "closed":
                    raise TelemetryBackendError("Unexpected telemetry worker shutdown response")
            except TelemetryBackendError:
                pass
            except (BrokenPipeError, EOFError, OSError):
                pass

        self._terminate_worker()

    def _require_connection(self) -> Connection:
        process = self._process
        connection = self._connection

        if process is None or connection is None or not process.is_alive():
            raise TelemetryBackendError("Persistent telemetry backend is not running")

        return connection

    def _receive_message(
        self,
        *,
        timeout_seconds: float,
    ) -> Mapping[str, object]:
        connection = self._connection

        if connection is None:
            raise TelemetryBackendError("Telemetry worker connection is unavailable")

        if not connection.poll(timeout_seconds):
            raise TelemetryBackendError("Timed out waiting for telemetry worker")

        try:
            message = connection.recv()
        except (EOFError, OSError) as error:
            raise TelemetryBackendError(
                "Telemetry worker connection closed unexpectedly"
            ) from error

        if not isinstance(message, dict):
            raise TelemetryBackendError("Telemetry worker returned a malformed response")

        kind = message.get("kind")

        if not isinstance(kind, str):
            raise TelemetryBackendError("Telemetry worker response has no valid kind")

        return message

    def _terminate_worker(self) -> None:
        connection = self._connection
        process = self._process

        if connection is not None:
            connection.close()

        if process is not None:
            process.join(timeout=1.0)

            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)

            if process.is_alive():
                process.kill()
                process.join(timeout=1.0)

        self._connection = None
        self._process = None

    def __enter__(self) -> "PersistentTtSmiBackend":
        self.start()
        return self

    def __exit__(
        self,
        exception_type: object,
        exception: object,
        traceback: object,
    ) -> None:
        self.close()

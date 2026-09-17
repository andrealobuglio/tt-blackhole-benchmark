"""Isolated persistent worker for direct tt-smi telemetry access."""

import os
from multiprocessing.connection import Connection
from typing import NoReturn, Protocol


class TelemetrySource(Protocol):
    """Direct telemetry source used inside the worker process."""

    def update_telem(self) -> None:
        """Refresh telemetry values."""
        ...

    def get_logs_json(self) -> str:
        """Return the current tt-smi snapshot as JSON."""
        ...


def serve_worker_requests(
    connection: Connection,
    telemetry_source: TelemetrySource,
) -> None:
    """Serve telemetry requests until close or pipe termination."""

    while True:
        try:
            command = connection.recv()
        except EOFError:
            return

        if command == "acquire":
            try:
                telemetry_source.update_telem()
                payload = telemetry_source.get_logs_json()
            except Exception as error:
                connection.send(
                    {
                        "kind": "error",
                        "message": str(error),
                    }
                )
            else:
                connection.send(
                    {
                        "kind": "sample",
                        "payload": payload,
                    }
                )

        elif command == "close":
            connection.send({"kind": "closed"})
            return

        else:
            connection.send(
                {
                    "kind": "error",
                    "message": f"Unknown worker command: {command!r}",
                }
            )


def run_tt_smi_worker(connection: Connection) -> NoReturn:
    """Initialize tt-smi once and serve requests in isolation."""

    exit_code = 0

    try:
        from tt_smi import constants  # type: ignore[import-untyped]
        from tt_smi.backend import (  # type: ignore[import-untyped]
            TTSMIBackend,
        )
        from tt_umd import (
            TopologyDiscovery,
        )

        cluster_descriptor, devices = TopologyDiscovery.discover(
            options=constants.get_default_discovery_options()
        )

        if not devices:
            raise RuntimeError("No Tenstorrent devices detected")

        backend = TTSMIBackend(
            devices=devices,
            umd_cluster_descriptor=cluster_descriptor,
            pretty_output=False,
        )

        connection.send({"kind": "ready"})
        serve_worker_requests(connection, backend)

    except Exception as error:
        exit_code = 1

        try:
            connection.send(
                {
                    "kind": "fatal",
                    "message": str(error),
                }
            )
        except (BrokenPipeError, EOFError, OSError):
            pass

    finally:
        connection.close()

        # tt-smi 5.2.0 aborts during native object destruction on
        # the current riscv64 platform. The worker owns no persistent
        # application state, so bypass native teardown and let the OS
        # close its process resources.
        os._exit(exit_code)

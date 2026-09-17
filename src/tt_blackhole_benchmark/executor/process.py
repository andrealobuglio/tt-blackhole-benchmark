"""Controlled subprocess execution for benchmark workloads."""

import os
import signal
import subprocess
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

from tt_blackhole_benchmark.executor.models import (
    ProcessExecution,
)


class ProcessExecutorError(RuntimeError):
    """Raised when a benchmark process cannot be started."""


def _terminate_process_group(
    process: subprocess.Popen[str],
    *,
    grace_seconds: float,
) -> tuple[str, str]:
    """Terminate a process group and collect remaining output."""

    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass

    try:
        return process.communicate(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

        return process.communicate()


def execute_process(
    command: Sequence[str],
    *,
    working_directory: str | Path,
    timeout_seconds: float,
    environment: Mapping[str, str] | None = None,
    termination_grace_seconds: float = 5.0,
) -> ProcessExecution:
    """Execute a command and return only directly observed values."""

    if not command:
        raise ValueError("command must not be empty")

    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")

    if termination_grace_seconds <= 0:
        raise ValueError("termination_grace_seconds must be positive")

    resolved_directory = Path(working_directory).resolve()

    if not resolved_directory.is_dir():
        raise ProcessExecutorError(f"Working directory does not exist: {resolved_directory}")

    process_environment = os.environ.copy()

    if environment is not None:
        process_environment.update(environment)

    started_at = datetime.now(UTC)
    started_monotonic_ns = time.monotonic_ns()

    try:
        process: subprocess.Popen[str] = subprocess.Popen(
            tuple(command),
            cwd=resolved_directory,
            env=process_environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
    except OSError as error:
        raise ProcessExecutorError(f"Cannot start benchmark process: {error}") from error

    timed_out = False

    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
        return_code: int | None = process.returncode
    except subprocess.TimeoutExpired:
        timed_out = True
        return_code = None
        stdout, stderr = _terminate_process_group(
            process,
            grace_seconds=termination_grace_seconds,
        )

    finished_at = datetime.now(UTC)
    finished_monotonic_ns = time.monotonic_ns()

    return ProcessExecution(
        command=tuple(command),
        working_directory=resolved_directory,
        started_at=started_at,
        finished_at=finished_at,
        started_monotonic_ns=started_monotonic_ns,
        finished_monotonic_ns=finished_monotonic_ns,
        return_code=return_code,
        timed_out=timed_out,
        stdout=stdout,
        stderr=stderr,
    )

"""Unit tests for executor observation models."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from tt_blackhole_benchmark.executor.models import (
    ProcessExecution,
)


def valid_execution() -> ProcessExecution:
    started_at = datetime.now(UTC)

    return ProcessExecution(
        command=("python", "benchmark.py"),
        working_directory=Path("/workspace/benchmark"),
        started_at=started_at,
        finished_at=started_at + timedelta(seconds=1),
        started_monotonic_ns=1_000,
        finished_monotonic_ns=2_000,
        return_code=0,
        timed_out=False,
        stdout="benchmark output",
        stderr="",
    )


def test_process_execution_accepts_raw_observations() -> None:
    execution = valid_execution()

    assert execution.command == ("python", "benchmark.py")
    assert execution.return_code == 0
    assert execution.timed_out is False
    assert execution.stdout == "benchmark output"


def test_process_execution_rejects_naive_timestamp() -> None:
    values = valid_execution().model_dump()
    values["started_at"] = datetime(2026, 9, 17, 10, 0)

    with pytest.raises(
        ValidationError,
        match="timestamp must be timezone-aware",
    ):
        ProcessExecution.model_validate(values)


def test_process_execution_rejects_reversed_wall_time() -> None:
    values = valid_execution().model_dump()
    values["finished_at"] = values["started_at"] - timedelta(seconds=1)

    with pytest.raises(
        ValidationError,
        match="finished_at must not precede started_at",
    ):
        ProcessExecution.model_validate(values)


def test_process_execution_rejects_reversed_monotonic_time() -> None:
    values = valid_execution().model_dump()
    values["finished_monotonic_ns"] = 999

    with pytest.raises(
        ValidationError,
        match="finished_monotonic_ns must not precede",
    ):
        ProcessExecution.model_validate(values)


def test_timed_out_execution_rejects_return_code() -> None:
    values = valid_execution().model_dump()
    values["timed_out"] = True

    with pytest.raises(
        ValidationError,
        match="must not have a return code",
    ):
        ProcessExecution.model_validate(values)


def test_timed_out_execution_accepts_missing_return_code() -> None:
    values = valid_execution().model_dump()
    values["timed_out"] = True
    values["return_code"] = None

    execution = ProcessExecution.model_validate(values)

    assert execution.timed_out is True
    assert execution.return_code is None

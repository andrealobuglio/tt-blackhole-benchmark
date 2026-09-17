"""Unit tests for controlled subprocess execution."""

import os
from pathlib import Path

import pytest

from tt_blackhole_benchmark.executor.process import (
    ProcessExecutorError,
    execute_process,
)


def shell_command(source: str) -> tuple[str, ...]:
    return ("/bin/sh", "-c", source)


def test_execute_process_captures_output_and_exit_code(
    tmp_path: Path,
) -> None:

    execution = execute_process(
        shell_command("printf 'standard output\\n'; printf 'standard error\\n' >&2; exit 7"),
        working_directory=tmp_path,
        timeout_seconds=5.0,
    )
    assert execution.return_code == 7
    assert execution.timed_out is False
    assert execution.stdout.strip() == "standard output"
    assert execution.stderr.strip() == "standard error"


def test_execute_process_records_working_directory(
    tmp_path: Path,
) -> None:
    execution = execute_process(
        shell_command("pwd"),
        working_directory=tmp_path,
        timeout_seconds=5.0,
    )

    assert execution.return_code == 0
    assert execution.stdout.strip() == str(tmp_path.resolve())


def test_execute_process_passes_environment(
    tmp_path: Path,
) -> None:
    execution = execute_process(
        shell_command("printf '%s\\n' \"$TT_BENCHMARK_TEST_VALUE\""),
        working_directory=tmp_path,
        timeout_seconds=5.0,
        environment={"TT_BENCHMARK_TEST_VALUE": "expected-value"},
    )

    assert execution.return_code == 0
    assert execution.stdout.strip() == "expected-value"


def test_execute_process_enforces_timeout(
    tmp_path: Path,
) -> None:
    execution = execute_process(
        shell_command("printf 'started\\n'; sleep 10"),
        working_directory=tmp_path,
        timeout_seconds=3.0,
        termination_grace_seconds=1.0,
    )

    assert execution.timed_out is True
    assert execution.return_code is None
    assert "started" in execution.stdout


def test_execute_process_rejects_missing_command(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ProcessExecutorError,
        match="Cannot start benchmark process",
    ):
        execute_process(
            ("/definitely/missing/command",),
            working_directory=tmp_path,
            timeout_seconds=1.0,
        )


def test_execute_process_rejects_missing_directory(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ProcessExecutorError,
        match="Working directory does not exist",
    ):
        execute_process(
            shell_command("printf 'unused\\n'"),
            working_directory=tmp_path / "missing",
            timeout_seconds=1.0,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("timeout_seconds", 0.0),
        ("termination_grace_seconds", 0.0),
    ],
)
def test_execute_process_rejects_invalid_timeout(
    field: str,
    value: float,
    tmp_path: Path,
) -> None:
    arguments = {
        "command": shell_command("printf 'unused\\n'"),
        "working_directory": tmp_path,
        "timeout_seconds": 1.0,
        field: value,
    }

    with pytest.raises(
        ValueError,
        match="must be positive",
    ):
        execute_process(**arguments)


def test_parent_environment_is_not_modified(
    tmp_path: Path,
) -> None:
    variable = "TT_BENCHMARK_TEMPORARY_VALUE"
    os.environ.pop(variable, None)

    execute_process(
        shell_command("printf 'ok\\n'"),
        working_directory=tmp_path,
        timeout_seconds=2.0,
        environment={variable: "child-only"},
    )

    assert variable not in os.environ

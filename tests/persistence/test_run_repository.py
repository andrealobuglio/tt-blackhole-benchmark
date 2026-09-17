"""Tests for benchmark run persistence."""

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tt_blackhole_benchmark.executor.process import ProcessExecution
from tt_blackhole_benchmark.models import Workload
from tt_blackhole_benchmark.persistence.campaign import (
    create_campaign_workspace,
)
from tt_blackhole_benchmark.persistence.run_repository import (
    RunRepositoryError,
    create_run_workspace,
    record_process_execution,
)


def make_campaign(tmp_path: Path):
    return create_campaign_workspace(
        results_directory=tmp_path,
        name="run-tests",
        configuration={"campaign": "test"},
        environment={"hostname": "test-host"},
        created_at=datetime(2026, 9, 17, tzinfo=UTC),
        campaign_id="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
    )


def make_workload() -> Workload:
    return Workload(
        input_tokens=64,
        output_tokens=8,
        batch_size=1,
        request_count=1,
    )


def test_create_run_workspace(tmp_path: Path) -> None:
    campaign = make_campaign(tmp_path)

    run = create_run_workspace(
        campaign=campaign,
        model="Qwen/Qwen2.5-0.5B-Instruct",
        workload=make_workload(),
        repetition_index=0,
        configuration={"trace_enabled": True},
        prompts=[{"prompt": "benchmark prompt"}],
        run_id="11111111-2222-3333-4444-555555555555",
    )

    assert run.root.is_dir()
    assert run.configuration_path.is_file()
    assert run.prompts_path.is_file()
    assert not run.stdout_path.exists()
    assert not run.stderr_path.exists()

    assert json.loads(run.configuration_path.read_text(encoding="utf-8")) == {"trace_enabled": True}

    assert json.loads(run.prompts_path.read_text(encoding="utf-8")) == [
        {"prompt": "benchmark prompt"}
    ]


def test_run_database_contains_raw_request_and_relative_paths(
    tmp_path: Path,
) -> None:
    campaign = make_campaign(tmp_path)

    run = create_run_workspace(
        campaign=campaign,
        model="test-model",
        workload=make_workload(),
        repetition_index=3,
        configuration={},
        prompts=[{"prompt": "test"}],
        run_id="11111111-2222-3333-4444-555555555555",
    )

    with sqlite3.connect(campaign.database_path) as connection:
        row = connection.execute(
            """
            SELECT
                repetition_index,
                model,
                input_tokens_requested,
                output_tokens_requested,
                batch_size,
                request_count,
                status,
                stdout_path,
                stderr_path,
                prompt_path,
                configuration_path
            FROM runs
            WHERE run_id = ?
            """,
            (run.run_id,),
        ).fetchone()

    assert row == (
        3,
        "test-model",
        64,
        8,
        1,
        1,
        "created",
        f"runs/{run.run_id}/stdout.log",
        f"runs/{run.run_id}/stderr.log",
        f"runs/{run.run_id}/prompts.json",
        f"runs/{run.run_id}/configuration.json",
    )


def test_record_successful_process_execution(
    tmp_path: Path,
) -> None:
    campaign = make_campaign(tmp_path)
    run = create_run_workspace(
        campaign=campaign,
        model="test-model",
        workload=make_workload(),
        repetition_index=0,
        configuration={},
        prompts=[{"prompt": "test"}],
    )

    execution = ProcessExecution(
        command=("python", "benchmark.py"),
        working_directory=Path("/benchmark"),
        started_at=datetime(2026, 9, 17, 10, 0, tzinfo=UTC),
        finished_at=datetime(2026, 9, 17, 10, 1, tzinfo=UTC),
        started_monotonic_ns=1_000,
        finished_monotonic_ns=2_000,
        return_code=0,
        timed_out=False,
        stdout="benchmark output\n",
        stderr="benchmark diagnostics\n",
    )

    record_process_execution(
        campaign=campaign,
        run=run,
        execution=execution,
    )

    assert run.stdout_path.read_text(encoding="utf-8") == "benchmark output\n"
    assert run.stderr_path.read_text(encoding="utf-8") == "benchmark diagnostics\n"

    with sqlite3.connect(campaign.database_path) as connection:
        row = connection.execute(
            """
            SELECT status,
                   started_at,
                   finished_at,
                   started_monotonic_ns,
                   finished_monotonic_ns,
                   return_code,
                   timed_out
            FROM runs
            WHERE run_id = ?
            """,
            (run.run_id,),
        ).fetchone()

    assert row == (
        "completed",
        "2026-09-17T10:00:00+00:00",
        "2026-09-17T10:01:00+00:00",
        1_000,
        2_000,
        0,
        0,
    )


@pytest.mark.parametrize(
    ("return_code", "timed_out", "expected_status"),
    [
        (1, False, "failed"),
        (None, True, "timed_out"),
    ],
)
def test_record_unsuccessful_process_execution(
    tmp_path: Path,
    return_code: int | None,
    timed_out: bool,
    expected_status: str,
) -> None:
    campaign = make_campaign(tmp_path)
    run = create_run_workspace(
        campaign=campaign,
        model="test-model",
        workload=make_workload(),
        repetition_index=0,
        configuration={},
        prompts=[],
    )

    execution = ProcessExecution(
        command=("benchmark",),
        working_directory=tmp_path,
        started_at=datetime(2026, 9, 17, tzinfo=UTC),
        finished_at=datetime(2026, 9, 17, 0, 1, tzinfo=UTC),
        started_monotonic_ns=100,
        finished_monotonic_ns=200,
        return_code=return_code,
        timed_out=timed_out,
        stdout="",
        stderr="failure",
    )

    record_process_execution(
        campaign=campaign,
        run=run,
        execution=execution,
    )

    with sqlite3.connect(campaign.database_path) as connection:
        status = connection.execute(
            "SELECT status FROM runs WHERE run_id = ?",
            (run.run_id,),
        ).fetchone()[0]

    assert status == expected_status


def test_existing_run_directory_is_not_overwritten(
    tmp_path: Path,
) -> None:
    campaign = make_campaign(tmp_path)

    arguments = {
        "campaign": campaign,
        "model": "test-model",
        "workload": make_workload(),
        "repetition_index": 0,
        "configuration": {},
        "prompts": [],
        "run_id": "11111111-2222-3333-4444-555555555555",
    }

    first = create_run_workspace(**arguments)
    marker = first.root / "preserve.txt"
    marker.write_text("preserve", encoding="utf-8")

    with pytest.raises(RunRepositoryError):
        create_run_workspace(**arguments)

    assert marker.read_text(encoding="utf-8") == "preserve"

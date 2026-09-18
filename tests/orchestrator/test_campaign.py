"""Tests for complete campaign orchestration."""

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

from tt_blackhole_benchmark.executor.models import ProcessExecution
from tt_blackhole_benchmark.models import (
    BenchmarkConfiguration,
    RuntimeConfiguration,
    TelemetryConfiguration,
    Workload,
)
from tt_blackhole_benchmark.orchestrator.campaign import (
    orchestrate_campaign,
)
from tt_blackhole_benchmark.persistence.campaign import (
    create_campaign_workspace,
)


def make_configuration(tmp_path: Path) -> BenchmarkConfiguration:
    return BenchmarkConfiguration(
        name="campaign-test",
        warmup_runs=1,
        repetitions=2,
        runtime=RuntimeConfiguration(
            model="test-model",
            max_sequence_length=1024,
            cache_path=tmp_path / "cache",
        ),
        telemetry=TelemetryConfiguration(
            enabled=False,
            interval_ms=500,
            idle_before_seconds=0,
            idle_after_seconds=0,
        ),
        workloads=(
            Workload(
                input_tokens=64,
                output_tokens=8,
                batch_size=1,
                request_count=1,
            ),
            Workload(
                input_tokens=128,
                output_tokens=16,
                batch_size=8,
                request_count=8,
            ),
        ),
    )


def test_orchestrate_campaign_expands_workloads_and_repetitions(
    tmp_path: Path,
) -> None:
    configuration = make_configuration(tmp_path)
    campaign = create_campaign_workspace(
        results_directory=tmp_path,
        name=configuration.name,
        configuration=configuration.model_dump(mode="json"),
        environment={"hostname": "test-host"},
        created_at=datetime(2026, 9, 18, tzinfo=UTC),
        campaign_id="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
    )

    executor_workloads: list[Workload] = []
    execution_count = 0

    def executor_factory(workload: Workload):
        executor_workloads.append(workload)

        def execute(prompt_file: Path) -> ProcessExecution:
            nonlocal execution_count

            current_index = execution_count
            execution_count += 1

            prompt_file.write_text(
                json.dumps([{"prompt": f"prompt-{current_index}"}]) + "\n",
                encoding="utf-8",
            )

            started_at = datetime(
                2026,
                9,
                18,
                tzinfo=UTC,
            ) + timedelta(minutes=current_index)

            return ProcessExecution(
                command=("fake-benchmark", str(current_index)),
                working_directory=tmp_path,
                started_at=started_at,
                finished_at=started_at + timedelta(seconds=1),
                started_monotonic_ns=current_index * 1_000,
                finished_monotonic_ns=current_index * 1_000 + 500,
                return_code=0,
                timed_out=False,
                stdout=f"output-{current_index}\n",
                stderr="",
            )

        return execute

    result = orchestrate_campaign(
        campaign=campaign,
        configuration=configuration,
        executor_factory=executor_factory,
        telemetry_backend=None,
    )

    assert len(result.runs) == 6
    assert executor_workloads == list(configuration.workloads)
    assert execution_count == 6

    with sqlite3.connect(campaign.database_path) as connection:
        rows = connection.execute(
            """
            SELECT
                input_tokens_requested,
                output_tokens_requested,
                batch_size,
                run_phase,
                repetition_index,
                status
            FROM runs
            ORDER BY rowid
            """
        ).fetchall()

    assert rows == [
        (64, 8, 1, "warmup", 0, "completed"),
        (64, 8, 1, "measurement", 0, "completed"),
        (64, 8, 1, "measurement", 1, "completed"),
        (128, 16, 8, "warmup", 0, "completed"),
        (128, 16, 8, "measurement", 0, "completed"),
        (128, 16, 8, "measurement", 1, "completed"),
    ]


def test_campaign_run_configuration_records_phase(
    tmp_path: Path,
) -> None:
    configuration = make_configuration(tmp_path).model_copy(
        update={
            "workloads": (
                Workload(
                    input_tokens=64,
                    output_tokens=8,
                    batch_size=1,
                    request_count=1,
                ),
            ),
            "repetitions": 1,
        }
    )

    campaign = create_campaign_workspace(
        results_directory=tmp_path,
        name=configuration.name,
        configuration=configuration.model_dump(mode="json"),
        environment={},
    )

    def executor_factory(workload: Workload):
        def execute(prompt_file: Path) -> ProcessExecution:
            prompt_file.write_text("[]\n", encoding="utf-8")
            timestamp = datetime.now(UTC)

            return ProcessExecution(
                command=("fake",),
                working_directory=tmp_path,
                started_at=timestamp,
                finished_at=timestamp,
                started_monotonic_ns=1,
                finished_monotonic_ns=1,
                return_code=0,
                timed_out=False,
                stdout="",
                stderr="",
            )

        return execute

    result = orchestrate_campaign(
        campaign=campaign,
        configuration=configuration,
        executor_factory=executor_factory,
        telemetry_backend=None,
    )

    saved_configurations = [
        json.loads(run.workspace.configuration_path.read_text(encoding="utf-8"))
        for run in result.runs
    ]

    assert [item["run_phase"] for item in saved_configurations] == [
        "warmup",
        "measurement",
    ]

"""Hardware end-to-end test for campaign orchestration."""

import os
import sqlite3
from pathlib import Path

import pytest

from tt_blackhole_benchmark.models import (
    BenchmarkConfiguration,
    RuntimeConfiguration,
    TelemetryConfiguration,
    Workload,
)
from tt_blackhole_benchmark.orchestrator.campaign import (
    orchestrate_campaign,
)
from tt_blackhole_benchmark.orchestrator.tt_metal import (
    create_tt_metal_executor,
)
from tt_blackhole_benchmark.persistence.campaign import (
    create_campaign_workspace,
)
from tt_blackhole_benchmark.telemetry.persistent_backend import (
    PersistentTtSmiBackend,
)


@pytest.mark.hardware
@pytest.mark.slow
def test_campaign_hardware_smoke(
    tmp_path: Path,
) -> None:
    tt_metal_root_value = os.getenv("TT_METAL_ROOT")
    cache_path_value = os.getenv("TT_METAL_CACHE")

    if not tt_metal_root_value or not cache_path_value:
        pytest.skip("requires TT_METAL_ROOT and TT_METAL_CACHE")

    tt_metal_root = Path(tt_metal_root_value)
    cache_path = Path(cache_path_value)

    timeout_seconds = float(
        os.getenv(
            "TT_BENCHMARK_TIMEOUT_SECONDS",
            "1800",
        )
    )

    workload = Workload(
        input_tokens=64,
        output_tokens=8,
        batch_size=1,
        request_count=1,
    )

    configuration = BenchmarkConfiguration(
        name="hardware-smoke",
        warmup_runs=0,
        repetitions=1,
        runtime=RuntimeConfiguration(
            model="Qwen/Qwen2.5-0.5B-Instruct",
            max_sequence_length=1024,
            trace_enabled=True,
            cache_path=cache_path,
        ),
        telemetry=TelemetryConfiguration(
            enabled=True,
            interval_ms=500,
            idle_before_seconds=1.0,
            idle_after_seconds=1.0,
        ),
        workloads=(workload,),
    )

    campaign = create_campaign_workspace(
        results_directory=tmp_path,
        name=configuration.name,
        configuration=configuration.model_dump(mode="json"),
        environment={
            "test_type": "hardware-smoke",
            "tt_metal_root": str(tt_metal_root),
        },
    )

    def executor_factory(selected_workload: Workload):
        return create_tt_metal_executor(
            tt_metal_root=tt_metal_root,
            runtime=configuration.runtime,
            workload=selected_workload,
            timeout_seconds=timeout_seconds,
        )

    with PersistentTtSmiBackend(
        startup_timeout_seconds=30.0,
        request_timeout_seconds=5.0,
    ) as telemetry_backend:
        result = orchestrate_campaign(
            campaign=campaign,
            configuration=configuration,
            executor_factory=executor_factory,
            telemetry_backend=telemetry_backend,
        )

    assert len(result.runs) == 1

    completed_run = result.runs[0]
    run_id = completed_run.workspace.run_id

    assert completed_run.process.timed_out is False
    assert completed_run.process.return_code == 0
    assert completed_run.workspace.prompts_path.is_file()
    assert completed_run.workspace.stdout_path.is_file()
    assert completed_run.workspace.stderr_path.is_file()

    with sqlite3.connect(campaign.database_path) as connection:
        run_row = connection.execute(
            """
            SELECT
                run_phase,
                input_tokens_requested,
                output_tokens_requested,
                batch_size,
                request_count,
                status,
                return_code,
                timed_out
            FROM runs
            WHERE run_id = ?
            """,
            (run_id,),
        ).fetchone()

        telemetry_count = connection.execute(
            """
            SELECT COUNT(*)
            FROM telemetry_samples
            WHERE run_id = ?
            """,
            (run_id,),
        ).fetchone()[0]

        telemetry_failure_count = connection.execute(
            """
            SELECT COUNT(*)
            FROM telemetry_failures
            WHERE run_id = ?
            """,
            (run_id,),
        ).fetchone()[0]

    assert run_row == (
        "measurement",
        64,
        8,
        1,
        1,
        "completed",
        0,
        0,
    )
    assert telemetry_count >= 2
    assert telemetry_failure_count == 0

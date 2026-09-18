"""End-to-end tests for single-run orchestration."""

import json
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tt_blackhole_benchmark.executor.models import ProcessExecution
from tt_blackhole_benchmark.models import Workload
from tt_blackhole_benchmark.orchestrator.run import (
    OrchestrationError,
    orchestrate_run,
)
from tt_blackhole_benchmark.persistence.campaign import (
    create_campaign_workspace,
)
from tt_blackhole_benchmark.telemetry.models import (
    TelemetryAcquisition,
    TelemetrySample,
    TelemetrySnapshot,
)


class FakeTelemetryBackend:
    """Produce valid telemetry with unique monotonic timestamps."""

    def __init__(self) -> None:
        self.call_count = 0
        self.acquired = threading.Event()

    def acquire(self) -> TelemetryAcquisition:
        self.call_count += 1
        captured_at = datetime.now(UTC)
        monotonic_ns = self.call_count * 1_000

        sample = TelemetrySample(
            captured_at=captured_at,
            monotonic_ns=monotonic_ns,
            device_index=0,
            bus_id="0000:01:00.0",
            board_type="p150a",
            power_w=35.0,
            voltage_v=0.74,
            current_a=48.0,
            asic_temperature_c=31.0,
            aiclk_mhz=800.0,
            fan_speed_percent=38.0,
            heartbeat=100 + self.call_count,
        )
        acquisition = TelemetryAcquisition(
            snapshot=TelemetrySnapshot(
                captured_at=captured_at,
                monotonic_ns=monotonic_ns,
                samples=(sample,),
            ),
            command=("fake-telemetry-worker",),
            exit_code=0,
        )

        self.acquired.set()
        return acquisition


def make_campaign(tmp_path: Path):
    return create_campaign_workspace(
        results_directory=tmp_path,
        name="orchestrator-test",
        configuration={"name": "orchestrator-test"},
        environment={"hostname": "test-host"},
        created_at=datetime(2026, 9, 18, tzinfo=UTC),
        campaign_id="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
    )


def make_workload() -> Workload:
    return Workload(
        input_tokens=64,
        output_tokens=8,
        batch_size=1,
        request_count=1,
    )


def test_orchestrate_run_persists_process_and_telemetry(
    tmp_path: Path,
) -> None:
    campaign = make_campaign(tmp_path)
    backend = FakeTelemetryBackend()

    def fake_executor(prompt_file: Path) -> ProcessExecution:
        assert backend.acquired.wait(timeout=1.0) is True

        prompt_file.write_text(
            json.dumps(
                [{"prompt": "benchmark prompt"}],
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        return ProcessExecution(
            command=("fake-benchmark",),
            working_directory=tmp_path,
            started_at=datetime(
                2026,
                9,
                18,
                10,
                0,
                tzinfo=UTC,
            ),
            finished_at=datetime(
                2026,
                9,
                18,
                10,
                1,
                tzinfo=UTC,
            ),
            started_monotonic_ns=10_000,
            finished_monotonic_ns=20_000,
            return_code=0,
            timed_out=False,
            stdout="benchmark output\n",
            stderr="",
        )

    result = orchestrate_run(
        campaign=campaign,
        model="test-model",
        workload=make_workload(),
        repetition_index=0,
        run_configuration={"trace_enabled": True},
        executor=fake_executor,
        telemetry_backend=backend,
        telemetry_interval_seconds=0.01,
        run_id="11111111-2222-3333-4444-555555555555",
    )

    assert result.process.return_code == 0
    assert result.workspace.stdout_path.read_text(encoding="utf-8") == "benchmark output\n"

    saved_prompts = json.loads(result.workspace.prompts_path.read_text(encoding="utf-8"))
    assert saved_prompts == [{"prompt": "benchmark prompt"}]

    with sqlite3.connect(campaign.database_path) as connection:
        run_row = connection.execute(
            """
            SELECT status, return_code, timed_out
            FROM runs
            WHERE run_id = ?
            """,
            (result.workspace.run_id,),
        ).fetchone()

        telemetry_rows = connection.execute(
            """
            SELECT
                run_id,
                device_index,
                power_w,
                source
            FROM telemetry_samples
            WHERE run_id = ?
            """,
            (result.workspace.run_id,),
        ).fetchall()

    assert run_row == ("completed", 0, 0)
    assert telemetry_rows
    assert all(
        row
        == (
            result.workspace.run_id,
            0,
            35.0,
            "tt-smi",
        )
        for row in telemetry_rows
    )


def test_orchestrate_run_without_telemetry(
    tmp_path: Path,
) -> None:
    campaign = make_campaign(tmp_path)

    def fake_executor(prompt_file: Path) -> ProcessExecution:
        prompt_file.write_text("[]\n", encoding="utf-8")

        return ProcessExecution(
            command=("fake-benchmark",),
            working_directory=tmp_path,
            started_at=datetime(2026, 9, 18, tzinfo=UTC),
            finished_at=datetime(
                2026,
                9,
                18,
                0,
                1,
                tzinfo=UTC,
            ),
            started_monotonic_ns=100,
            finished_monotonic_ns=200,
            return_code=0,
            timed_out=False,
            stdout="",
            stderr="",
        )

    result = orchestrate_run(
        campaign=campaign,
        model="test-model",
        workload=make_workload(),
        repetition_index=0,
        run_configuration={},
        executor=fake_executor,
        telemetry_backend=None,
        telemetry_interval_seconds=0.5,
    )

    with sqlite3.connect(campaign.database_path) as connection:
        sample_count = connection.execute(
            """
            SELECT COUNT(*)
            FROM telemetry_samples
            WHERE run_id = ?
            """,
            (result.workspace.run_id,),
        ).fetchone()[0]

    assert sample_count == 0


def test_orchestrate_run_records_executor_failure(
    tmp_path: Path,
) -> None:
    campaign = make_campaign(tmp_path)

    def failing_executor(
        prompt_file: Path,
    ) -> ProcessExecution:
        raise RuntimeError("model initialization failed")

    with pytest.raises(
        OrchestrationError,
        match="Benchmark executor failed",
    ) as error:
        orchestrate_run(
            campaign=campaign,
            model="test-model",
            workload=make_workload(),
            repetition_index=0,
            run_configuration={},
            executor=failing_executor,
            telemetry_backend=None,
            telemetry_interval_seconds=0.5,
            run_id="99999999-2222-3333-4444-555555555555",
        )

    assert isinstance(error.value.__cause__, RuntimeError)

    run_id = "99999999-2222-3333-4444-555555555555"
    error_path = campaign.runs_directory / run_id / "orchestrator_error.log"

    diagnostic = error_path.read_text(encoding="utf-8")

    assert "Benchmark executor failure" in diagnostic
    assert "RuntimeError: model initialization failed" in diagnostic

    with sqlite3.connect(campaign.database_path) as connection:
        row = connection.execute(
            """
            SELECT
                status,
                started_at,
                finished_at,
                return_code
            FROM runs
            WHERE run_id = ?
            """,
            (run_id,),
        ).fetchone()

    assert row == (
        "executor_failed",
        None,
        None,
        None,
    )


class DuplicateTelemetryBackend:
    """Produce the same raw sample twice."""

    def __init__(self) -> None:
        self.call_count = 0
        self.second_acquisition_reached = threading.Event()
        self.captured_at = datetime(
            2026,
            9,
            18,
            11,
            0,
            tzinfo=UTC,
        )

    def acquire(self) -> TelemetryAcquisition:
        self.call_count += 1

        if self.call_count >= 2:
            self.second_acquisition_reached.set()

        sample = TelemetrySample(
            captured_at=self.captured_at,
            monotonic_ns=50_000,
            device_index=0,
            bus_id="0000:01:00.0",
            board_type="p150a",
            power_w=36.0,
            voltage_v=0.74,
            current_a=49.0,
            asic_temperature_c=32.0,
            aiclk_mhz=800.0,
            fan_speed_percent=38.0,
            heartbeat=200,
        )

        return TelemetryAcquisition(
            snapshot=TelemetrySnapshot(
                captured_at=self.captured_at,
                monotonic_ns=50_000,
                samples=(sample,),
            ),
            command=("duplicate-telemetry-worker",),
            exit_code=0,
        )


def test_orchestrate_run_records_telemetry_failure_after_process(
    tmp_path: Path,
) -> None:
    campaign = make_campaign(tmp_path)
    backend = DuplicateTelemetryBackend()
    run_id = "88888888-2222-3333-4444-555555555555"

    def fake_executor(
        prompt_file: Path,
    ) -> ProcessExecution:
        prompt_file.write_text("[]\n", encoding="utf-8")

        assert backend.second_acquisition_reached.wait(timeout=1.0) is True

        return ProcessExecution(
            command=("fake-benchmark",),
            working_directory=tmp_path,
            started_at=datetime(
                2026,
                9,
                18,
                11,
                0,
                tzinfo=UTC,
            ),
            finished_at=datetime(
                2026,
                9,
                18,
                11,
                1,
                tzinfo=UTC,
            ),
            started_monotonic_ns=60_000,
            finished_monotonic_ns=70_000,
            return_code=0,
            timed_out=False,
            stdout="process completed\n",
            stderr="",
        )

    with pytest.raises(
        OrchestrationError,
        match="Telemetry collection failed",
    ) as error:
        orchestrate_run(
            campaign=campaign,
            model="test-model",
            workload=make_workload(),
            repetition_index=0,
            run_configuration={},
            executor=fake_executor,
            telemetry_backend=backend,
            telemetry_interval_seconds=0.01,
            run_id=run_id,
        )

    assert error.value.__cause__ is not None

    run_root = campaign.runs_directory / run_id

    assert (run_root / "stdout.log").read_text(encoding="utf-8") == "process completed\n"

    diagnostic = (run_root / "orchestrator_error.log").read_text(encoding="utf-8")

    assert "Telemetry collector failure" in diagnostic
    assert "Unable to persist telemetry" in diagnostic

    with sqlite3.connect(campaign.database_path) as connection:
        run_row = connection.execute(
            """
            SELECT
                status,
                started_at,
                finished_at,
                return_code,
                timed_out
            FROM runs
            WHERE run_id = ?
            """,
            (run_id,),
        ).fetchone()

        sample_count = connection.execute(
            """
            SELECT COUNT(*)
            FROM telemetry_samples
            WHERE run_id = ?
            """,
            (run_id,),
        ).fetchone()[0]

    assert run_row == (
        "telemetry_failed",
        "2026-09-18T11:00:00+00:00",
        "2026-09-18T11:01:00+00:00",
        0,
        0,
    )

    assert sample_count == 1

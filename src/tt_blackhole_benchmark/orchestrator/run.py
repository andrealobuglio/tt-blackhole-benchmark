"""Orchestration of one benchmark run."""

from __future__ import annotations

import time
import traceback
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tt_blackhole_benchmark.executor.models import ProcessExecution
from tt_blackhole_benchmark.models import Workload
from tt_blackhole_benchmark.persistence.campaign import CampaignWorkspace
from tt_blackhole_benchmark.persistence.run_repository import (
    RunWorkspace,
    create_run_workspace,
    record_orchestration_failure,
    record_process_execution,
)
from tt_blackhole_benchmark.persistence.telemetry_repository import (
    record_telemetry_acquisition,
    record_telemetry_failure,
)
from tt_blackhole_benchmark.telemetry.collector import (
    TelemetryBackend,
    TelemetryCollector,
)

BenchmarkExecutor = Callable[[Path], ProcessExecution]


class OrchestrationError(RuntimeError):
    """Raised when execution or telemetry orchestration fails."""


@dataclass(frozen=True, slots=True)
class OrchestratedRun:
    """Artifacts and raw process observation produced by one run."""

    workspace: RunWorkspace
    process: ProcessExecution


def _format_exception(
    heading: str,
    error: Exception,
) -> str:
    formatted = "".join(
        traceback.format_exception(
            type(error),
            error,
            error.__traceback__,
        )
    )
    return f"{heading}\n\n{formatted}"


def orchestrate_run(
    *,
    campaign: CampaignWorkspace,
    model: str,
    workload: Workload,
    repetition_index: int,
    run_phase: str = "measurement",
    run_configuration: Mapping[str, Any],
    executor: BenchmarkExecutor,
    telemetry_backend: TelemetryBackend | None,
    telemetry_interval_seconds: float,
    idle_before_seconds: float = 0.0,
    idle_after_seconds: float = 0.0,
    collector_stop_timeout_seconds: float = 5.0,
    run_id: str | None = None,
) -> OrchestratedRun:
    """Execute and persist one benchmark run with optional telemetry."""

    if telemetry_interval_seconds <= 0:
        raise ValueError("telemetry_interval_seconds must be positive")

    if idle_before_seconds < 0:
        raise ValueError("idle_before_seconds must be non-negative")

    if idle_after_seconds < 0:
        raise ValueError("idle_after_seconds must be non-negative")

    if collector_stop_timeout_seconds <= 0:
        raise ValueError("collector_stop_timeout_seconds must be positive")

    run = create_run_workspace(
        campaign=campaign,
        model=model,
        workload=workload,
        repetition_index=repetition_index,
        run_phase=run_phase,
        configuration=run_configuration,
        prompts=[],
        run_id=run_id,
    )

    collector: TelemetryCollector | None = None

    if telemetry_backend is not None:
        collector = TelemetryCollector(
            telemetry_backend,
            interval_seconds=telemetry_interval_seconds,
            acquisition_sink=lambda acquisition: record_telemetry_acquisition(
                campaign=campaign,
                run=run,
                acquisition=acquisition,
            ),
            failure_sink=lambda failure: record_telemetry_failure(
                campaign=campaign,
                run=run,
                failure=failure,
            ),
        )
        collector.start()

    execution: ProcessExecution | None = None
    executor_error: Exception | None = None
    collector_error: Exception | None = None

    try:
        if idle_before_seconds > 0:
            time.sleep(idle_before_seconds)

        execution = executor(run.prompts_path)

        if idle_after_seconds > 0:
            time.sleep(idle_after_seconds)
    except Exception as error:
        executor_error = error
    finally:
        if collector is not None:
            try:
                collector.stop(timeout_seconds=collector_stop_timeout_seconds)
            except Exception as error:
                collector_error = error

    if execution is not None:
        record_process_execution(
            campaign=campaign,
            run=run,
            execution=execution,
        )

    if executor_error is not None:
        diagnostic = _format_exception(
            "Benchmark executor failure",
            executor_error,
        )

        if collector_error is not None:
            diagnostic += "\n\n"
            diagnostic += _format_exception(
                "Telemetry collector failure during cleanup",
                collector_error,
            )

        record_orchestration_failure(
            campaign=campaign,
            run=run,
            status="executor_failed",
            diagnostic=diagnostic,
        )

        raise OrchestrationError(
            f"Benchmark executor failed for run {run.run_id}"
        ) from executor_error

    if collector_error is not None:
        record_orchestration_failure(
            campaign=campaign,
            run=run,
            status="telemetry_failed",
            diagnostic=_format_exception(
                "Telemetry collector failure",
                collector_error,
            ),
        )

        raise OrchestrationError(
            f"Telemetry collection failed for run {run.run_id}"
        ) from collector_error

    if execution is None:
        raise RuntimeError("Benchmark executor did not produce a process observation")

    return OrchestratedRun(
        workspace=run,
        process=execution,
    )

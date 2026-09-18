"""Orchestration of all runs in one benchmark campaign."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from tt_blackhole_benchmark.models import (
    BenchmarkConfiguration,
    Workload,
)
from tt_blackhole_benchmark.orchestrator.run import (
    BenchmarkExecutor,
    OrchestratedRun,
    orchestrate_run,
)
from tt_blackhole_benchmark.persistence.campaign import (
    CampaignWorkspace,
)
from tt_blackhole_benchmark.telemetry.collector import (
    TelemetryBackend,
)

ExecutorFactory = Callable[[Workload], BenchmarkExecutor]


@dataclass(frozen=True, slots=True)
class CampaignExecution:
    """Completed run observations belonging to one campaign."""

    campaign: CampaignWorkspace
    runs: tuple[OrchestratedRun, ...]


def orchestrate_campaign(
    *,
    campaign: CampaignWorkspace,
    configuration: BenchmarkConfiguration,
    executor_factory: ExecutorFactory,
    telemetry_backend: TelemetryBackend | None,
    collector_stop_timeout_seconds: float = 5.0,
) -> CampaignExecution:
    """Execute warm-up and measured runs for every workload."""

    completed_runs: list[OrchestratedRun] = []

    telemetry = configuration.telemetry
    selected_backend = telemetry_backend if telemetry.enabled else None

    for workload in configuration.workloads:
        executor = executor_factory(workload)

        phases = (
            ("warmup", configuration.warmup_runs),
            ("measurement", configuration.repetitions),
        )

        for run_phase, run_count in phases:
            for repetition_index in range(run_count):
                run_configuration = {
                    "run_phase": run_phase,
                    "repetition_index": repetition_index,
                    "runtime": configuration.runtime.model_dump(mode="json"),
                    "telemetry": telemetry.model_dump(mode="json"),
                    "workload": workload.model_dump(mode="json"),
                }

                completed = orchestrate_run(
                    campaign=campaign,
                    model=configuration.runtime.model,
                    workload=workload,
                    repetition_index=repetition_index,
                    run_phase=run_phase,
                    run_configuration=run_configuration,
                    executor=executor,
                    telemetry_backend=selected_backend,
                    telemetry_interval_seconds=(telemetry.interval_ms / 1000.0),
                    idle_before_seconds=(telemetry.idle_before_seconds),
                    idle_after_seconds=(telemetry.idle_after_seconds),
                    collector_stop_timeout_seconds=(collector_stop_timeout_seconds),
                )
                completed_runs.append(completed)

    return CampaignExecution(
        campaign=campaign,
        runs=tuple(completed_runs),
    )

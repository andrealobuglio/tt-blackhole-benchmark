"""TT-Metal adapter for generic benchmark orchestration."""

from __future__ import annotations

from pathlib import Path

from tt_blackhole_benchmark.executor.models import ProcessExecution
from tt_blackhole_benchmark.executor.tt_metal_executor import (
    execute_tt_metal_workload,
)
from tt_blackhole_benchmark.models import (
    DispatchMode,
    RuntimeConfiguration,
    Workload,
)
from tt_blackhole_benchmark.orchestrator.run import BenchmarkExecutor


def create_tt_metal_executor(
    *,
    tt_metal_root: str | Path,
    runtime: RuntimeConfiguration,
    workload: Workload,
    timeout_seconds: float,
) -> BenchmarkExecutor:
    """Create a single-workload executor backed by TT-Metal."""

    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")

    if runtime.cache_path is None:
        raise ValueError("runtime.cache_path is required for TT-Metal execution")

    if runtime.dispatch_mode is not DispatchMode.FAST:
        raise ValueError("Only fast dispatch is supported by the benchmark executor")

    root = Path(tt_metal_root)
    cache_path = runtime.cache_path

    def execute(prompt_file: Path) -> ProcessExecution:
        execution = execute_tt_metal_workload(
            tt_metal_root=root,
            prompt_file=prompt_file,
            model=runtime.model,
            workload=workload,
            max_sequence_length=runtime.max_sequence_length,
            cache_path=cache_path,
            trace_enabled=runtime.trace_enabled,
            timeout_seconds=timeout_seconds,
        )

        return execution.process

    return execute

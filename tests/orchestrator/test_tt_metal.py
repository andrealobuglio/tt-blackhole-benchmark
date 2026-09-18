"""Tests for the TT-Metal orchestration adapter."""

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tt_blackhole_benchmark.executor.models import ProcessExecution
from tt_blackhole_benchmark.models import (
    DispatchMode,
    RuntimeConfiguration,
    Workload,
)
from tt_blackhole_benchmark.orchestrator import tt_metal
from tt_blackhole_benchmark.orchestrator.tt_metal import (
    create_tt_metal_executor,
)


def make_workload() -> Workload:
    return Workload(
        input_tokens=64,
        output_tokens=8,
        batch_size=1,
        request_count=1,
    )


def make_runtime(
    tmp_path: Path,
    *,
    dispatch_mode: DispatchMode = DispatchMode.FAST,
    cache_path: Path | None = None,
) -> RuntimeConfiguration:
    return RuntimeConfiguration(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        max_sequence_length=1024,
        dispatch_mode=dispatch_mode,
        trace_enabled=True,
        cache_path=cache_path or tmp_path / "jit-cache",
    )


def make_process(tmp_path: Path) -> ProcessExecution:
    return ProcessExecution(
        command=("fake-tt-metal",),
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
        started_monotonic_ns=1_000,
        finished_monotonic_ns=2_000,
        return_code=0,
        timed_out=False,
        stdout="output\n",
        stderr="",
    )


def test_tt_metal_adapter_forwards_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, Any] = {}
    expected_process = make_process(tmp_path)

    def fake_execute_tt_metal_workload(
        **arguments: Any,
    ) -> SimpleNamespace:
        observed.update(arguments)
        return SimpleNamespace(process=expected_process)

    monkeypatch.setattr(
        tt_metal,
        "execute_tt_metal_workload",
        fake_execute_tt_metal_workload,
    )

    workload = make_workload()
    runtime = make_runtime(tmp_path)
    tt_metal_root = tmp_path / "tt-metal"
    prompt_file = tmp_path / "run" / "prompts.json"

    executor = create_tt_metal_executor(
        tt_metal_root=tt_metal_root,
        runtime=runtime,
        workload=workload,
        timeout_seconds=1800.0,
    )

    process = executor(prompt_file)

    assert process is expected_process
    assert observed == {
        "tt_metal_root": tt_metal_root,
        "prompt_file": prompt_file,
        "model": runtime.model,
        "workload": workload,
        "max_sequence_length": 1024,
        "cache_path": runtime.cache_path,
        "trace_enabled": True,
        "timeout_seconds": 1800.0,
    }


def test_tt_metal_adapter_requires_cache_path(
    tmp_path: Path,
) -> None:
    runtime = RuntimeConfiguration(
        model="test-model",
        max_sequence_length=1024,
        dispatch_mode=DispatchMode.FAST,
        cache_path=None,
    )

    with pytest.raises(
        ValueError,
        match="cache_path is required",
    ):
        create_tt_metal_executor(
            tt_metal_root=tmp_path,
            runtime=runtime,
            workload=make_workload(),
            timeout_seconds=1800.0,
        )


def test_tt_metal_adapter_rejects_slow_dispatch(
    tmp_path: Path,
) -> None:
    runtime = make_runtime(
        tmp_path,
        dispatch_mode=DispatchMode.SLOW,
    )

    with pytest.raises(
        ValueError,
        match="Only fast dispatch",
    ):
        create_tt_metal_executor(
            tt_metal_root=tmp_path,
            runtime=runtime,
            workload=make_workload(),
            timeout_seconds=1800.0,
        )


@pytest.mark.parametrize("timeout_seconds", [0.0, -1.0])
def test_tt_metal_adapter_rejects_invalid_timeout(
    tmp_path: Path,
    timeout_seconds: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="timeout_seconds must be positive",
    ):
        create_tt_metal_executor(
            tt_metal_root=tmp_path,
            runtime=make_runtime(tmp_path),
            workload=make_workload(),
            timeout_seconds=timeout_seconds,
        )

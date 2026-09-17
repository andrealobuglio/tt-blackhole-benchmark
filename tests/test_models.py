"""Unit tests for benchmark domain models."""

import pytest
from pydantic import ValidationError

from tt_blackhole_benchmark.models import (
    BenchmarkConfiguration,
    DispatchMode,
    RuntimeConfiguration,
    TelemetryBackendType,
    TelemetryConfiguration,
    Workload,
)


def test_workload_accepts_valid_values() -> None:
    workload = Workload(
        input_tokens=128,
        output_tokens=64,
        batch_size=8,
        request_count=32,
    )

    assert workload.input_tokens == 128
    assert workload.output_tokens == 64
    assert workload.batch_size == 8
    assert workload.request_count == 32


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("input_tokens", 0),
        ("output_tokens", 0),
        ("batch_size", 0),
        ("request_count", 0),
    ],
)
def test_workload_rejects_non_positive_values(
    field: str,
    value: int,
) -> None:
    values = {
        "input_tokens": 128,
        "output_tokens": 64,
        "batch_size": 8,
        "request_count": 32,
        field: value,
    }

    with pytest.raises(ValidationError):
        Workload(**values)


def test_workload_rejects_too_few_requests() -> None:
    with pytest.raises(
        ValidationError,
        match="request_count must be greater than or equal to batch_size",
    ):
        Workload(
            input_tokens=128,
            output_tokens=64,
            batch_size=16,
            request_count=8,
        )


def test_default_telemetry_configuration_matches_methodology() -> None:
    telemetry = TelemetryConfiguration()

    assert telemetry.enabled is True
    assert telemetry.backend is TelemetryBackendType.PERSISTENT_TT_SMI
    assert telemetry.interval_ms == 500
    assert telemetry.idle_before_seconds == 2.0
    assert telemetry.idle_after_seconds == 2.0


def test_runtime_defaults_to_fast_dispatch_and_trace() -> None:
    runtime = RuntimeConfiguration(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        max_sequence_length=1024,
    )

    assert runtime.dispatch_mode is DispatchMode.FAST
    assert runtime.trace_enabled is True


def test_benchmark_rejects_workload_exceeding_context_length() -> None:
    runtime = RuntimeConfiguration(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        max_sequence_length=128,
    )
    workload = Workload(
        input_tokens=96,
        output_tokens=64,
        batch_size=1,
        request_count=1,
    )

    with pytest.raises(
        ValidationError,
        match=r"input_tokens \+ output_tokens must not exceed",
    ):
        BenchmarkConfiguration(
            name="invalid-campaign",
            runtime=runtime,
            workloads=(workload,),
        )


def test_models_are_immutable() -> None:
    workload = Workload(
        input_tokens=128,
        output_tokens=64,
        batch_size=1,
        request_count=1,
    )

    with pytest.raises(ValidationError):
        workload.batch_size = 2


def test_telemetry_rejects_unknown_backend() -> None:
    with pytest.raises(ValidationError):
        TelemetryConfiguration.model_validate({"backend": "unknown-backend"})

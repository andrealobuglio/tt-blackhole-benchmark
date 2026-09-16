"""Core domain models for benchmark configuration."""

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DispatchMode(StrEnum):
    """Available TT-Metal dispatch modes."""

    FAST = "fast"
    SLOW = "slow"


class Workload(BaseModel):
    """A single benchmark point in the workload space."""

    model_config = ConfigDict(frozen=True)

    input_tokens: int = Field(gt=0)
    output_tokens: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    request_count: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_request_count(self) -> "Workload":
        if self.request_count < self.batch_size:
            raise ValueError("request_count must be greater than or equal to batch_size")
        return self


class RuntimeConfiguration(BaseModel):
    """Configuration of the model runtime."""

    model_config = ConfigDict(frozen=True)

    model: str = Field(min_length=1)
    max_sequence_length: int = Field(gt=0)
    dispatch_mode: DispatchMode = DispatchMode.FAST
    trace_enabled: bool = True
    cache_path: Path | None = None


class TelemetryConfiguration(BaseModel):
    """Configuration of telemetry sampling."""

    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    backend: str = "tt-smi"
    interval_ms: int = Field(default=500, ge=100)
    idle_before_seconds: float = Field(default=2.0, ge=0)
    idle_after_seconds: float = Field(default=2.0, ge=0)


class BenchmarkConfiguration(BaseModel):
    """Complete configuration of a benchmark campaign."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1)
    seed: int = 42
    warmup_runs: int = Field(default=1, ge=0)
    repetitions: int = Field(default=3, gt=0)
    runtime: RuntimeConfiguration
    telemetry: TelemetryConfiguration = Field(default_factory=TelemetryConfiguration)
    workloads: tuple[Workload, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_sequence_lengths(self) -> "BenchmarkConfiguration":
        for workload in self.workloads:
            total_tokens = workload.input_tokens + workload.output_tokens
            if total_tokens > self.runtime.max_sequence_length:
                raise ValueError(
                    "input_tokens + output_tokens must not exceed "
                    f"max_sequence_length: {total_tokens} > "
                    f"{self.runtime.max_sequence_length}"
                )
        return self

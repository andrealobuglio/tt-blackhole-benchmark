"""Domain models for accelerator telemetry."""

from datetime import UTC, datetime

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


class TelemetrySample(BaseModel):
    """A telemetry sample for one accelerator device."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    captured_at: datetime
    monotonic_ns: int = Field(ge=0)

    device_index: int = Field(ge=0)
    bus_id: str = Field(min_length=1)
    board_type: str = Field(min_length=1)

    power_w: float = Field(ge=0)
    voltage_v: float = Field(ge=0)
    current_a: float = Field(ge=0)
    asic_temperature_c: float
    aiclk_mhz: float = Field(ge=0)
    fan_speed_percent: float = Field(ge=0, le=100)
    heartbeat: int = Field(ge=0)

    source: str = "tt-smi"

    @field_validator("captured_at")
    @classmethod
    def validate_and_normalize_timestamp(
        cls,
        value: datetime,
    ) -> datetime:
        """Require a timezone-aware timestamp and normalize it to UTC."""

        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("captured_at must be timezone-aware")

        return value.astimezone(UTC)


class TelemetrySnapshot(BaseModel):
    """Telemetry samples produced by one backend acquisition."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    captured_at: datetime
    monotonic_ns: int = Field(ge=0)
    samples: tuple[TelemetrySample, ...] = Field(min_length=1)

    @field_validator("captured_at")
    @classmethod
    def validate_and_normalize_timestamp(
        cls,
        value: datetime,
    ) -> datetime:
        """Require a timezone-aware timestamp and normalize it to UTC."""

        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("captured_at must be timezone-aware")

        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_samples(self) -> "TelemetrySnapshot":
        """Ensure samples belong to this acquisition and devices are unique."""

        device_indices = [sample.device_index for sample in self.samples]

        if len(device_indices) != len(set(device_indices)):
            raise ValueError("Telemetry snapshot contains duplicate device indices")

        for sample in self.samples:
            if sample.captured_at != self.captured_at:
                raise ValueError("Sample captured_at must match snapshot captured_at")

            if sample.monotonic_ns != self.monotonic_ns:
                raise ValueError("Sample monotonic_ns must match snapshot monotonic_ns")

        return self

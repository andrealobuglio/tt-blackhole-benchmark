"""Raw observations produced by benchmark processes."""

from datetime import UTC, datetime
from pathlib import Path

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


class ProcessExecution(BaseModel):
    """Direct observations from one subprocess execution."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    command: tuple[str, ...] = Field(min_length=1)
    working_directory: Path

    started_at: datetime
    finished_at: datetime
    started_monotonic_ns: int = Field(ge=0)
    finished_monotonic_ns: int = Field(ge=0)

    return_code: int | None
    timed_out: bool

    stdout: str
    stderr: str

    @field_validator("started_at", "finished_at")
    @classmethod
    def normalize_timestamp(
        cls,
        value: datetime,
    ) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware")

        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_observation_order(self) -> "ProcessExecution":
        if self.finished_at < self.started_at:
            raise ValueError("finished_at must not precede started_at")

        if self.finished_monotonic_ns < self.started_monotonic_ns:
            raise ValueError("finished_monotonic_ns must not precede started_monotonic_ns")

        if self.timed_out and self.return_code is not None:
            raise ValueError("a timed-out process must not have a return code")

        return self

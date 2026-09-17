"""Unit tests for the TT-Metal execution adapter."""

import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

from tt_blackhole_benchmark.executor.models import (
    ProcessExecution,
)
from tt_blackhole_benchmark.executor.tt_metal_executor import (
    execute_tt_metal_workload,
)
from tt_blackhole_benchmark.models import Workload


def fake_encoder(prompt: str) -> tuple[int, ...]:
    repetitions = prompt.count(" benchmark")
    return tuple(range(29 + repetitions))


class FakeProcessRunner:
    def __init__(self) -> None:
        self.command: tuple[str, ...] | None = None
        self.environment: Mapping[str, str | None] | None = None

    def __call__(
        self,
        command: Sequence[str],
        *,
        working_directory: str | Path,
        timeout_seconds: float,
        environment: Mapping[str, str | None] | None = None,
    ) -> ProcessExecution:
        self.command = tuple(command)
        self.environment = environment
        timestamp = datetime.now(UTC)

        return ProcessExecution(
            command=tuple(command),
            working_directory=Path(working_directory),
            started_at=timestamp,
            finished_at=timestamp,
            started_monotonic_ns=1_000,
            finished_monotonic_ns=1_000,
            return_code=0,
            timed_out=False,
            stdout="Encoded prompt lengths:64\nPASSED\n",
            stderr="",
        )


def test_execute_tt_metal_workload_builds_raw_artifacts(
    tmp_path: Path,
) -> None:
    runner = FakeProcessRunner()
    workload = Workload(
        input_tokens=64,
        output_tokens=8,
        batch_size=1,
        request_count=1,
    )
    prompt_file = tmp_path / "run" / "prompts.json"

    execution = execute_tt_metal_workload(
        tt_metal_root=tmp_path,
        prompt_file=prompt_file,
        model="Qwen/Qwen2.5-0.5B-Instruct",
        workload=workload,
        max_sequence_length=1024,
        cache_path=tmp_path / "cache",
        trace_enabled=True,
        timeout_seconds=60.0,
        encoder=fake_encoder,
        process_runner=runner,
    )

    assert len(execution.prompt.token_ids) == 64
    assert execution.prompt_file == prompt_file
    assert execution.process.return_code == 0

    assert runner.command is not None
    assert "--batch_size=1" in runner.command
    assert "--max_generated_tokens=7" in runner.command

    assert runner.environment is not None
    assert runner.environment["TT_METAL_SLOW_DISPATCH_MODE"] is None

    prompt_content = json.loads(prompt_file.read_text(encoding="utf-8"))
    assert prompt_content == [{"prompt": execution.prompt.text}]

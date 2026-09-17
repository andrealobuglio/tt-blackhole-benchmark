"""Execution adapter for the TT-Metal text demo."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from tt_blackhole_benchmark.executor.models import (
    ProcessExecution,
)
from tt_blackhole_benchmark.executor.process import (
    execute_process,
)
from tt_blackhole_benchmark.executor.prompts import (
    GeneratedPrompt,
    PromptEncoder,
    create_hf_chat_encoder,
    generate_exact_length_prompt,
    write_prompt_file,
)
from tt_blackhole_benchmark.executor.tt_metal_command import (
    TtMetalInvocation,
    build_tt_metal_invocation,
)
from tt_blackhole_benchmark.models import Workload


class ProcessRunner(Protocol):
    """Process runner interface used by the adapter."""

    def __call__(
        self,
        command: Sequence[str],
        *,
        working_directory: str | Path,
        timeout_seconds: float,
        environment: Mapping[str, str | None] | None = None,
    ) -> ProcessExecution:
        """Execute one process."""
        ...


@dataclass(frozen=True)
class TtMetalExecution:
    """Raw artifacts and observations from one TT-Metal run."""

    workload: Workload
    prompt: GeneratedPrompt
    prompt_file: Path
    invocation: TtMetalInvocation
    process: ProcessExecution


def execute_tt_metal_workload(
    *,
    tt_metal_root: str | Path,
    prompt_file: str | Path,
    model: str,
    workload: Workload,
    max_sequence_length: int,
    cache_path: str | Path,
    trace_enabled: bool,
    timeout_seconds: float,
    encoder: PromptEncoder | None = None,
    process_runner: ProcessRunner = execute_process,
) -> TtMetalExecution:
    """Generate exact prompts and execute one TT-Metal workload."""

    selected_encoder = encoder or create_hf_chat_encoder(model)

    prompt = generate_exact_length_prompt(
        target_tokens=workload.input_tokens,
        encoder=selected_encoder,
    )

    written_prompt_file = write_prompt_file(
        prompt_file,
        prompt=prompt,
        request_count=workload.request_count,
    )

    invocation = build_tt_metal_invocation(
        tt_metal_root=tt_metal_root,
        prompt_file=written_prompt_file,
        model=model,
        workload=workload,
        max_sequence_length=max_sequence_length,
        cache_path=cache_path,
        trace_enabled=trace_enabled,
    )

    process = process_runner(
        invocation.command,
        working_directory=invocation.working_directory,
        timeout_seconds=timeout_seconds,
        environment=invocation.environment,
    )

    return TtMetalExecution(
        workload=workload,
        prompt=prompt,
        prompt_file=written_prompt_file,
        invocation=invocation,
        process=process,
    )

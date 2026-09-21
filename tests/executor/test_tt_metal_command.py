"""Unit tests for TT-Metal command construction."""

from pathlib import Path

import pytest

from tt_blackhole_benchmark.executor.tt_metal_command import (
    build_tt_metal_invocation,
)
from tt_blackhole_benchmark.models import Workload


def make_workload(
    *,
    output_tokens: int = 64,
    batch_size: int = 8,
    request_count: int = 32,
) -> Workload:
    return Workload(
        input_tokens=128,
        output_tokens=output_tokens,
        batch_size=batch_size,
        request_count=request_count,
    )


def test_build_tt_metal_invocation(
    tmp_path: Path,
) -> None:
    prompts = tmp_path / "prompts.json"
    cache = tmp_path / "cache"

    invocation = build_tt_metal_invocation(
        tt_metal_root=tmp_path,
        prompt_file=prompts,
        model="Qwen/Qwen2.5-0.5B-Instruct",
        workload=make_workload(),
        max_sequence_length=1024,
        cache_path=cache,
        trace_enabled=True,
        python_executable="/usr/bin/python3",
    )

    assert invocation.working_directory == tmp_path.resolve()
    assert "--batch_size=8" in invocation.command
    assert "--repeat_batches=4" in invocation.command
    assert "--max_generated_tokens=63" in invocation.command
    assert "--stop_at_eos=0" in invocation.command
    assert "--enable_trace" in invocation.command

    assert invocation.environment["TT_METAL_SLOW_DISPATCH_MODE"] is None
    assert invocation.environment["HF_MODEL"] == "Qwen/Qwen2.5-0.5B-Instruct"


def test_builds_disable_trace_option(
    tmp_path: Path,
) -> None:
    invocation = build_tt_metal_invocation(
        tt_metal_root=tmp_path,
        prompt_file=tmp_path / "prompts.json",
        model="Qwen/Qwen2.5-0.5B-Instruct",
        workload=make_workload(),
        max_sequence_length=1024,
        cache_path=tmp_path / "cache",
        trace_enabled=False,
    )

    assert "--disable_trace" in invocation.command
    assert "--enable_trace" not in invocation.command


def test_rejects_non_divisible_request_count(
    tmp_path: Path,
) -> None:
    workload = make_workload(
        batch_size=8,
        request_count=10,
    )

    with pytest.raises(
        ValueError,
        match="request_count must be divisible",
    ):
        build_tt_metal_invocation(
            tt_metal_root=tmp_path,
            prompt_file=tmp_path / "prompts.json",
            model="Qwen/Qwen2.5-0.5B-Instruct",
            workload=workload,
            max_sequence_length=1024,
            cache_path=tmp_path / "cache",
            trace_enabled=True,
        )


def test_rejects_fewer_than_two_output_tokens(
    tmp_path: Path,
) -> None:
    workload = make_workload(output_tokens=1)

    with pytest.raises(
        ValueError,
        match="requires at least 2 output tokens",
    ):
        build_tt_metal_invocation(
            tt_metal_root=tmp_path,
            prompt_file=tmp_path / "prompts.json",
            model="Qwen/Qwen2.5-0.5B-Instruct",
            workload=workload,
            max_sequence_length=1024,
            cache_path=tmp_path / "cache",
            trace_enabled=True,
        )


def test_preserves_virtualenv_python_symlink(
    tmp_path: Path,
) -> None:
    python_link = tmp_path / "venv" / "bin" / "python"
    python_link.parent.mkdir(parents=True)
    python_link.symlink_to("/usr/bin/python3")

    invocation = build_tt_metal_invocation(
        tt_metal_root=tmp_path,
        prompt_file=tmp_path / "prompts.json",
        model="Qwen/Qwen2.5-0.5B-Instruct",
        workload=make_workload(),
        max_sequence_length=1024,
        cache_path=tmp_path / "cache",
        trace_enabled=True,
        python_executable=python_link,
    )

    assert invocation.command[0] == str(python_link)


@pytest.mark.parametrize("batch_size", [1, 8, 32])
def test_invocation_selects_only_performance_case(
    tmp_path: Path,
    batch_size: int,
) -> None:
    workload = Workload(
        input_tokens=128,
        output_tokens=32,
        batch_size=batch_size,
        request_count=batch_size,
    )

    invocation = build_tt_metal_invocation(
        tt_metal_root=tmp_path / "tt-metal",
        prompt_file=tmp_path / "prompts.json",
        model="test-model",
        workload=workload,
        max_sequence_length=1024,
        cache_path=tmp_path / "cache",
        trace_enabled=True,
        python_executable="/usr/bin/python3",
    )

    selector_index = invocation.command.index("-k")

    assert invocation.command[selector_index + 1] == "performance and batch-1"
    assert f"--batch_size={batch_size}" in invocation.command

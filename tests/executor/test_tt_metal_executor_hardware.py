"""Hardware smoke test for the TT-Metal executor."""

import os
from pathlib import Path

import pytest

from tt_blackhole_benchmark.executor.tt_metal_executor import (
    execute_tt_metal_workload,
)
from tt_blackhole_benchmark.models import Workload


@pytest.mark.hardware
@pytest.mark.slow
def test_tt_metal_executor_smoke(
    tmp_path: Path,
) -> None:
    tt_metal_root = os.getenv("TT_METAL_ROOT")
    cache_path = os.getenv("TT_METAL_CACHE")

    if not tt_metal_root or not cache_path:
        pytest.skip("requires TT_METAL_ROOT and TT_METAL_CACHE")

    execution = execute_tt_metal_workload(
        tt_metal_root=tt_metal_root,
        prompt_file=tmp_path / "prompts.json",
        model="Qwen/Qwen2.5-0.5B-Instruct",
        workload=Workload(
            input_tokens=64,
            output_tokens=8,
            batch_size=1,
            request_count=1,
        ),
        max_sequence_length=1024,
        cache_path=cache_path,
        trace_enabled=True,
        timeout_seconds=1_800.0,
    )

    process = execution.process
    combined_output = process.stdout + "\n" + process.stderr

    assert process.timed_out is False
    assert process.return_code == 0
    assert "Encoded prompt lengths:64" in combined_output
    assert "PASSED" in combined_output

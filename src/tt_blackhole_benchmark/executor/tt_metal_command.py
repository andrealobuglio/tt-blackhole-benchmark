"""Construction of reproducible TT-Metal demo invocations."""

import sys
from dataclasses import dataclass
from pathlib import Path

from tt_blackhole_benchmark.models import Workload


@dataclass(frozen=True)
class TtMetalInvocation:
    """Command and environment for one TT-Metal execution."""

    command: tuple[str, ...]
    working_directory: Path
    environment: dict[str, str | None]


def build_tt_metal_invocation(
    *,
    tt_metal_root: str | Path,
    prompt_file: str | Path,
    model: str,
    workload: Workload,
    max_sequence_length: int,
    cache_path: str | Path,
    trace_enabled: bool,
    python_executable: str | Path = sys.executable,
) -> TtMetalInvocation:
    """Build a TT-Metal pytest invocation without executing it."""

    root = Path(tt_metal_root).resolve()
    prompts = Path(prompt_file).resolve()
    executable = Path(python_executable)
    cache = Path(cache_path).resolve()

    if workload.request_count % workload.batch_size != 0:
        raise ValueError("request_count must be divisible by batch_size")

    if workload.output_tokens < 2:
        raise ValueError("TT-Metal full mode requires at least 2 output tokens")

    repeat_batches = workload.request_count // workload.batch_size

    # The prefill produces the first output token. The decode loop
    # must therefore run output_tokens - 1 iterations.
    max_generated_tokens = workload.output_tokens - 1

    trace_option = "--enable_trace" if trace_enabled else "--disable_trace"

    command = (
        str(executable),
        "-m",
        "pytest",
        "-s",
        "-v",
        "models/tt_transformers/demo/simple_text_demo.py",
        # Select only the performance parameter set. A plain
        # "batch-1" expression also selects accuracy-batch-1,
        # causing two complete inference executions per run.
        "-k",
        "performance and batch-1",
        "-k",
        "performance and batch-1",
        f"--input_prompts={prompts}",
        "--instruct=1",
        f"--repeat_batches={repeat_batches}",
        f"--max_seq_len={max_sequence_length}",
        f"--batch_size={workload.batch_size}",
        f"--max_generated_tokens={max_generated_tokens}",
        "--data_parallel=1",
        "--stop_at_eos=0",
        "--mode=full",
        trace_option,
    )

    environment: dict[str, str | None] = {
        "HF_MODEL": model,
        "MESH_DEVICE": "P150",
        "PYTHONUNBUFFERED": "1",
        "TT_METAL_CACHE": str(cache),
        "TT_METAL_SLOW_DISPATCH_MODE": None,
    }

    return TtMetalInvocation(
        command=command,
        working_directory=root,
        environment=environment,
    )

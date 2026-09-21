"""Command-line interface for the benchmark application."""

import argparse
import os
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from tt_blackhole_benchmark.config import (
    ConfigurationError,
    load_configuration,
)
from tt_blackhole_benchmark.environment import collect_environment
from tt_blackhole_benchmark.models import (
    DispatchMode,
    TelemetryBackendType,
    Workload,
)
from tt_blackhole_benchmark.orchestrator.campaign import (
    orchestrate_campaign,
)
from tt_blackhole_benchmark.orchestrator.run import (
    BenchmarkExecutor,
)
from tt_blackhole_benchmark.orchestrator.tt_metal import (
    create_tt_metal_executor,
)
from tt_blackhole_benchmark.persistence.campaign import (
    create_campaign_workspace,
)
from tt_blackhole_benchmark.telemetry.persistent_backend import (
    PersistentTtSmiBackend,
)

DEFAULT_TIMEOUT_SECONDS = 1800.0


def positive_float(value: str) -> float:
    """Parse a strictly positive floating-point argument."""

    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"expected a number, received {value!r}") from error

    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")

    return parsed


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""

    parser = argparse.ArgumentParser(
        prog="tt-benchmark",
        description="Benchmark suite for Tenstorrent Blackhole accelerators.",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate a benchmark YAML configuration.",
    )
    validate_parser.add_argument(
        "configuration",
        type=Path,
        help="Path to the YAML configuration file.",
    )

    run_parser = subparsers.add_parser(
        "run",
        help="Execute a benchmark campaign.",
    )
    run_parser.add_argument(
        "configuration",
        type=Path,
        help="Path to the YAML configuration file.",
    )
    run_parser.add_argument(
        "--results",
        type=Path,
        default=Path("results"),
        help="Directory in which campaign results are stored.",
    )
    run_parser.add_argument(
        "--tt-metal-root",
        type=Path,
        required=True,
        help="Path to the TT-Metal repository.",
    )
    run_parser.add_argument(
        "--timeout-seconds",
        type=positive_float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=(
            f"Maximum duration of each TT-Metal execution (default: {DEFAULT_TIMEOUT_SECONDS:g})."
        ),
    )

    return parser


def validate_command(configuration_path: Path) -> int:
    """Validate and summarize a benchmark configuration."""

    try:
        configuration = load_configuration(configuration_path)
    except (ConfigurationError, ValidationError) as error:
        print(f"Configuration error: {error}")
        return 2

    print(f"Configuration valid: {configuration.name}")
    print(f"Model: {configuration.runtime.model}")
    print(f"Workloads: {len(configuration.workloads)}")
    print(f"Repetitions: {configuration.repetitions}")
    print(f"Telemetry: {'enabled' if configuration.telemetry.enabled else 'disabled'}")
    return 0


def run_command(
    *,
    configuration_path: Path,
    results_directory: Path,
    tt_metal_root: Path,
    timeout_seconds: float,
) -> int:
    """Execute one complete benchmark campaign."""

    try:
        configuration = load_configuration(configuration_path)
    except (ConfigurationError, ValidationError) as error:
        print(f"Configuration error: {error}")
        return 2

    if not tt_metal_root.is_dir():
        print(
            "Configuration error: TT-Metal root does not exist "
            f"or is not a directory: {tt_metal_root}"
        )
        return 2

    if configuration.runtime.cache_path is None:
        print("Configuration error: runtime.cache_path is required for TT-Metal execution")
        return 2

    if configuration.runtime.dispatch_mode is not DispatchMode.FAST:
        print("Configuration error: only fast dispatch is supported")
        return 2

    if "TT_METAL_SLOW_DISPATCH_MODE" in os.environ:
        print("Configuration error: TT_METAL_SLOW_DISPATCH_MODE must be unset for fast dispatch")
        return 2

    telemetry = configuration.telemetry

    if telemetry.enabled and telemetry.backend is not TelemetryBackendType.PERSISTENT_TT_SMI:
        print(
            "Configuration error: enabled telemetry currently "
            "requires the tt-smi-persistent backend"
        )
        return 2

    try:
        environment = collect_environment(
            tt_metal_root=tt_metal_root,
        )
        campaign = create_campaign_workspace(
            results_directory=results_directory,
            name=configuration.name,
            configuration=configuration.model_dump(mode="json"),
            environment=environment,
        )

        def executor_factory(
            workload: Workload,
        ) -> BenchmarkExecutor:
            return create_tt_metal_executor(
                tt_metal_root=tt_metal_root,
                runtime=configuration.runtime,
                workload=workload,
                timeout_seconds=timeout_seconds,
            )

        if telemetry.enabled:
            with PersistentTtSmiBackend() as telemetry_backend:
                execution = orchestrate_campaign(
                    campaign=campaign,
                    configuration=configuration,
                    executor_factory=executor_factory,
                    telemetry_backend=telemetry_backend,
                )
        else:
            execution = orchestrate_campaign(
                campaign=campaign,
                configuration=configuration,
                executor_factory=executor_factory,
                telemetry_backend=None,
            )
    except Exception as error:
        print(f"Benchmark failed: {error}")
        return 1

    print(f"Campaign completed: {execution.campaign.campaign_id}")
    print(f"Results: {execution.campaign.root}")
    print(f"Runs: {len(execution.runs)}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface."""

    parser = build_parser()
    arguments = parser.parse_args(argv)

    if arguments.command == "validate":
        return validate_command(arguments.configuration)

    if arguments.command == "run":
        return run_command(
            configuration_path=arguments.configuration,
            results_directory=arguments.results,
            tt_metal_root=arguments.tt_metal_root,
            timeout_seconds=arguments.timeout_seconds,
        )

    parser.error(f"Unknown command: {arguments.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

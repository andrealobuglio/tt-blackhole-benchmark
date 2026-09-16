"""Command-line interface for the benchmark application."""

import argparse
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from tt_blackhole_benchmark.config import (
    ConfigurationError,
    load_configuration,
)


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


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface."""

    parser = build_parser()
    arguments = parser.parse_args(argv)

    if arguments.command == "validate":
        return validate_command(arguments.configuration)

    parser.error(f"Unknown command: {arguments.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

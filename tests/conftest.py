"""Shared pytest configuration."""

from collections.abc import Sequence

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register project-specific pytest options."""

    parser.addoption(
        "--run-hardware",
        action="store_true",
        default=False,
        help="Run tests requiring Tenstorrent hardware.",
    )


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: Sequence[pytest.Item],
) -> None:
    """Skip hardware tests unless explicitly requested."""

    if config.getoption("--run-hardware"):
        return

    skip_hardware = pytest.mark.skip(
        reason="requires --run-hardware",
    )

    for item in items:
        if "hardware" in item.keywords:
            item.add_marker(skip_hardware)

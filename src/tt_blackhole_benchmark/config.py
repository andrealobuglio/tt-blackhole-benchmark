"""Loading and validation of benchmark configuration files."""

from pathlib import Path
from typing import Any

import yaml

from tt_blackhole_benchmark.models import BenchmarkConfiguration


class ConfigurationError(ValueError):
    """Raised when a configuration file cannot be loaded."""


def load_configuration(path: str | Path) -> BenchmarkConfiguration:
    """Load and validate a benchmark configuration from YAML."""

    config_path = Path(path)

    if not config_path.is_file():
        raise ConfigurationError(f"Configuration file does not exist: {config_path}")

    try:
        with config_path.open(encoding="utf-8") as config_file:
            raw_configuration: Any = yaml.safe_load(config_file)
    except (OSError, yaml.YAMLError) as error:
        raise ConfigurationError(
            f"Cannot read configuration file {config_path}: {error}"
        ) from error

    if not isinstance(raw_configuration, dict):
        raise ConfigurationError(f"Configuration root must be a mapping: {config_path}")

    return BenchmarkConfiguration.model_validate(raw_configuration)

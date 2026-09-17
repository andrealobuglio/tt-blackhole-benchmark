"""Unit tests for YAML configuration loading."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from tt_blackhole_benchmark.config import (
    ConfigurationError,
    load_configuration,
)
from tt_blackhole_benchmark.models import DispatchMode, TelemetryBackendType


def test_load_smoke_configuration() -> None:
    configuration = load_configuration("configs/smoke.yaml")

    assert configuration.name == "qwen-smoke-test"
    assert configuration.repetitions == 1
    assert configuration.runtime.dispatch_mode is DispatchMode.FAST
    assert configuration.runtime.trace_enabled is True
    assert len(configuration.workloads) == 1
    assert configuration.workloads[0].input_tokens == 128
    assert configuration.telemetry.backend is TelemetryBackendType.PERSISTENT_TT_SMI


def test_missing_configuration_file_is_rejected(
    tmp_path: Path,
) -> None:
    missing_path = tmp_path / "missing.yaml"

    with pytest.raises(
        ConfigurationError,
        match="Configuration file does not exist",
    ):
        load_configuration(missing_path)


def test_invalid_yaml_is_rejected(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(
        "name: [invalid",
        encoding="utf-8",
    )

    with pytest.raises(
        ConfigurationError,
        match="Cannot read configuration file",
    ):
        load_configuration(config_path)


def test_non_mapping_yaml_is_rejected(tmp_path: Path) -> None:
    config_path = tmp_path / "list.yaml"
    config_path.write_text(
        "- first\n- second\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ConfigurationError,
        match="Configuration root must be a mapping",
    ):
        load_configuration(config_path)


def test_semantically_invalid_configuration_is_rejected(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "invalid-workload.yaml"
    config_path.write_text(
        """
name: invalid-workload
runtime:
  model: Qwen/Qwen2.5-0.5B-Instruct
  max_sequence_length: 128
workloads:
  - input_tokens: 96
    output_tokens: 64
    batch_size: 1
    request_count: 1
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(
        ValidationError,
        match=r"input_tokens \+ output_tokens must not exceed",
    ):
        load_configuration(config_path)

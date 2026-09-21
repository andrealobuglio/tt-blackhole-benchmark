"""Unit tests for the command-line interface."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tt_blackhole_benchmark import cli
from tt_blackhole_benchmark.cli import (
    DEFAULT_TIMEOUT_SECONDS,
    build_parser,
    main,
)


def test_validate_command_accepts_smoke_configuration(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(["validate", "configs/smoke.yaml"])

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Configuration valid: qwen-smoke-test" in captured.out
    assert "Qwen/Qwen2.5-0.5B-Instruct" in captured.out
    assert "Workloads: 1" in captured.out


def test_validate_command_rejects_missing_file(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(["validate", str(tmp_path / "missing.yaml")])

    captured = capsys.readouterr()

    assert exit_code == 2
    assert "Configuration error:" in captured.out
    assert "does not exist" in captured.out


def test_run_command_parses_required_arguments() -> None:
    parser = build_parser()

    arguments = parser.parse_args(
        [
            "run",
            "configs/smoke.yaml",
            "--results",
            "benchmark-results",
            "--tt-metal-root",
            "/opt/tt-metal",
            "--timeout-seconds",
            "900",
        ]
    )

    assert arguments.command == "run"
    assert arguments.configuration == Path("configs/smoke.yaml")
    assert arguments.results == Path("benchmark-results")
    assert arguments.tt_metal_root == Path("/opt/tt-metal")
    assert arguments.timeout_seconds == 900.0


def test_run_command_uses_safe_defaults() -> None:
    parser = build_parser()

    arguments = parser.parse_args(
        [
            "run",
            "configs/smoke.yaml",
            "--tt-metal-root",
            "/opt/tt-metal",
        ]
    )

    assert arguments.results == Path("results")
    assert arguments.timeout_seconds == DEFAULT_TIMEOUT_SECONDS


@pytest.mark.parametrize(
    "timeout_seconds",
    ["0", "-1", "not-a-number"],
)
def test_run_command_rejects_invalid_timeout(
    timeout_seconds: str,
) -> None:
    parser = build_parser()

    with pytest.raises(SystemExit) as error:
        parser.parse_args(
            [
                "run",
                "configs/smoke.yaml",
                "--tt-metal-root",
                "/opt/tt-metal",
                "--timeout-seconds",
                timeout_seconds,
            ]
        )

    assert error.value.code == 2


def test_run_command_requires_tt_metal_root() -> None:
    parser = build_parser()

    with pytest.raises(SystemExit) as error:
        parser.parse_args(
            [
                "run",
                "configs/smoke.yaml",
            ]
        )

    assert error.value.code == 2


def test_run_command_creates_campaign_without_hardware(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    configuration_path = tmp_path / "campaign.yaml"
    results_directory = tmp_path / "results"
    tt_metal_root = tmp_path / "tt-metal"
    cache_path = tmp_path / "cache"

    tt_metal_root.mkdir()

    configuration_path.write_text(
        f"""
name: cli-test
warmup_runs: 0
repetitions: 1
runtime:
  model: test-model
  max_sequence_length: 128
  dispatch_mode: fast
  trace_enabled: true
  cache_path: {cache_path}
telemetry:
  enabled: false
workloads:
  - input_tokens: 64
    output_tokens: 8
    batch_size: 1
    request_count: 1
""".strip()
        + "\n",
        encoding="utf-8",
    )

    observed: dict[str, Any] = {}

    def fake_orchestrate_campaign(
        **arguments: Any,
    ) -> SimpleNamespace:
        observed.update(arguments)
        return SimpleNamespace(
            campaign=arguments["campaign"],
            runs=(object(),),
        )

    monkeypatch.delenv(
        "TT_METAL_SLOW_DISPATCH_MODE",
        raising=False,
    )
    monkeypatch.setattr(
        cli,
        "orchestrate_campaign",
        fake_orchestrate_campaign,
    )

    exit_code = main(
        [
            "run",
            str(configuration_path),
            "--results",
            str(results_directory),
            "--tt-metal-root",
            str(tt_metal_root),
            "--timeout-seconds",
            "900",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Campaign completed:" in captured.out
    assert "Runs: 1" in captured.out

    campaign = observed["campaign"]

    assert campaign.root.parent == results_directory.resolve()
    assert observed["telemetry_backend"] is None

    saved_environment = json.loads(campaign.environment_path.read_text(encoding="utf-8"))

    assert saved_environment["repositories"]["tt_metal"]["root"] == str(tt_metal_root.resolve())


def test_run_command_rejects_slow_dispatch_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    tt_metal_root = tmp_path / "tt-metal"
    tt_metal_root.mkdir()

    monkeypatch.setenv("TT_METAL_SLOW_DISPATCH_MODE", "1")

    exit_code = main(
        [
            "run",
            "configs/smoke.yaml",
            "--results",
            str(tmp_path / "results"),
            "--tt-metal-root",
            str(tt_metal_root),
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 2
    assert "TT_METAL_SLOW_DISPATCH_MODE must be unset" in captured.out
    assert not (tmp_path / "results").exists()

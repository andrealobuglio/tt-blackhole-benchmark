"""Unit tests for the command-line interface."""

from pathlib import Path

import pytest

from tt_blackhole_benchmark.cli import main


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

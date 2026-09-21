"""Tests for execution-environment metadata acquisition."""

import sys
from pathlib import Path

import pytest

from tt_blackhole_benchmark import environment
from tt_blackhole_benchmark.environment import collect_environment


def test_collect_environment_records_reproducibility_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tt_metal_root = tmp_path / "tt-metal"

    def fake_collect_git_repository(
        root: str | Path,
    ) -> dict[str, object]:
        return {
            "root": str(Path(root).resolve()),
            "available": True,
            "commit": "a" * 40,
            "branch": "test-branch",
            "dirty": False,
        }

    monkeypatch.setattr(
        environment,
        "collect_git_repository",
        fake_collect_git_repository,
    )

    metadata = collect_environment(
        tt_metal_root=tt_metal_root,
        environment={
            "TT_METAL_HOME": str(tt_metal_root),
            "ARCH_NAME": "blackhole",
            "UNRELATED_SECRET": "must-not-be-recorded",
        },
    )

    assert metadata["python"]["executable"] == sys.executable
    assert metadata["python"]["version"]
    assert metadata["system"]["machine"]
    assert metadata["system"]["system"]
    assert metadata["repositories"]["tt_metal"]["root"] == str(tt_metal_root.resolve())
    assert metadata["repositories"]["tt_metal"]["commit"] == "a" * 40
    assert metadata["environment_variables"] == {
        "TT_METAL_HOME": str(tt_metal_root),
        "ARCH_NAME": "blackhole",
    }
    assert "software" in metadata
    assert "tt-smi" in metadata["software"]["python_packages"]


def test_collect_environment_records_slow_dispatch_when_present(
    tmp_path: Path,
) -> None:
    metadata = collect_environment(
        tt_metal_root=tmp_path,
        environment={
            "TT_METAL_SLOW_DISPATCH_MODE": "1",
        },
    )

    assert metadata["environment_variables"]["TT_METAL_SLOW_DISPATCH_MODE"] == "1"


def test_collect_environment_omits_absent_variables(
    tmp_path: Path,
) -> None:
    metadata = collect_environment(
        tt_metal_root=tmp_path,
        environment={},
    )

    assert metadata["environment_variables"] == {}


def test_collect_git_repository_records_revision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = {
        ("rev-parse", "--is-inside-work-tree"): "true",
        ("rev-parse", "HEAD"): "b" * 40,
        (
            "status",
            "--porcelain",
            "--untracked-files=normal",
        ): " M src/example.py",
        (
            "symbolic-ref",
            "--quiet",
            "--short",
            "HEAD",
        ): "feature/benchmark-cli",
    }

    def fake_run_git(
        root: Path,
        *arguments: str,
    ) -> str:
        assert root == tmp_path.resolve()
        return responses[arguments]

    monkeypatch.setattr(environment, "_run_git", fake_run_git)

    metadata = environment.collect_git_repository(tmp_path)

    assert metadata == {
        "root": str(tmp_path.resolve()),
        "available": True,
        "branch": "feature/benchmark-cli",
        "commit": "b" * 40,
        "dirty": True,
    }


def test_collect_git_repository_handles_unavailable_repository(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run_git(
        root: Path,
        *arguments: str,
    ) -> str:
        raise RuntimeError("not a Git repository")

    monkeypatch.setattr(environment, "_run_git", fake_run_git)

    metadata = environment.collect_git_repository(tmp_path)

    assert metadata["root"] == str(tmp_path.resolve())
    assert metadata["available"] is False
    assert metadata["error"] == "not a Git repository"


def test_collect_software_versions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    versions = {
        "tt-blackhole-benchmark": "0.1.0",
        "tt-smi": "5.2.0",
        "tt-umd": "0.9.5",
    }

    def fake_package_version(name: str) -> str:
        if name not in versions:
            raise environment.PackageNotFoundError(name)

        return versions[name]

    monkeypatch.setattr(
        environment,
        "package_version",
        fake_package_version,
    )
    monkeypatch.setattr(
        environment.shutil,
        "which",
        lambda command: "/venv/bin/tt-smi" if command == "tt-smi" else None,
    )

    metadata = environment.collect_software_versions()

    assert metadata["python_packages"]["tt-blackhole-benchmark"] == "0.1.0"
    assert metadata["python_packages"]["tt-smi"] == "5.2.0"
    assert metadata["python_packages"]["tt-umd"] == "0.9.5"
    assert metadata["python_packages"]["tt-flash"] is None
    assert metadata["executables"]["tt-smi"] == "/venv/bin/tt-smi"

"""Acquisition of reproducibility metadata from the execution environment."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from collections.abc import Mapping
from importlib.metadata import (
    PackageNotFoundError,
)
from importlib.metadata import (
    version as package_version,
)
from pathlib import Path
from typing import Any

_RELEVANT_ENVIRONMENT_VARIABLES = (
    "TT_METAL_HOME",
    "TT_METAL_SLOW_DISPATCH_MODE",
    "ARCH_NAME",
)

_RELEVANT_PACKAGES = (
    "tt-blackhole-benchmark",
    "tt-flash",
    "tt-smi",
    "tt-tools-common",
    "tt-umd",
    "vllm-tt-plugin",
)


def _run_git(
    root: Path,
    *arguments: str,
) -> str:
    """Execute a read-only Git query in one repository."""

    completed = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=False,
        capture_output=True,
        text=True,
        timeout=5.0,
    )

    if completed.returncode != 0:
        message = completed.stderr.strip() or "Git command failed"
        raise RuntimeError(message)

    return completed.stdout.strip()


def collect_git_repository(root: str | Path) -> dict[str, Any]:
    """Collect the current revision and working-tree state."""

    repository_root = Path(root).resolve()
    metadata: dict[str, Any] = {
        "root": str(repository_root),
    }

    try:
        inside_work_tree = _run_git(
            repository_root,
            "rev-parse",
            "--is-inside-work-tree",
        )

        if inside_work_tree != "true":
            raise RuntimeError("path is not inside a Git working tree")

        commit = _run_git(
            repository_root,
            "rev-parse",
            "HEAD",
        )
        status = _run_git(
            repository_root,
            "status",
            "--porcelain",
            "--untracked-files=normal",
        )

        try:
            branch = _run_git(
                repository_root,
                "symbolic-ref",
                "--quiet",
                "--short",
                "HEAD",
            )
        except RuntimeError:
            branch = None

        metadata.update(
            {
                "available": True,
                "branch": branch,
                "commit": commit,
                "dirty": bool(status),
            }
        )
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
        metadata.update(
            {
                "available": False,
                "error": str(error),
            }
        )

    return metadata


def collect_environment(
    *,
    tt_metal_root: str | Path,
    benchmark_root: str | Path | None = None,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Collect directly observable execution-environment metadata."""

    selected_environment = os.environ if environment is None else environment
    root = Path(tt_metal_root).resolve()
    selected_benchmark_root = (
        Path(__file__).resolve().parents[2]
        if benchmark_root is None
        else Path(benchmark_root).resolve()
    )

    relevant_variables = {
        name: selected_environment[name]
        for name in _RELEVANT_ENVIRONMENT_VARIABLES
        if name in selected_environment
    }

    return {
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "executable": sys.executable,
        },
        "system": {
            "hostname": platform.node(),
            "machine": platform.machine(),
            "platform": platform.platform(),
            "release": platform.release(),
            "system": platform.system(),
        },
        "repositories": {
            "benchmark": collect_git_repository(selected_benchmark_root),
            "tt_metal": collect_git_repository(root),
        },
        "environment_variables": relevant_variables,
        "software": collect_software_versions(),
    }


def collect_software_versions() -> dict[str, Any]:
    """Collect relevant installed-package and executable versions."""

    packages: dict[str, str | None] = {}

    for name in _RELEVANT_PACKAGES:
        try:
            packages[name] = package_version(name)
        except PackageNotFoundError:
            packages[name] = None

    return {
        "python_packages": packages,
        "executables": {
            "tt-smi": shutil.which("tt-smi"),
        },
    }

"""Persistence of benchmark runs and their raw process observations."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from tt_blackhole_benchmark.executor.models import ProcessExecution
from tt_blackhole_benchmark.models import Workload
from tt_blackhole_benchmark.persistence.campaign import CampaignWorkspace
from tt_blackhole_benchmark.persistence.database import connect_database


class RunRepositoryError(RuntimeError):
    """Raised when a benchmark run cannot be persisted."""


@dataclass(frozen=True, slots=True)
class RunWorkspace:
    """Identity and artifact paths belonging to one benchmark run."""

    run_id: str
    campaign_id: str
    root: Path
    configuration_path: Path
    prompts_path: Path
    stdout_path: Path
    stderr_path: Path
    orchestration_error_path: Path
    run_phase: str = "measurement"
    if run_phase not in {"warmup", "measurement"}:
        raise ValueError("run_phase must be 'warmup' or 'measurement'")


def _relative_path(
    path: Path,
    campaign_root: Path,
) -> str:
    try:
        return path.relative_to(campaign_root).as_posix()
    except ValueError as error:
        raise RunRepositoryError(f"Artifact path is outside campaign directory: {path}") from error


def _write_text_atomic(
    path: Path,
    contents: str,
) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(contents, encoding="utf-8")
    temporary_path.replace(path)


def _write_json_atomic(
    path: Path,
    contents: Any,
) -> None:
    serialized = (
        json.dumps(
            contents,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n"
    )
    _write_text_atomic(path, serialized)


def create_run_workspace(
    *,
    campaign: CampaignWorkspace,
    model: str,
    workload: Workload,
    repetition_index: int,
    configuration: Mapping[str, Any],
    prompts: Sequence[Mapping[str, Any]],
    run_phase: str = "measurement",
    run_id: str | None = None,
) -> RunWorkspace:
    """Create the artifact directory and database row for one run."""

    if not model:
        raise ValueError("model must not be empty")

    if repetition_index < 0:
        raise ValueError("repetition_index must be non-negative")

    if run_phase not in {"warmup", "measurement"}:
        raise ValueError("run_phase must be 'warmup' or 'measurement'")
    identifier = run_id or str(uuid4())
    run_root = campaign.runs_directory / identifier
    orchestration_error_path = run_root / "orchestrator_error.log"

    configuration_path = run_root / "configuration.json"
    prompts_path = run_root / "prompts.json"
    stdout_path = run_root / "stdout.log"
    stderr_path = run_root / "stderr.log"

    try:
        run_root.mkdir(parents=False, exist_ok=False)

        _write_json_atomic(configuration_path, configuration)
        _write_json_atomic(prompts_path, list(prompts))

        with connect_database(campaign.database_path) as connection:
            connection.execute(
                """
                INSERT INTO runs (run_id,
                                  campaign_id,
                                  repetition_index,
                                  run_phase,
                                  model,
                                  input_tokens_requested,
                                  output_tokens_requested,
                                  batch_size,
                                  request_count,
                                  status,
                                  stdout_path,
                                  stderr_path,
                                  prompt_path,
                                  configuration_path,
                                  orchestration_error_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    identifier,
                    campaign.campaign_id,
                    repetition_index,
                    run_phase,
                    model,
                    workload.input_tokens,
                    workload.output_tokens,
                    workload.batch_size,
                    workload.request_count,
                    "created",
                    _relative_path(stdout_path, campaign.root),
                    _relative_path(stderr_path, campaign.root),
                    _relative_path(prompts_path, campaign.root),
                    _relative_path(
                        configuration_path,
                        campaign.root,
                    ),
                    _relative_path(
                        orchestration_error_path,
                        campaign.root,
                    ),
                ),
            )
            connection.commit()
    except Exception as error:
        raise RunRepositoryError(f"Unable to create run workspace {identifier}") from error

    return RunWorkspace(
        run_id=identifier,
        campaign_id=campaign.campaign_id,
        root=run_root,
        configuration_path=configuration_path,
        prompts_path=prompts_path,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        orchestration_error_path=orchestration_error_path,
    )


def record_process_execution(
    *,
    campaign: CampaignWorkspace,
    run: RunWorkspace,
    execution: ProcessExecution,
) -> None:
    """Persist raw process output and lifecycle observations."""

    if run.campaign_id != campaign.campaign_id:
        raise ValueError("Run does not belong to the supplied campaign")

    if execution.timed_out:
        status = "timed_out"
    elif execution.return_code == 0:
        status = "completed"
    else:
        status = "failed"

    try:
        _write_text_atomic(run.stdout_path, execution.stdout)
        _write_text_atomic(run.stderr_path, execution.stderr)

        with connect_database(campaign.database_path) as connection:
            cursor = connection.execute(
                """
                UPDATE runs
                SET
                    status = ?,
                    started_at = ?,
                    finished_at = ?,
                    started_monotonic_ns = ?,
                    finished_monotonic_ns = ?,
                    return_code = ?,
                    timed_out = ?
                WHERE run_id = ? AND campaign_id = ?
                """,
                (
                    status,
                    execution.started_at.isoformat(),
                    execution.finished_at.isoformat(),
                    execution.started_monotonic_ns,
                    execution.finished_monotonic_ns,
                    execution.return_code,
                    int(execution.timed_out),
                    run.run_id,
                    campaign.campaign_id,
                ),
            )

            if cursor.rowcount != 1:
                raise RunRepositoryError(f"Run not found in database: {run.run_id}")

            connection.commit()
    except RunRepositoryError:
        raise
    except Exception as error:
        raise RunRepositoryError(f"Unable to record execution for run {run.run_id}") from error


def record_orchestration_failure(
    *,
    campaign: CampaignWorkspace,
    run: RunWorkspace,
    status: str,
    diagnostic: str,
) -> None:
    """Persist a raw orchestration failure diagnostic."""

    allowed_statuses = {
        "executor_failed",
        "telemetry_failed",
    }

    if status not in allowed_statuses:
        raise ValueError(f"Unsupported orchestration failure status: {status}")

    if not diagnostic:
        raise ValueError("diagnostic must not be empty")

    if run.campaign_id != campaign.campaign_id:
        raise ValueError("Run does not belong to the supplied campaign")

    try:
        _write_text_atomic(
            run.orchestration_error_path,
            diagnostic,
        )

        with connect_database(campaign.database_path) as connection:
            cursor = connection.execute(
                """
                UPDATE runs
                SET status = ?
                WHERE run_id = ? AND campaign_id = ?
                """,
                (
                    status,
                    run.run_id,
                    campaign.campaign_id,
                ),
            )

            if cursor.rowcount != 1:
                raise RunRepositoryError(f"Run not found in database: {run.run_id}")

            connection.commit()
    except RunRepositoryError:
        raise
    except Exception as error:
        raise RunRepositoryError(
            f"Unable to record orchestration failure for run {run.run_id}"
        ) from error

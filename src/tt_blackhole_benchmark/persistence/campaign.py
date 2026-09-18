"""Creation and representation of benchmark campaign workspaces."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml

from tt_blackhole_benchmark.persistence.database import (
    connect_database,
    initialize_database,
)

_CAMPAIGN_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


class CampaignWorkspaceError(RuntimeError):
    """Raised when a campaign workspace cannot be created."""


@dataclass(frozen=True, slots=True)
class CampaignWorkspace:
    """Filesystem paths and identity associated with one benchmark campaign."""

    campaign_id: str
    name: str
    created_at: datetime
    root: Path
    database_path: Path
    runs_directory: Path
    configuration_path: Path
    environment_path: Path


def validate_campaign_name(name: str) -> str:
    """Validate and return a filesystem-safe campaign name."""

    if not _CAMPAIGN_NAME_PATTERN.fullmatch(name):
        raise ValueError(
            "Campaign name must contain 1 to 64 characters and use only "
            "letters, numbers, dots, underscores, or hyphens; "
            "the first character must be alphanumeric"
        )

    return name


def _write_json(
    path: Path,
    contents: Mapping[str, Any],
) -> None:
    path.write_text(
        json.dumps(
            contents,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_yaml(
    path: Path,
    contents: Mapping[str, Any],
) -> None:
    path.write_text(
        yaml.safe_dump(
            dict(contents),
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )


def create_campaign_workspace(
    *,
    results_directory: str | Path,
    name: str,
    configuration: Mapping[str, Any],
    environment: Mapping[str, Any],
    created_at: datetime | None = None,
    campaign_id: str | None = None,
) -> CampaignWorkspace:
    """Create and initialize the workspace for one benchmark campaign."""

    validated_name = validate_campaign_name(name)

    creation_time = created_at or datetime.now(UTC)
    if creation_time.tzinfo is None:
        raise ValueError("created_at must be timezone-aware")

    creation_time = creation_time.astimezone(UTC)
    identifier = campaign_id or str(uuid4())

    directory_timestamp = creation_time.strftime("%Y%m%dT%H%M%SZ")
    directory_name = f"{directory_timestamp}_{validated_name}_{identifier[:8]}"

    results_root = Path(results_directory).resolve()
    campaign_root = results_root / directory_name

    database_path = campaign_root / "benchmark.sqlite"
    runs_directory = campaign_root / "runs"
    configuration_path = campaign_root / "campaign.yaml"
    environment_path = campaign_root / "environment.json"

    try:
        campaign_root.mkdir(parents=True, exist_ok=False)
        runs_directory.mkdir()

        _write_yaml(configuration_path, configuration)
        _write_json(environment_path, environment)

        with connect_database(database_path) as connection:
            initialize_database(connection)
            connection.execute(
                """
                INSERT INTO campaigns (
                    campaign_id,
                    name,
                    created_at,
                    configuration_path,
                    environment_path
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    identifier,
                    validated_name,
                    creation_time.isoformat(),
                    configuration_path.relative_to(campaign_root).as_posix(),
                    environment_path.relative_to(campaign_root).as_posix(),
                ),
            )
            connection.commit()
    except Exception as error:
        raise CampaignWorkspaceError(
            f"Unable to create campaign workspace at {campaign_root}"
        ) from error

    return CampaignWorkspace(
        campaign_id=identifier,
        name=validated_name,
        created_at=creation_time,
        root=campaign_root,
        database_path=database_path,
        runs_directory=runs_directory,
        configuration_path=configuration_path,
        environment_path=environment_path,
    )

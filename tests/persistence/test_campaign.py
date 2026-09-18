"""Unit tests for benchmark campaign workspaces."""

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml

from tt_blackhole_benchmark.persistence.campaign import (
    CampaignWorkspaceError,
    create_campaign_workspace,
    validate_campaign_name,
)


def test_validate_campaign_name_accepts_safe_name() -> None:
    assert validate_campaign_name("qwen-batch-sweep_01") == ("qwen-batch-sweep_01")


@pytest.mark.parametrize(
    "name",
    [
        "",
        "../campaign",
        "campaign/name",
        "campaign name",
        "_campaign",
        "a" * 65,
    ],
)
def test_validate_campaign_name_rejects_invalid_name(
    name: str,
) -> None:
    with pytest.raises(ValueError):
        validate_campaign_name(name)


def test_create_campaign_workspace(
    tmp_path: Path,
) -> None:
    created_at = datetime(2026, 9, 17, 12, 30, tzinfo=UTC)

    configuration = {
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "workloads": [
            {
                "input_tokens": 64,
                "output_tokens": 8,
                "batch_size": 1,
            }
        ],
    }
    environment = {
        "hostname": "mc-peak-2",
        "architecture": "riscv64",
    }

    workspace = create_campaign_workspace(
        results_directory=tmp_path,
        name="smoke-test",
        configuration=configuration,
        environment=environment,
        created_at=created_at,
        campaign_id="12345678-1234-5678-1234-567812345678",
    )

    assert workspace.root.name == ("20260917T123000Z_smoke-test_12345678")
    assert workspace.database_path.is_file()
    assert workspace.runs_directory.is_dir()
    assert workspace.configuration_path.is_file()
    assert workspace.environment_path.is_file()

    saved_configuration = yaml.safe_load(workspace.configuration_path.read_text(encoding="utf-8"))
    saved_environment = json.loads(workspace.environment_path.read_text(encoding="utf-8"))

    assert saved_configuration == configuration
    assert saved_environment == environment


def test_campaign_database_contains_relative_paths(
    tmp_path: Path,
) -> None:
    workspace = create_campaign_workspace(
        results_directory=tmp_path,
        name="relative-paths",
        configuration={"model": "test-model"},
        environment={"hostname": "test-host"},
        created_at=datetime(2026, 9, 17, tzinfo=UTC),
        campaign_id="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
    )

    with sqlite3.connect(workspace.database_path) as connection:
        row = connection.execute(
            """
            SELECT
                campaign_id,
                configuration_path,
                environment_path
            FROM campaigns
            """
        ).fetchone()

    assert row == (
        "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        "campaign.yaml",
        "environment.json",
    )

    assert not Path(row[1]).is_absolute()
    assert not Path(row[2]).is_absolute()


def test_campaign_directories_are_unique(
    tmp_path: Path,
) -> None:
    created_at = datetime(2026, 9, 17, tzinfo=UTC)

    first = create_campaign_workspace(
        results_directory=tmp_path,
        name="same-name",
        configuration={},
        environment={},
        created_at=created_at,
        campaign_id="11111111-1111-1111-1111-111111111111",
    )
    second = create_campaign_workspace(
        results_directory=tmp_path,
        name="same-name",
        configuration={},
        environment={},
        created_at=created_at,
        campaign_id="22222222-2222-2222-2222-222222222222",
    )

    assert first.root != second.root
    assert first.root.is_dir()
    assert second.root.is_dir()


def test_existing_campaign_directory_is_not_overwritten(
    tmp_path: Path,
) -> None:
    arguments = {
        "results_directory": tmp_path,
        "name": "existing",
        "configuration": {},
        "environment": {},
        "created_at": datetime(2026, 9, 17, tzinfo=UTC),
        "campaign_id": "12345678-1234-1234-1234-123456789abc",
    }

    first = create_campaign_workspace(**arguments)
    marker = first.root / "do-not-delete.txt"
    marker.write_text("preserve me", encoding="utf-8")

    with pytest.raises(CampaignWorkspaceError):
        create_campaign_workspace(**arguments)

    assert marker.read_text(encoding="utf-8") == "preserve me"


def test_rejects_naive_creation_timestamp(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match="created_at must be timezone-aware",
    ):
        create_campaign_workspace(
            results_directory=tmp_path,
            name="invalid-time",
            configuration={},
            environment={},
            created_at=datetime(2026, 9, 17),
        )

"""Tests for the SQLite campaign schema."""

import sqlite3
from pathlib import Path

import pytest

from tt_blackhole_benchmark.persistence.database import (
    SCHEMA_VERSION,
    DatabaseSchemaError,
    connect_database,
    initialize_database,
)

EXPECTED_TABLES = {
    "campaigns",
    "runs",
    "schema_metadata",
    "telemetry_failures",
    "telemetry_samples",
}


def table_names(
    connection: sqlite3.Connection,
) -> set[str]:
    rows = connection.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
        """
    ).fetchall()

    return {row["name"] for row in rows if not row["name"].startswith("sqlite_")}


def column_names(
    connection: sqlite3.Connection,
    table: str,
) -> set[str]:
    rows = connection.execute(f"PRAGMA table_info({table})")
    return {row["name"] for row in rows}


def test_initialize_database_creates_schema(
    tmp_path: Path,
) -> None:
    connection = connect_database(tmp_path / "benchmark.sqlite")

    try:
        initialize_database(connection)

        assert table_names(connection) == EXPECTED_TABLES

        version = connection.execute(
            """
            SELECT schema_version
            FROM schema_metadata
            """
        ).fetchone()

        assert version["schema_version"] == SCHEMA_VERSION
    finally:
        connection.close()


def test_foreign_keys_are_enabled(
    tmp_path: Path,
) -> None:
    connection = connect_database(tmp_path / "benchmark.sqlite")

    try:
        enabled = connection.execute("PRAGMA foreign_keys").fetchone()[0]

        assert enabled == 1
    finally:
        connection.close()


def test_schema_contains_no_derived_metrics(
    tmp_path: Path,
) -> None:
    connection = connect_database(tmp_path / "benchmark.sqlite")

    forbidden_fragments = {
        "duration",
        "energy",
        "joule",
        "throughput",
        "tokens_per_second",
        "average_power",
        "sampling_interval",
    }

    try:
        initialize_database(connection)

        all_columns = set()

        for table in EXPECTED_TABLES:
            all_columns.update(column_names(connection, table))

        for fragment in forbidden_fragments:
            assert all(fragment not in column for column in all_columns)
    finally:
        connection.close()


def test_initialize_database_is_idempotent(
    tmp_path: Path,
) -> None:
    connection = connect_database(tmp_path / "benchmark.sqlite")

    try:
        initialize_database(connection)
        initialize_database(connection)

        rows = connection.execute("SELECT COUNT(*) FROM schema_metadata").fetchone()[0]

        assert rows == 1
    finally:
        connection.close()


def test_rejects_incompatible_schema_version(
    tmp_path: Path,
) -> None:
    connection = connect_database(tmp_path / "benchmark.sqlite")

    try:
        initialize_database(connection)

        with connection:
            connection.execute(
                """
                UPDATE schema_metadata
                SET schema_version = ?
                """,
                (SCHEMA_VERSION + 1,),
            )

        with pytest.raises(
            DatabaseSchemaError,
            match="Unsupported database schema version",
        ):
            initialize_database(connection)
    finally:
        connection.close()

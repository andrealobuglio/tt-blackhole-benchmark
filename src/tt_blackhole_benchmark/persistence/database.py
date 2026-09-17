"""SQLite database initialization for benchmark campaigns."""

import sqlite3
from pathlib import Path

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_metadata (
    schema_version INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS campaigns (
    campaign_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    created_at TEXT NOT NULL,
    configuration_path TEXT NOT NULL,
    environment_path TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    campaign_id TEXT NOT NULL,
    repetition_index INTEGER NOT NULL,
    model TEXT NOT NULL,
    input_tokens_requested INTEGER NOT NULL,
    output_tokens_requested INTEGER NOT NULL,
    batch_size INTEGER NOT NULL,
    request_count INTEGER NOT NULL,
    status TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    started_monotonic_ns INTEGER,
    finished_monotonic_ns INTEGER,
    return_code INTEGER,
    timed_out INTEGER,
    stdout_path TEXT,
    stderr_path TEXT,
    prompt_path TEXT,
    configuration_path TEXT NOT NULL,
    FOREIGN KEY (campaign_id)
        REFERENCES campaigns(campaign_id)
        ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS telemetry_samples (
    sample_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    captured_at TEXT NOT NULL,
    monotonic_ns INTEGER NOT NULL,
    device_index INTEGER NOT NULL,
    bus_id TEXT NOT NULL,
    board_type TEXT NOT NULL,
    power_w REAL NOT NULL,
    voltage_v REAL NOT NULL,
    current_a REAL NOT NULL,
    asic_temperature_c REAL NOT NULL,
    aiclk_mhz REAL NOT NULL,
    fan_speed_percent REAL NOT NULL,
    heartbeat INTEGER NOT NULL,
    source TEXT NOT NULL,
    backend_command TEXT NOT NULL,
    backend_exit_code INTEGER NOT NULL,
    backend_stderr TEXT NOT NULL,
    UNIQUE (run_id, monotonic_ns, device_index),
    FOREIGN KEY (run_id)
        REFERENCES runs(run_id)
        ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS telemetry_failures (
    failure_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    captured_at TEXT NOT NULL,
    monotonic_ns INTEGER NOT NULL,
    message TEXT NOT NULL,
    FOREIGN KEY (run_id)
        REFERENCES runs(run_id)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_runs_campaign
    ON runs(campaign_id);

CREATE INDEX IF NOT EXISTS idx_telemetry_run_time
    ON telemetry_samples(run_id, monotonic_ns);

CREATE INDEX IF NOT EXISTS idx_failures_run_time
    ON telemetry_failures(run_id, monotonic_ns);
"""


class DatabaseSchemaError(RuntimeError):
    """Raised when the database schema is incompatible."""


def connect_database(path: str | Path) -> sqlite3.Connection:
    """Open a configured SQLite campaign database."""

    database_path = Path(path)
    database_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row

    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA synchronous = FULL")
    connection.execute("PRAGMA busy_timeout = 5000")

    return connection


def initialize_database(
    connection: sqlite3.Connection,
) -> None:
    """Create or validate the campaign database schema."""

    with connection:
        connection.executescript(SCHEMA)

        row = connection.execute(
            """
            SELECT schema_version
            FROM schema_metadata
            """
        ).fetchone()

        if row is None:
            connection.execute(
                """
                INSERT INTO schema_metadata(schema_version)
                VALUES (?)
                """,
                (SCHEMA_VERSION,),
            )
        elif row["schema_version"] != SCHEMA_VERSION:
            raise DatabaseSchemaError(
                f"Unsupported database schema version: {row['schema_version']}"
            )

"""Persistence of raw benchmark observations."""

from tt_blackhole_benchmark.persistence.campaign import (
    CampaignWorkspace,
    CampaignWorkspaceError,
    create_campaign_workspace,
    validate_campaign_name,
)
from tt_blackhole_benchmark.persistence.database import (
    SCHEMA_VERSION,
    DatabaseSchemaError,
    connect_database,
    initialize_database,
)
from tt_blackhole_benchmark.persistence.run_repository import (
    RunRepositoryError,
    RunWorkspace,
    create_run_workspace,
    record_process_execution,
)
from tt_blackhole_benchmark.persistence.telemetry_repository import (
    TelemetryRepositoryError,
    record_telemetry_acquisition,
    record_telemetry_failure,
)

__all__ = [
    "SCHEMA_VERSION",
    "DatabaseSchemaError",
    "connect_database",
    "initialize_database",
    "CampaignWorkspace",
    "CampaignWorkspaceError",
    "create_campaign_workspace",
    "validate_campaign_name",
    "RunRepositoryError",
    "RunWorkspace",
    "create_run_workspace",
    "record_process_execution",
    "TelemetryRepositoryError",
    "record_telemetry_acquisition",
    "record_telemetry_failure",
]

"""Hardware integration tests for the tt-smi backend."""

import pytest

from tt_blackhole_benchmark.telemetry.backend import (
    TtSmiBackend,
)


@pytest.mark.hardware
def test_tt_smi_acquires_blackhole_telemetry() -> None:
    backend = TtSmiBackend(timeout_seconds=30.0)

    acquisition = backend.acquire()
    samples = acquisition.snapshot.samples

    assert samples
    assert samples[0].board_type == "p150a"
    assert samples[0].bus_id
    assert samples[0].power_w > 0
    assert samples[0].voltage_v > 0
    assert samples[0].current_a > 0
    assert samples[0].aiclk_mhz > 0
    assert acquisition.command == ("tt-smi", "-s")

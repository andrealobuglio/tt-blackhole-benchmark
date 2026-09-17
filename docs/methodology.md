## Telemetry data policy

The primary dataset stores only directly observed measurements and
execution metadata. Telemetry samples include the UTC timestamp,
monotonic timestamp, device identity, power, voltage, current,
temperature, AI clock, fan speed, and firmware heartbeat.

Derived quantities are not stored in the primary tables. Energy,
average power, sampling intervals, throughput per watt, joules per
output token, output tokens per joule, percentiles, and confidence
intervals are computed in analysis notebooks. This allows alternative
analysis methods to be applied to the same raw observations.

Telemetry is sampled through a persistent, process-isolated TT-SMI
backend. Device discovery and backend initialization occur once, while
telemetry is refreshed at 500 ms intervals. The process isolation
contains a native cleanup failure observed with TT-SMI 5.2.0 on the
riscv64 host.
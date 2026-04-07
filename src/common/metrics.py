from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Optional


@dataclass(slots=True)
class AccessMetrics:
    protocol: str

    # Bucket-level server I/O
    online_bucket_reads: int = 0
    online_bucket_writes: int = 0
    offline_bucket_reads: int = 0
    offline_bucket_writes: int = 0

    # Byte-level communication
    online_bytes_down: int = 0
    online_bytes_up: int = 0
    offline_bytes_down: int = 0
    offline_bytes_up: int = 0

    # RTT accounting
    online_rtt: int = 0
    offline_rtt: int = 0

    # Request composition
    real_requests_served: int = 0
    virtual_requests_executed: int = 0
    virtual_ticks_generated: int = 0

    # Queue / degradation
    queue_length_before: int = 0
    queue_length_after: int = 0
    fallback_flag: bool = False

    # Local state
    stash_size_before: int = 0
    stash_size_after: int = 0

    # Optional counters for debugging / appendix
    path_length_touched: int = 0
    eviction_count: int = 0
    reshuffle_count: int = 0
    dummy_blocks_read: int = 0
    dummy_blocks_written: int = 0

    def add(self, other: "AccessMetrics") -> None:
        if self.protocol != other.protocol:
            raise ValueError(
                f"Cannot merge metrics from different protocols: "
                f"{self.protocol!r} vs {other.protocol!r}"
            )

        for f in fields(self):
            name = f.name
            if name == "protocol":
                continue

            lhs = getattr(self, name)
            rhs = getattr(other, name)

            if isinstance(lhs, bool):
                setattr(self, name, lhs or rhs)
            else:
                setattr(self, name, lhs + rhs)

    def clone(self) -> "AccessMetrics":
        copied = AccessMetrics(protocol=self.protocol)
        for f in fields(self):
            name = f.name
            if name == "protocol":
                continue
            setattr(copied, name, getattr(self, name))
        return copied

    def record_bucket_read(
        self,
        *,
        online: bool,
        byte_count: int,
        rtt_count: int = 0,
        dummy_blocks: int = 0,
    ) -> None:
        if online:
            self.online_bucket_reads += 1
            self.online_bytes_down += byte_count
            self.online_rtt += rtt_count
        else:
            self.offline_bucket_reads += 1
            self.offline_bytes_down += byte_count
            self.offline_rtt += rtt_count

        self.dummy_blocks_read += dummy_blocks

    def record_bucket_write(
        self,
        *,
        online: bool,
        byte_count: int,
        rtt_count: int = 0,
        dummy_blocks: int = 0,
    ) -> None:
        if online:
            self.online_bucket_writes += 1
            self.online_bytes_up += byte_count
            self.online_rtt += rtt_count
        else:
            self.offline_bucket_writes += 1
            self.offline_bytes_up += byte_count
            self.offline_rtt += rtt_count

        self.dummy_blocks_written += dummy_blocks

    @property
    def total_bucket_reads(self) -> int:
        return self.online_bucket_reads + self.offline_bucket_reads

    @property
    def total_bucket_writes(self) -> int:
        return self.online_bucket_writes + self.offline_bucket_writes

    @property
    def total_bytes_down(self) -> int:
        return self.online_bytes_down + self.offline_bytes_down

    @property
    def total_bytes_up(self) -> int:
        return self.online_bytes_up + self.offline_bytes_up

    @property
    def total_rtt(self) -> int:
        return self.online_rtt + self.offline_rtt


@dataclass(slots=True)
class TimingRecord:
    arrival_time: Optional[float] = None
    service_start_time: Optional[float] = None
    response_time: Optional[float] = None
    end_to_end_latency: Optional[float] = None

    network_time: float = 0.0
    server_io_time: float = 0.0
    crypto_time: float = 0.0
    client_cpu_time: float = 0.0
    queueing_delay: float = 0.0
    total_modeled_latency: float = 0.0

    def finalize(self) -> None:
        self.total_modeled_latency = (
            self.network_time
            + self.server_io_time
            + self.crypto_time
            + self.client_cpu_time
            + self.queueing_delay
        )

        if self.arrival_time is not None and self.response_time is not None:
            self.end_to_end_latency = self.response_time - self.arrival_time


@dataclass(slots=True)
class DebugState:
    current_leaf: Optional[int] = None
    current_bucket: Optional[tuple[int, int]] = None
    mapped_bucket: Optional[tuple[int, int]] = None
    note: Optional[str] = None


@dataclass(slots=True)
class AccessResult:
    data: Optional[bytes]
    metrics: AccessMetrics
    timing: TimingRecord
    debug: DebugState = field(default_factory=DebugState)

    @classmethod
    def empty(cls, protocol: str) -> "AccessResult":
        return cls(
            data=None,
            metrics=AccessMetrics(protocol=protocol),
            timing=TimingRecord(),
            debug=DebugState(),
        )
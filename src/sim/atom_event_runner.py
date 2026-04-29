from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd

from src.common.config import AtomConfig
from src.common.latency_model import LatencyEstimate, LatencyModel
from src.common.types import BucketAddress, BlockAddress, OperationType, Request, RequestKind
from src.traces.schema import TraceRecord

time_eps = 1e-12

# state for compensation
@dataclass(slots=True)
class CompensationObligation:
    substituted_dummy: BucketAddress
    real_level: int
    real_index: int
    real_trace_id: int
    service_tail: float
    tail_applies: bool


# Trace runner for AtomORAM timer ticks and compensation scheduling
@dataclass(slots=True)
class AtomEventRunner:
    latency_model: LatencyModel
    atom_config: AtomConfig

    global_virtual_bytes_down: int = 0
    global_virtual_bytes_up: int = 0
    global_virtual_ticks_executed: int = 0

    required_virtual_bytes_down: int = 0
    required_virtual_bytes_up: int = 0
    required_virtual_ticks_executed: int = 0

    def run(
        self,
        *,
        protocol: Any,
        records: list[TraceRecord],
        block_size: int,
        required_virtual_ticks: int = 0,
        max_idle_ticks_after_last_arrival: int = 0,
        record_virtuals: Optional[bool] = None,
    ) -> pd.DataFrame:
        _ = required_virtual_ticks

        ordered = sorted(records, key=lambda r: (r.timestamp, r.trace_id))
        if not ordered:
            return pd.DataFrame()

        if record_virtuals is None:
            record_virtuals = len(ordered) <= 1500

        tick_interval = self.atom_config.tick_interval_sec
        if tick_interval <= 0:
            raise ValueError("tick_interval_sec must be positive.")

        pending_real = deque()
        rows: list[dict[str, Any]] = []

        next_arrival_idx = 0
        tick_index = 0
        tick_time = ordered[0].timestamp
        next_real_release_time = ordered[0].timestamp
        burst_tail_added = False
        idle_virtuals_after_last_arrival = 0
        compensation: Optional[CompensationObligation] = None

        self.global_virtual_bytes_down = 0
        self.global_virtual_bytes_up = 0
        self.global_virtual_ticks_executed = 0
        self.required_virtual_bytes_down = 0
        self.required_virtual_bytes_up = 0
        self.required_virtual_ticks_executed = 0

        while True:
            while (
                next_arrival_idx < len(ordered)
                and ordered[next_arrival_idx].timestamp <= tick_time + time_eps
            ):
                pending_real.append(ordered[next_arrival_idx])
                next_arrival_idx += 1

            if (
                burst_tail_added
                and compensation is None
                and not pending_real
                and tick_time + time_eps >= next_real_release_time
            ):
                burst_tail_added = False        

            # Compensation has priority over all real requests. 
            if compensation is not None:
                generated_dummy = self._sample_virtual_bucket(protocol)
                if generated_dummy.level == compensation.real_level:
                    execute_bucket = compensation.substituted_dummy
                    compensation_satisfied = True
                else:
                    execute_bucket = generated_dummy
                    compensation_satisfied = False

                result, estimate = self._execute_virtual_access(
                    protocol=protocol,
                    execute_bucket=execute_bucket,
                    tick_index=tick_index,
                    tick_time=tick_time,
                    pending_real_len=len(pending_real),
                )

                self._record_global_virtual(result)
                self._record_required_virtual(result)

                if record_virtuals:
                    row = self._make_row(
                        protocol=protocol,
                        record=None,
                        result=result,
                        estimate=estimate,
                        tick_index=tick_index,
                        tick_time=tick_time,
                        service_kind="virtual",
                        generated_virtual_tick=True,
                        executed_virtual_access=True,
                    )
                    row["required_virtual_access"] = True
                    row["compensation_wait_tick"] = True
                    row["compensation_satisfied"] = compensation_satisfied
                    row["generated_dummy_level"] = generated_dummy.level
                    row["generated_dummy_index"] = generated_dummy.index
                    row["executed_bucket_level"] = execute_bucket.level
                    row["executed_bucket_index"] = execute_bucket.index
                    row["compensating_real_level"] = compensation.real_level
                    row["compensating_real_index"] = compensation.real_index
                    row["compensating_real_trace_id"] = compensation.real_trace_id
                    rows.append(row)

                if compensation_satisfied:
                    if compensation.tail_applies:
                        next_real_release_time = max(
                            next_real_release_time,
                            tick_time + compensation.service_tail,
                        )
                    compensation = None

                tick_index += 1
                tick_time += tick_interval
                continue    

            if pending_real and tick_time + time_eps >= next_real_release_time:
                generated_dummy = self._sample_virtual_bucket(protocol)
                queue_length_before = len(pending_real)
                record = pending_real.popleft()
                queue_length_after = len(pending_real)

                request = Request(
                    request_id=record.trace_id,
                    kind=RequestKind.REAL,
                    op=record.op,
                    address=BlockAddress(logical_id=record.logical_id),
                    data=self._make_write_payload(record, block_size),
                    arrival_time=record.timestamp,
                    issued_time=tick_time,
                    tag=record.source,
                )

                result = protocol.access(request)
                result.timing.service_start_time = tick_time

                queueing_delay = tick_time - record.timestamp
                if queueing_delay < -time_eps:
                    raise ValueError(
                        "Negative queueing delay detected: "
                        f"tick_time={tick_time!r}, "
                        f"arrival_time={record.timestamp!r}, "
                        f"diff={queueing_delay!r}, "
                        f"tick_index={tick_index}, "
                        f"next_real_release_time={next_real_release_time!r}, "
                        f"pending_real_len={len(pending_real)}"
                    )

                queueing_delay = max(0.0, queueing_delay)

                result.metrics.queue_length_before = queue_length_before
                result.metrics.queue_length_after = queue_length_after
                result.metrics.fallback_flag = queueing_delay > time_eps
                result.metrics.virtual_ticks_generated = 1
                result.metrics.virtual_requests_executed = 0
                result.metrics.real_requests_served = 1

                estimate = self.latency_model.annotate(
                    result,
                    queueing_delay=queueing_delay,
                )

                if estimate.queueing_delay < 0:
                    raise ValueError("Negative queueing delay in latency estimate.")
                if estimate.online_latency < estimate.queueing_delay:
                    raise ValueError("Online latency smaller than queueing delay.")

                if result.debug.current_bucket is None:
                    raise ValueError("AtomORAM did not expose the real target bucket.")
                real_level, real_index = result.debug.current_bucket
                service_tail = estimate.total_latency - estimate.queueing_delay
                tail_applies = not burst_tail_added

                compensation = CompensationObligation(
                    substituted_dummy=generated_dummy,
                    real_level=real_level,
                    real_index=real_index,
                    real_trace_id=record.trace_id,
                    service_tail=service_tail,
                    tail_applies=tail_applies,
                )

                if tail_applies:
                    burst_tail_added = True

                row = self._make_row(
                    protocol=protocol,
                    record=record,
                    result=result,
                    estimate=estimate,
                    tick_index=tick_index,
                    tick_time=tick_time,
                    service_kind="real",
                    generated_virtual_tick=True,
                    executed_virtual_access=False,
                )
                row["required_virtual_access"] = False
                row["compensation_wait_tick"] = False
                row["compensation_satisfied"] = False
                row["generated_dummy_level"] = generated_dummy.level
                row["generated_dummy_index"] = generated_dummy.index
                row["executed_bucket_level"] = real_level
                row["executed_bucket_index"] = real_index
                row["compensating_real_level"] = real_level
                row["compensating_real_index"] = real_index
                row["compensating_real_trace_id"] = record.trace_id
                rows.append(row)

                idle_virtuals_after_last_arrival = 0
                tick_index += 1
                tick_time += tick_interval
                continue

            if pending_real and tick_time + time_eps < next_real_release_time:
                tick_time = next_real_release_time
                continue

            if next_arrival_idx < len(ordered):
                next_arrival_time = ordered[next_arrival_idx].timestamp
                if tick_time < next_arrival_time:
                    tick_time = next_arrival_time
                    continue

            if next_arrival_idx >= len(ordered) and not pending_real:
                if idle_virtuals_after_last_arrival >= max_idle_ticks_after_last_arrival:
                    break

                generated_dummy = self._sample_virtual_bucket(protocol)
                result, estimate = self._execute_virtual_access(
                    protocol=protocol,
                    execute_bucket=generated_dummy,
                    tick_index=tick_index,
                    tick_time=tick_time,
                    pending_real_len=0,
                )

                self._record_global_virtual(result)

                if record_virtuals:
                    row = self._make_row(
                        protocol=protocol,
                        record=None,
                        result=result,
                        estimate=estimate,
                        tick_index=tick_index,
                        tick_time=tick_time,
                        service_kind="virtual",
                        generated_virtual_tick=True,
                        executed_virtual_access=True,
                    )
                    row["required_virtual_access"] = False
                    row["compensation_wait_tick"] = False
                    row["compensation_satisfied"] = False
                    row["generated_dummy_level"] = generated_dummy.level
                    row["generated_dummy_index"] = generated_dummy.index
                    row["executed_bucket_level"] = generated_dummy.level
                    row["executed_bucket_index"] = generated_dummy.index
                    row["compensating_real_level"] = None
                    row["compensating_real_index"] = None
                    row["compensating_real_trace_id"] = None
                    rows.append(row)

                idle_virtuals_after_last_arrival += 1
                tick_index += 1
                tick_time += tick_interval
                continue

            break

        return pd.DataFrame(rows)

    @staticmethod
    def _make_write_payload(record: TraceRecord, block_size: int) -> Optional[bytes]:
        if record.op != OperationType.WRITE:
            return None
        return bytes([record.logical_id % 251]) * block_size

    @staticmethod
    def _sample_virtual_bucket(protocol: Any) -> BucketAddress:
        sampler = getattr(protocol, "sample_virtual_bucket_address", None)
        if sampler is None:
            raise AttributeError(
                "AtomEventRunner requires protocol.sample_virtual_bucket_address()."
            )
        return sampler()

    # execute one dummy access and annotate its modeled latency
    def _execute_virtual_access(
        self,
        *,
        protocol: Any,
        execute_bucket: BucketAddress,
        tick_index: int,
        tick_time: float,
        pending_real_len: int,
    ) -> tuple[Any, LatencyEstimate]:
        virtual_request = Request(
            request_id=-(tick_index + 1),
            kind=RequestKind.VIRTUAL,
            op=OperationType.READ,
            address=execute_bucket,
            data=None,
            arrival_time=tick_time,
            issued_time=tick_time,
            tag="virtual_tick",
        )

        result = protocol.access(virtual_request)
        result.timing.service_start_time = tick_time

        result.metrics.queue_length_before = pending_real_len
        result.metrics.queue_length_after = pending_real_len
        result.metrics.fallback_flag = False
        result.metrics.virtual_ticks_generated = 1
        result.metrics.virtual_requests_executed = 1
        result.metrics.real_requests_served = 0

        estimate = self.latency_model.annotate(result, queueing_delay=0.0)

        if estimate.queueing_delay < 0:
            raise ValueError("Negative queueing delay in latency estimate.")
        if estimate.online_latency < estimate.queueing_delay:
            raise ValueError("Online latency smaller than queueing delay.")

        return result, estimate

    def _record_global_virtual(self, result: Any) -> None:
        self.global_virtual_ticks_executed += 1
        self.global_virtual_bytes_down += result.metrics.total_bytes_down
        self.global_virtual_bytes_up += result.metrics.total_bytes_up

    def _record_required_virtual(self, result: Any) -> None:
        self.required_virtual_ticks_executed += 1
        self.required_virtual_bytes_down += result.metrics.total_bytes_down
        self.required_virtual_bytes_up += result.metrics.total_bytes_up

    @staticmethod
    def _make_row(
        *,
        protocol: Any,
        record: Optional[TraceRecord],
        result: Any,
        estimate: LatencyEstimate,
        tick_index: int,
        tick_time: float,
        service_kind: str,
        generated_virtual_tick: bool,
        executed_virtual_access: bool,
    ) -> dict[str, Any]:
        pending_flush_count = getattr(protocol, "pending_flush_count", None)

        if record is None:
            trace_id = None
            arrival_time = None
            logical_id = None
            size_bytes = None
            source = "virtual_tick"
            request_group = None
            op = "read"
        else:
            trace_id = record.trace_id
            arrival_time = record.timestamp
            logical_id = record.logical_id
            size_bytes = record.size_bytes
            source = record.source
            request_group = record.request_group
            op = record.op.value

        return {
            "tick_index": tick_index,
            "tick_time": tick_time,
            "service_kind": service_kind,
            "generated_virtual_tick": generated_virtual_tick,
            "executed_virtual_access": executed_virtual_access,
            "trace_id": trace_id,
            "arrival_time": arrival_time,
            "logical_id": logical_id,
            "size_bytes": size_bytes,
            "source": source,
            "request_group": request_group,
            "protocol": result.metrics.protocol,
            "service_start_time": result.timing.service_start_time,
            "response_time": result.timing.response_time,
            "end_to_end_latency": result.timing.end_to_end_latency,
            "online_latency": estimate.online_latency,
            "offline_latency": estimate.offline_latency,
            "total_latency": estimate.total_latency,
            "queueing_delay": estimate.queueing_delay,
            "op": op,
            "online_bucket_reads": result.metrics.online_bucket_reads,
            "online_bucket_writes": result.metrics.online_bucket_writes,
            "offline_bucket_reads": result.metrics.offline_bucket_reads,
            "offline_bucket_writes": result.metrics.offline_bucket_writes,
            "online_bytes_down": result.metrics.online_bytes_down,
            "online_bytes_up": result.metrics.online_bytes_up,
            "offline_bytes_down": result.metrics.offline_bytes_down,
            "offline_bytes_up": result.metrics.offline_bytes_up,
            "online_rtt": result.metrics.online_rtt,
            "offline_rtt": result.metrics.offline_rtt,
            "stash_size_before": result.metrics.stash_size_before,
            "stash_peak_during_access": result.metrics.stash_peak_during_access,
            "stash_size_after": result.metrics.stash_size_after,
            "queue_length_before": result.metrics.queue_length_before,
            "queue_length_after": result.metrics.queue_length_after,
            "fallback_flag": result.metrics.fallback_flag,
            "virtual_ticks_generated": result.metrics.virtual_ticks_generated,
            "virtual_requests_executed": result.metrics.virtual_requests_executed,
            "real_requests_served": result.metrics.real_requests_served,
            "pending_flush_count": pending_flush_count,
        }
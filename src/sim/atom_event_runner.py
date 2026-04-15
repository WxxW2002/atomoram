from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd

from src.common.config import AtomConfig
from src.common.latency_model import LatencyEstimate, LatencyModel
from src.common.types import BlockAddress, OperationType, Request, RequestKind
from src.traces.schema import TraceRecord


@dataclass(slots=True)
class AtomEventRunner:
    latency_model: LatencyModel
    atom_config: AtomConfig

    # 所有实际执行过的 virtual accesses
    global_virtual_bytes_down: int = 0
    global_virtual_bytes_up: int = 0
    global_virtual_ticks_executed: int = 0

    # 仅统计每个真实请求后“必须执行”的 mixing virtual accesses
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
        ordered = sorted(records, key=lambda r: (r.timestamp, r.trace_id))
        if not ordered:
            return pd.DataFrame()

        if record_virtuals is None:
            record_virtuals = len(ordered) <= 1500

        pending_real = deque()
        rows: list[dict[str, Any]] = []

        next_arrival_idx = 0
        slot_index = 0
        current_time = ordered[0].timestamp
        cooldown_remaining = 0
        idle_virtuals_after_last_arrival = 0

        self.global_virtual_bytes_down = 0
        self.global_virtual_bytes_up = 0
        self.global_virtual_ticks_executed = 0

        self.required_virtual_bytes_down = 0
        self.required_virtual_bytes_up = 0
        self.required_virtual_ticks_executed = 0

        while True:
            while (
                next_arrival_idx < len(ordered)
                and ordered[next_arrival_idx].timestamp <= current_time
            ):
                pending_real.append(ordered[next_arrival_idx])
                next_arrival_idx += 1

            pending_flush_count = getattr(protocol, "pending_flush_count", 0)

            should_run_real = (
                len(pending_real) > 0
                and cooldown_remaining <= 0
                and pending_flush_count <= 0
            )

            if should_run_real:
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
                    issued_time=current_time,
                    tag=record.source,
                )

                result = protocol.access(request)
                result.timing.service_start_time = current_time

                queueing_delay = current_time - record.timestamp
                if queueing_delay < 0:
                    raise ValueError("Negative queueing delay detected.")

                result.metrics.queue_length_before = queue_length_before
                result.metrics.queue_length_after = queue_length_after
                result.metrics.fallback_flag = queueing_delay > 0
                result.metrics.virtual_ticks_generated = 0
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

                rows.append(
                    self._make_row(
                        protocol=protocol,
                        record=record,
                        result=result,
                        estimate=estimate,
                        tick_index=slot_index,
                        tick_time=current_time,
                        service_kind="real",
                        generated_virtual_tick=False,
                        executed_virtual_access=False,
                    )
                )

                service_time = estimate.online_latency - estimate.queueing_delay
                if service_time < 0:
                    raise ValueError("Negative online service time detected.")

                current_time += service_time
                slot_index += 1

                cooldown_remaining = required_virtual_ticks
                idle_virtuals_after_last_arrival = 0
                continue

            need_virtual = (cooldown_remaining > 0) or (pending_flush_count > 0)

            if not need_virtual:
                if next_arrival_idx < len(ordered):
                    next_arrival_time = ordered[next_arrival_idx].timestamp
                    if current_time < next_arrival_time:
                        current_time = next_arrival_time
                    continue

                if idle_virtuals_after_last_arrival >= max_idle_ticks_after_last_arrival:
                    break

                idle_virtuals_after_last_arrival += 1

            is_required_virtual = cooldown_remaining > 0

            virtual_request = Request(
                request_id=-(slot_index + 1),
                kind=RequestKind.VIRTUAL,
                op=OperationType.READ,
                address=None,
                data=None,
                arrival_time=current_time,
                issued_time=current_time,
                tag="virtual_tick",
            )

            result = protocol.access(virtual_request)
            result.timing.service_start_time = current_time

            result.metrics.queue_length_before = len(pending_real)
            result.metrics.queue_length_after = len(pending_real)
            result.metrics.fallback_flag = False
            result.metrics.virtual_ticks_generated = 1
            result.metrics.virtual_requests_executed = 1
            result.metrics.real_requests_served = 0

            estimate = self.latency_model.annotate(result, queueing_delay=0.0)

            if estimate.queueing_delay < 0:
                raise ValueError("Negative queueing delay in latency estimate.")
            if estimate.online_latency < estimate.queueing_delay:
                raise ValueError("Online latency smaller than queueing delay.")

            self.global_virtual_ticks_executed += 1
            self.global_virtual_bytes_down += result.metrics.total_bytes_down
            self.global_virtual_bytes_up += result.metrics.total_bytes_up

            if is_required_virtual:
                self.required_virtual_ticks_executed += 1
                self.required_virtual_bytes_down += result.metrics.total_bytes_down
                self.required_virtual_bytes_up += result.metrics.total_bytes_up

            if record_virtuals:
                row = self._make_row(
                    protocol=protocol,
                    record=None,
                    result=result,
                    estimate=estimate,
                    tick_index=slot_index,
                    tick_time=current_time,
                    service_kind="virtual",
                    generated_virtual_tick=True,
                    executed_virtual_access=True,
                )
                row["required_virtual_access"] = is_required_virtual
                rows.append(row)

            service_time = estimate.online_latency - estimate.queueing_delay
            if service_time < 0:
                raise ValueError("Negative online service time detected.")

            current_time += service_time
            slot_index += 1

            if cooldown_remaining > 0:
                cooldown_remaining -= 1

        return pd.DataFrame(rows)

    @staticmethod
    def _make_write_payload(record: TraceRecord, block_size: int) -> Optional[bytes]:
        if record.op != OperationType.WRITE:
            return None
        payload_len = min(max(record.size_bytes, 1), block_size)
        fill_byte = record.logical_id % 251
        return bytes([fill_byte]) * payload_len

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
            "tick_index": tick_index,   # 兼容旧字段名；现在表示串行 access slot index
            "tick_time": tick_time,     # 兼容旧字段名；现在表示该 access 的实际开始时间
            "service_kind": service_kind,
            "generated_virtual_tick": generated_virtual_tick,
            "executed_virtual_access": executed_virtual_access,
            "trace_id": trace_id,
            "arrival_time": arrival_time,
            "service_start_time": tick_time,
            "response_time": result.timing.response_time,
            "end_to_end_latency": result.timing.end_to_end_latency,
            "online_latency": estimate.online_latency,
            "offline_latency": estimate.offline_latency,
            "total_latency": estimate.total_latency,
            "queueing_delay": estimate.queueing_delay,
            "op": op,
            "logical_id": logical_id,
            "size_bytes": size_bytes,
            "source": source,
            "request_group": request_group,
            "protocol": result.metrics.protocol,
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
            "stash_size_after": result.metrics.stash_size_after,
            "queue_length_before": result.metrics.queue_length_before,
            "queue_length_after": result.metrics.queue_length_after,
            "fallback_flag": result.metrics.fallback_flag,
            "virtual_ticks_generated": result.metrics.virtual_ticks_generated,
            "virtual_requests_executed": result.metrics.virtual_requests_executed,
            "real_requests_served": result.metrics.real_requests_served,
            "pending_flush_count": pending_flush_count,
        }
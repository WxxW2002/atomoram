from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd

from src.common.latency_model import LatencyEstimate, LatencyModel
from src.common.types import BlockAddress, OperationType, Request, RequestKind
from src.traces.schema import TraceRecord


@dataclass(slots=True)
class TraceRunner:
    latency_model: LatencyModel

    def run(
        self,
        *,
        protocol: Any,
        records: list[TraceRecord],
        block_size: int,
    ) -> pd.DataFrame:
        ordered = sorted(records, key=lambda r: (r.timestamp, r.trace_id))

        rows: list[dict[str, Any]] = []
        available_time = 0.0

        for record in ordered:
            arrival_time = record.timestamp
            service_start_time = max(arrival_time, available_time)
            queueing_delay = service_start_time - arrival_time

            request = Request(
                request_id=record.trace_id,
                kind=RequestKind.REAL,
                op=record.op,
                address=BlockAddress(logical_id=record.logical_id),
                data=self._make_write_payload(record, block_size)
                if record.op == OperationType.WRITE
                else None,
                arrival_time=arrival_time,
                issued_time=service_start_time,
                tag=record.source,
            )

            result = protocol.access(request)
            result.timing.service_start_time = service_start_time

            estimate = self.latency_model.annotate(
                result,
                queueing_delay=queueing_delay,
            )

            service_time_excluding_queue = estimate.total_latency - estimate.queueing_delay
            service_finish_time = service_start_time + service_time_excluding_queue
            available_time = service_finish_time

            rows.append(
                self._make_result_row(
                    protocol=protocol,
                    record=record,
                    result=result,
                    estimate=estimate,
                    service_start_time=service_start_time,
                    service_finish_time=service_finish_time,
                )
            )

        return pd.DataFrame(rows)

    @staticmethod
    def _make_write_payload(record: TraceRecord, block_size: int) -> bytes:
        payload_len = min(max(record.size_bytes, 1), block_size)
        fill_byte = record.logical_id % 251
        return bytes([fill_byte]) * payload_len

    @staticmethod
    def _make_result_row(
        *,
        protocol: Any,
        record: TraceRecord,
        result: Any,
        estimate: LatencyEstimate,
        service_start_time: float,
        service_finish_time: float,
    ) -> dict[str, Any]:
        pending_flush_count = getattr(protocol, "pending_flush_count", None)

        return {
            "trace_id": record.trace_id,
            "timestamp": record.timestamp,
            "service_start_time": service_start_time,
            "service_finish_time": service_finish_time,
            "response_time": result.timing.response_time,
            "end_to_end_latency": result.timing.end_to_end_latency,
            "online_latency": estimate.online_latency,
            "offline_latency": estimate.offline_latency,
            "total_latency": estimate.total_latency,
            "queueing_delay": estimate.queueing_delay,
            "op": record.op.value,
            "logical_id": record.logical_id,
            "size_bytes": record.size_bytes,
            "source": record.source,
            "request_group": record.request_group,
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
            "fallback_flag": result.metrics.fallback_flag,
            "virtual_ticks_generated": result.metrics.virtual_ticks_generated,
            "pending_flush_count": pending_flush_count,
        }
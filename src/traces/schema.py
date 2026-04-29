from __future__ import annotations

import pandas as pd
from dataclasses import dataclass, field

from typing import Any
from src.common.types import OperationType

# normalized trace event consumed by experiment runners
@dataclass(slots=True)
class TraceRecord:
    trace_id: int
    timestamp: float
    op: OperationType
    logical_id: int
    size_bytes: int
    source: str
    original_index: int
    original_offset: int
    request_group: int
    subrequest_index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


def normalize_operation(value: Any) -> OperationType:
    if isinstance(value, OperationType):
        return value

    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"read", "r"}:
            return OperationType.READ
        if normalized in {"write", "w"}:
            return OperationType.WRITE
        if normalized == "0":
            return OperationType.READ
        if normalized == "1":
            return OperationType.WRITE

    if isinstance(value, (int, float)):
        if int(value) == 0:
            return OperationType.READ
        if int(value) == 1:
            return OperationType.WRITE

    raise ValueError(f"Unsupported operation encoding: {value!r}")

def compact_trace_records(
    records: list[TraceRecord],
) -> tuple[list[TraceRecord], dict[int, int]]:
    mapping: dict[int, int] = {}
    next_id = 0
    compacted: list[TraceRecord] = []

    for record in records:
        if record.logical_id not in mapping:
            mapping[record.logical_id] = next_id
            next_id += 1

        compacted.append(
            TraceRecord(
                trace_id=record.trace_id,
                timestamp=record.timestamp,
                op=record.op,
                logical_id=mapping[record.logical_id],
                size_bytes=record.size_bytes,
                source=record.source,
                original_index=record.original_index,
                original_offset=record.original_offset,
                request_group=record.request_group,
                subrequest_index=record.subrequest_index,
                metadata=dict(record.metadata),
            )
        )

    return compacted, mapping

def make_single_request_record(
    *,
    trace_id: int,
    timestamp: float,
    op: OperationType,
    logical_id: int,
    block_size: int,
    source: str,
    original_index: int,
    original_offset: int,
    request_group: int,
    metadata: dict[str, Any] | None = None,
) -> TraceRecord:
    if block_size <= 0:
        raise ValueError("block_size must be positive.")
    if logical_id < 0:
        raise ValueError("logical_id must be non-negative.")

    return TraceRecord(
        trace_id=trace_id,
        timestamp=timestamp,
        op=op,
        logical_id=logical_id,
        size_bytes=block_size,
        source=source,
        original_index=original_index,
        original_offset=original_offset,
        request_group=request_group,
        subrequest_index=0,
        metadata=dict(metadata or {}),
    )

def records_to_dataframe(records: list[TraceRecord]) -> pd.DataFrame:
    rows = []
    for r in records:
        rows.append(
            {
                "trace_id": r.trace_id,
                "timestamp": r.timestamp,
                "op": r.op.value,
                "logical_id": r.logical_id,
                "size_bytes": r.size_bytes,
                "source": r.source,
                "original_index": r.original_index,
                "original_offset": r.original_offset,
                "request_group": r.request_group,
                "subrequest_index": r.subrequest_index,
                "metadata": r.metadata,
            }
        )
    return pd.DataFrame(rows)
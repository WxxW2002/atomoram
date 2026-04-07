from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.common.types import OperationType


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


def touched_block_ids(offset: int, size: int, block_size: int) -> list[int]:
    if offset < 0:
        raise ValueError("offset must be non-negative.")
    if size <= 0:
        raise ValueError("size must be positive.")
    if block_size <= 0:
        raise ValueError("block_size must be positive.")

    start_block = offset // block_size
    end_block = (offset + size - 1) // block_size
    return list(range(start_block, end_block + 1))


def split_request_into_block_records(
    *,
    base_trace_id: int,
    timestamp: float,
    op: OperationType,
    offset: int,
    size: int,
    block_size: int,
    source: str,
    original_index: int,
    request_group: int,
    metadata: dict[str, Any] | None = None,
) -> list[TraceRecord]:
    block_ids = touched_block_ids(offset=offset, size=size, block_size=block_size)
    records: list[TraceRecord] = []

    remaining = size
    block_offset = offset % block_size
    next_trace_id = base_trace_id

    for sub_idx, block_id in enumerate(block_ids):
        covered = min(block_size - block_offset, remaining)
        records.append(
            TraceRecord(
                trace_id=next_trace_id,
                timestamp=timestamp,
                op=op,
                logical_id=block_id,
                size_bytes=covered,
                source=source,
                original_index=original_index,
                original_offset=offset,
                request_group=request_group,
                subrequest_index=sub_idx,
                metadata=dict(metadata or {}),
            )
        )
        next_trace_id += 1
        remaining -= covered
        block_offset = 0

    return records


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
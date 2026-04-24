from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from src.traces.schema import (
    TraceRecord,
    compact_trace_records,
    make_single_request_record,
    normalize_operation,
)

ALICLOUD_COLUMNS = [
    "DeviceID",
    "Type",
    "Offset",
    "Size",
    "Timestamp",
]

ALICLOUD_TIME_UNIT = 1e6


def load_alicloud_trace(
    path: str | Path,
    *,
    block_size: int = 4096,
    max_rows: Optional[int] = None,
    compact_addresses: bool = True,
) -> list[TraceRecord]:
    csv_path = Path(path)
    df = pd.read_csv(
        csv_path,
        header=None,
        names=ALICLOUD_COLUMNS,
        low_memory=False,
        nrows=max_rows,
    )

    if not df.empty and str(df.iloc[0]["DeviceID"]).strip().lower() == "deviceid":
        df = df.iloc[1:].reset_index(drop=True)

    for col in ("Timestamp", "Offset", "Size"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Timestamp", "Offset", "Size", "Type"]).reset_index(drop=True)
    df = df[df["Size"] > 0].reset_index(drop=True)

    if df.empty:
        return []

    t0 = float(df.iloc[0]["Timestamp"])
    records: list[TraceRecord] = []
    next_trace_id = 0

    for row_idx, row in df.iterrows():
        timestamp_sec = (float(row["Timestamp"]) - t0) / ALICLOUD_TIME_UNIT
        op = normalize_operation(row["Type"])
        offset = int(row["Offset"])
        size = int(row["Size"])

        metadata = {
            "device_id": row["DeviceID"],
            "raw_size_bytes": size,
        }

        logical_id = offset // block_size
        records.append(
            make_single_request_record(
                trace_id=next_trace_id,
                timestamp=timestamp_sec,
                op=op,
                logical_id=logical_id,
                block_size=block_size,
                source="alicloud",
                original_index=row_idx,
                original_offset=offset,
                request_group=row_idx,
                metadata=metadata,
            )
        )
        next_trace_id += 1

    if compact_addresses:
        records, _ = compact_trace_records(records)

    return records
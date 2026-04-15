from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from src.traces.schema import TraceRecord, make_single_request_record, normalize_operation


GOOGLE_COLUMNS = [
    "filename",
    "file_offset",
    "application",
    "c_time",
    "io_zone",
    "redundancy_type",
    "op_type",
    "service_class",
    "from_flash_cache",
    "cache_hit",
    "request_io_size_bytes",
    "disk_io_size_bytes",
    "response_io_size_bytes",
    "start_time",
    "disk_time",
    "simulated_disk_start_time",
    "simulated_latency",
]


def load_google_trace(
    path: str | Path,
    *,
    block_size: int = 4096,
    max_rows: Optional[int] = None,
    compact_addresses: bool = True,
    split_multi_block_requests: bool = False,
) -> list[TraceRecord]:
    csv_path = Path(path)
    df = pd.read_csv(
        csv_path,
        low_memory=False,
        nrows=max_rows,
    )

    missing = [c for c in GOOGLE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Google trace missing required columns: {missing}")

    for col in ("file_offset", "c_time", "request_io_size_bytes", "start_time"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(
        subset=["filename", "file_offset", "request_io_size_bytes", "start_time", "op_type"]
    ).reset_index(drop=True)
    df = df[df["request_io_size_bytes"] > 0].reset_index(drop=True)

    if df.empty:
        return []

    t0 = float(df.iloc[0]["start_time"])
    records: list[TraceRecord] = []
    next_trace_id = 0

    key_to_id: dict[tuple[str, int, int], int] = {}
    next_logical_id = 0

    for row_idx, row in df.iterrows():
        timestamp_sec = float(row["start_time"]) - t0
        op = normalize_operation(row["op_type"])
        offset = int(row["file_offset"])
        size = int(row["request_io_size_bytes"])
        filename = str(row["filename"])
        c_time = int(row["c_time"])

        block_id = offset // block_size
        key = (filename, c_time, block_id)

        if key not in key_to_id:
            key_to_id[key] = next_logical_id
            next_logical_id += 1
        logical_id = key_to_id[key]

        metadata = {
            "application": row["application"],
            "io_zone": row["io_zone"],
            "redundancy_type": row["redundancy_type"],
            "service_class": row["service_class"],
            "from_flash_cache": row["from_flash_cache"],
            "cache_hit": row["cache_hit"],
            "disk_io_size_bytes": row["disk_io_size_bytes"],
            "response_io_size_bytes": row["response_io_size_bytes"],
            "disk_time": row["disk_time"],
            "simulated_disk_start_time": row["simulated_disk_start_time"],
            "simulated_latency": row["simulated_latency"],
            "filename": filename,
            "c_time": c_time,
            "raw_size_bytes": size,
        }

        records.append(
            make_single_request_record(
                trace_id=next_trace_id,
                timestamp=timestamp_sec,
                op=op,
                logical_id=logical_id,
                block_size=block_size,
                source="google",
                original_index=row_idx,
                original_offset=offset,
                request_group=row_idx,
                metadata=metadata,
            )
        )
        next_trace_id += 1

    return records
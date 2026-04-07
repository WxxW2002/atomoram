from __future__ import annotations

import random
from typing import Optional

from src.common.types import OperationType
from src.traces.schema import TraceRecord


def _sample_operation(rng: random.Random, read_ratio: float) -> OperationType:
    return OperationType.READ if rng.random() < read_ratio else OperationType.WRITE


def generate_constant_interval_trace(
    *,
    num_requests: int,
    address_space: int,
    interval_sec: float,
    read_ratio: float = 0.5,
    start_time: float = 0.0,
    seed: Optional[int] = None,
    source: str = "synthetic_constant",
    request_size_bytes: int = 4096,
) -> list[TraceRecord]:
    if num_requests < 0:
        raise ValueError("num_requests must be non-negative.")
    if address_space <= 0:
        raise ValueError("address_space must be positive.")
    if interval_sec < 0:
        raise ValueError("interval_sec must be non-negative.")

    rng = random.Random(seed)
    records: list[TraceRecord] = []

    for i in range(num_requests):
        records.append(
            TraceRecord(
                trace_id=i,
                timestamp=start_time + i * interval_sec,
                op=_sample_operation(rng, read_ratio),
                logical_id=rng.randrange(address_space),
                size_bytes=request_size_bytes,
                source=source,
                original_index=i,
                original_offset=-1,
                request_group=i,
                subrequest_index=0,
                metadata={},
            )
        )

    return records


def generate_sparse_trace(
    *,
    num_requests: int,
    address_space: int,
    alpha: float,
    lambda1: float,
    tree_height: int,
    t_virtual_sec: float,
    read_ratio: float = 0.5,
    start_time: float = 0.0,
    seed: Optional[int] = None,
    source: str = "synthetic_sparse",
    request_size_bytes: int = 4096,
) -> list[TraceRecord]:
    if alpha < 0:
        raise ValueError("alpha must be non-negative.")
    if lambda1 <= 0:
        raise ValueError("lambda1 must be positive.")
    if tree_height < 0:
        raise ValueError("tree_height must be non-negative.")
    if t_virtual_sec < 0:
        raise ValueError("t_virtual_sec must be non-negative.")

    interval_sec = alpha * lambda1 * tree_height * t_virtual_sec
    return generate_constant_interval_trace(
        num_requests=num_requests,
        address_space=address_space,
        interval_sec=interval_sec,
        read_ratio=read_ratio,
        start_time=start_time,
        seed=seed,
        source=source,
        request_size_bytes=request_size_bytes,
    )


def generate_two_burst_trace(
    *,
    burst1_size: int,
    burst2_size: int,
    address_space: int,
    intra_burst_interval_sec: float,
    idle_gap_sec: float,
    read_ratio: float = 0.5,
    start_time: float = 0.0,
    seed: Optional[int] = None,
    source: str = "synthetic_burst_gap",
    request_size_bytes: int = 4096,
) -> list[TraceRecord]:
    if burst1_size < 0 or burst2_size < 0:
        raise ValueError("burst sizes must be non-negative.")
    if intra_burst_interval_sec < 0:
        raise ValueError("intra_burst_interval_sec must be non-negative.")
    if idle_gap_sec < 0:
        raise ValueError("idle_gap_sec must be non-negative.")
    if address_space <= 0:
        raise ValueError("address_space must be positive.")

    rng = random.Random(seed)
    records: list[TraceRecord] = []

    t = start_time
    next_id = 0

    for _ in range(burst1_size):
        records.append(
            TraceRecord(
                trace_id=next_id,
                timestamp=t,
                op=_sample_operation(rng, read_ratio),
                logical_id=rng.randrange(address_space),
                size_bytes=request_size_bytes,
                source=source,
                original_index=next_id,
                original_offset=-1,
                request_group=0,
                metadata={"burst": 1},
            )
        )
        next_id += 1
        t += intra_burst_interval_sec

    t += idle_gap_sec

    for _ in range(burst2_size):
        records.append(
            TraceRecord(
                trace_id=next_id,
                timestamp=t,
                op=_sample_operation(rng, read_ratio),
                logical_id=rng.randrange(address_space),
                size_bytes=request_size_bytes,
                source=source,
                original_index=next_id,
                original_offset=-1,
                request_group=1,
                metadata={"burst": 2},
            )
        )
        next_id += 1
        t += intra_burst_interval_sec

    return records
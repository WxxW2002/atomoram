from __future__ import annotations

from typing import Optional

from src.common.config import StorageConfig
from src.common.interfaces import AbstractORAM
from src.common.metrics import AccessMetrics, AccessResult, TimingRecord
from src.common.utils import truncate_payload
from src.common.types import (
    BlockAddress,
    DataBlock,
    OperationType,
    ProtocolKind,
    Request,
    RequestKind,
)

class DirectStore(AbstractORAM):
    """
    Non-ORAM lower bound.
    """

    def __init__(self, config: StorageConfig) -> None:
        if config.block_size <= 0:
            raise ValueError("block_size must be positive.")
        self.config = config
        self.block_size = config.block_size
        self._store: dict[int, DataBlock] = {}

    def reset(self) -> None:
        self._store.clear()

    def access(self, request: Request) -> AccessResult:
        if request.address is None:
            raise ValueError("DirectStore.access() requires request.address.")

        logical_id = request.address.logical_id
        metrics = AccessMetrics(protocol=ProtocolKind.DIRECT.value)
        timing = TimingRecord(
            arrival_time=request.arrival_time,
            service_start_time=request.arrival_time,
        )

        if request.kind == RequestKind.REAL:
            metrics.real_requests_served = 1
        else:
            metrics.virtual_requests_executed = 1

        if request.op == OperationType.READ:
            metrics.record_bucket_read(
                online=True,
                byte_count=self.block_size,
                rtt_count=1,
                dummy_blocks=0,
            )
            data = truncate_payload(self._store.get(logical_id))

        elif request.op == OperationType.WRITE:
            if request.data is None:
                raise ValueError("WRITE request must carry request.data.")

            payload = request.data
            if len(payload) > self.block_size:
                raise ValueError(
                    f"Write payload exceeds block size: {len(payload)} > {self.block_size}"
                )

            self._store[logical_id] = DataBlock(
                block_id=logical_id,
                payload=bytes(payload),
                is_dummy=False,
                leaf=None,
                metadata={"logical_payload_size": len(payload)},
            )
            metrics.record_bucket_write(
                online=True,
                byte_count=self.block_size,
                rtt_count=1,
                dummy_blocks=0,
            )
            data = bytes(payload)

        else:
            raise ValueError(f"Unsupported operation: {request.op}")

        metrics.queue_length_before = 0
        metrics.queue_length_after = 0
        metrics.stash_size_before = 0
        metrics.stash_size_after = 0
        metrics.path_length_touched = 1

        if request.arrival_time is not None:
            timing.response_time = request.arrival_time
            timing.finalize()

        return AccessResult(data=data, metrics=metrics, timing=timing)
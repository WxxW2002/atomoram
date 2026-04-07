from __future__ import annotations

import random
from typing import Optional

from src.backend.tree_backend import TreeBackend
from src.common.config import StorageConfig
from src.common.interfaces import AbstractORAM
from src.common.metrics import AccessMetrics, AccessResult, TimingRecord
from src.common.types import (
    DataBlock,
    OperationType,
    ProtocolKind,
    Request,
    RequestKind,
)


def _truncate_payload(block: Optional[DataBlock]) -> Optional[bytes]:
    if block is None:
        return None
    logical_size = block.metadata.get("logical_payload_size", len(block.payload))
    return block.payload[:logical_size]


class PathORAM(AbstractORAM):
    """
    Path ORAM adapted from the V-ORAM reference implementation.

    Core flow:
    1. Read the whole path for the currently assigned leaf.
    2. Move all real blocks on that path into the stash.
    3. Access/update the target block in the stash.
    4. Reassign the target block to a new random leaf.
    5. Evict stash blocks back to the same path from leaf to root.

    Notes:
    - Entire access is treated as online cost.
    - This is an academic prototype, not a production implementation.
    """

    def __init__(
        self,
        config: StorageConfig,
        *,
        rng_seed: Optional[int] = None,
    ) -> None:
        self.config = config
        self.backend = TreeBackend(config=config)
        self.leaf_count = self.backend.leaf_count
        self.bucket_size = self.backend.bucket_size
        self.num_levels = self.backend.num_levels

        # Maximum logical capacity when the address space is densely packed.
        self.logical_block_capacity = self.leaf_count * self.bucket_size

        self._rng = random.Random(rng_seed)
        self.stash: dict[int, DataBlock] = {}
        self.position_map: list[Optional[int]] = [None] * self.logical_block_capacity

    def reset(self) -> None:
        self.backend.reset()
        self.stash.clear()
        self.position_map = [None] * self.logical_block_capacity

    def access(self, request: Request) -> AccessResult:
        if request.address is None:
            raise ValueError("PathORAM.access() requires request.address.")

        logical_id = request.address.logical_id
        self._validate_logical_id(logical_id)

        metrics = AccessMetrics(protocol=ProtocolKind.PATH.value)
        timing = TimingRecord(
            arrival_time=request.arrival_time,
            service_start_time=request.arrival_time,
        )
        metrics.stash_size_before = len(self.stash)
        metrics.queue_length_before = 0
        metrics.queue_length_after = 0

        if request.kind == RequestKind.REAL:
            metrics.real_requests_served = 1
        else:
            metrics.virtual_requests_executed = 1

        current_leaf = self.position_map[logical_id]
        if current_leaf is None:
            current_leaf = self._sample_leaf()

        new_leaf = self._sample_leaf()
        self.position_map[logical_id] = new_leaf

        path = self.backend.path_to_leaf(current_leaf)
        self._read_path_into_stash(path=path, metrics=metrics)

        result_data: Optional[bytes] = None

        if request.op == OperationType.WRITE:
            if request.data is None:
                raise ValueError("WRITE request must carry request.data.")
            if len(request.data) > self.backend.block_size:
                raise ValueError(
                    f"Write payload exceeds block size: "
                    f"{len(request.data)} > {self.backend.block_size}"
                )

            self.stash[logical_id] = DataBlock(
                block_id=logical_id,
                payload=bytes(request.data),
                is_dummy=False,
                leaf=new_leaf,
                metadata={"logical_payload_size": len(request.data)},
            )
            result_data = bytes(request.data)

        elif request.op == OperationType.READ:
            block = self.stash.get(logical_id)
            if block is not None:
                block.leaf = new_leaf
                result_data = _truncate_payload(block)
            else:
                result_data = None

        else:
            raise ValueError(f"Unsupported operation: {request.op}")

        self._evict_path(path=path, metrics=metrics)

        metrics.path_length_touched = self.num_levels
        metrics.stash_size_after = len(self.stash)

        # Follow V-ORAM’s accounting style: one RTT for reading the whole path,
        # one RTT for writing the whole path back.
        metrics.online_rtt = 2

        if request.arrival_time is not None:
            timing.response_time = request.arrival_time
            timing.finalize()

        result = AccessResult(data=result_data, metrics=metrics, timing=timing)
        result.debug.current_leaf = current_leaf
        result.debug.mapped_bucket = (path[-1].level, path[-1].index)
        result.debug.note = f"reassigned logical block {logical_id} to leaf {new_leaf}"
        return result

    def _validate_logical_id(self, logical_id: int) -> None:
        if logical_id < 0 or logical_id >= self.logical_block_capacity:
            raise ValueError(
                f"logical_id={logical_id} is outside the supported range "
                f"[0, {self.logical_block_capacity - 1}]."
            )

    def _sample_leaf(self) -> int:
        return self._rng.randrange(self.leaf_count)

    def _read_path_into_stash(self, path, metrics: AccessMetrics) -> None:
        for address in path:
            bucket = self.backend.read_bucket(address)
            metrics.record_bucket_read(
                online=True,
                byte_count=self.backend.bucket_storage_bytes,
                rtt_count=0,
                dummy_blocks=bucket.dummy_count(),
            )

            for block in bucket.non_dummy_blocks():
                if block.block_id is None:
                    continue
                self.stash[block.block_id] = block

    def _evict_path(self, path, metrics: AccessMetrics) -> None:
        # Evict from leaf to root, similar in spirit to the V-ORAM reference code.
        for address in reversed(path):
            eligible_ids = [
                block_id
                for block_id, block in self.stash.items()
                if block.leaf is not None and self.backend.is_bucket_on_path(address, block.leaf)
            ]

            selected_ids = self._select_blocks_for_bucket(eligible_ids)
            blocks_to_write = []

            for block_id in selected_ids:
                block = self.stash.pop(block_id)
                blocks_to_write.append(block)

            while len(blocks_to_write) < self.bucket_size:
                blocks_to_write.append(self.backend.make_dummy_block())

            bucket = self.backend.normalize_bucket(
                bucket=self.backend.make_empty_bucket(address)
            )
            bucket.blocks = [block.clone() for block in blocks_to_write]
            self.backend.write_bucket(bucket)

            metrics.record_bucket_write(
                online=True,
                byte_count=self.backend.bucket_storage_bytes,
                rtt_count=0,
                dummy_blocks=bucket.dummy_count(),
            )

    def _select_blocks_for_bucket(self, eligible_ids: list[int]) -> list[int]:
        if not eligible_ids:
            return []
        count = min(len(eligible_ids), self.bucket_size)
        if count == len(eligible_ids):
            return list(eligible_ids)
        return self._rng.sample(eligible_ids, count)
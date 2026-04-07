from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

from src.backend.tree_backend import TreeBackend
from src.common.config import AtomConfig, StorageConfig
from src.common.interfaces import AbstractORAM
from src.common.metrics import AccessMetrics, AccessResult, TimingRecord
from src.common.types import (
    Bucket,
    BucketAddress,
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


@dataclass(slots=True)
class PendingMaintenance:
    target_bucket: BucketAddress
    parent_bucket: BucketAddress
    request_id: int
    is_real: bool


class AtomORAM(AbstractORAM):
    """
    Phase-4 AtomORAM with:
    - online single-bucket access
    - immediate consumption of one flushable offline edge
    - stale-bucket invalidation
    - local two-bucket greedy writeback

    Current semantics:
    - access() completes one logical access:
        OnlineAccessBucket(u) + OfflineFlushEdge(parent(u), u) if parent exists.
    - root is a special case: no upward pairing is performed for root-target access.
    """

    def __init__(
        self,
        storage_config: StorageConfig,
        *,
        atom_config: Optional[AtomConfig] = None,
        rng_seed: Optional[int] = None,
    ) -> None:
        self.storage_config = storage_config
        self.atom_config = atom_config or AtomConfig()
        self.backend = TreeBackend(config=storage_config)

        self.bucket_size = self.backend.bucket_size
        self.block_size = self.backend.block_size
        self.tree_height = self.backend.tree_height
        self.num_levels = self.backend.num_levels
        self.leaf_count = self.backend.leaf_count
        self.bucket_count = self.backend.bucket_count

        self.logical_block_capacity = self.leaf_count * self.bucket_size

        self._rng = random.Random(rng_seed)

        # position_map[logical_id] is the current bucket address if the freshest copy
        # is known to reside on the server. None means the freshest copy is currently
        # in stash (or the block does not exist yet).
        self.position_map: list[Optional[BucketAddress]] = [None] * self.logical_block_capacity

        # Fresh logical state.
        self.stash: dict[int, DataBlock] = {}

        # Buckets whose server contents are logically stale because they have already
        # been pulled into stash and not yet rewritten with fresh contents.
        self.invalidated_buckets: set[tuple[int, int]] = set()

        # Future OfflineFlushEdge(parent(u), u) work items.
        self.pending_maintenance: list[PendingMaintenance] = []

    def reset(self) -> None:
        self.backend.reset()
        self.position_map = [None] * self.logical_block_capacity
        self.stash.clear()
        self.invalidated_buckets.clear()
        self.pending_maintenance.clear()

    def access(self, request: Request) -> AccessResult:
        metrics = AccessMetrics(protocol=ProtocolKind.ATOM.value)
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

        if request.kind == RequestKind.REAL:
            if request.address is None:
                raise ValueError("Real AtomORAM request requires request.address.")
            logical_id = request.address.logical_id
            self._validate_logical_id(logical_id)
            target_bucket = self._resolve_real_target_bucket(logical_id)
        else:
            logical_id = None
            target_bucket = self._sample_uniform_bucket_address()

        self._online_access_bucket(target_bucket=target_bucket, metrics=metrics)

        result_data: Optional[bytes] = None

        if request.kind == RequestKind.REAL:
            assert logical_id is not None

            if request.op == OperationType.READ:
                block = self.stash.get(logical_id)
                if block is not None:
                    block.leaf = self._sample_leaf()
                    result_data = _truncate_payload(block)
                else:
                    result_data = None

            elif request.op == OperationType.WRITE:
                if request.data is None:
                    raise ValueError("WRITE request must carry request.data.")
                if len(request.data) > self.block_size:
                    raise ValueError(
                        f"Write payload exceeds block size: "
                        f"{len(request.data)} > {self.block_size}"
                    )

                new_leaf = self._sample_leaf()
                self.stash[logical_id] = DataBlock(
                    block_id=logical_id,
                    payload=bytes(request.data),
                    is_dummy=False,
                    leaf=new_leaf,
                    metadata={"logical_payload_size": len(request.data)},
                )
                self.position_map[logical_id] = None
                result_data = bytes(request.data)

            else:
                raise ValueError(f"Unsupported operation: {request.op}")

        self._enqueue_pending_maintenance(
            target_bucket=target_bucket,
            request_id=request.request_id,
            is_real=(request.kind == RequestKind.REAL),
        )

        maintenance_executed = self._run_one_pending_maintenance(metrics=metrics)

        metrics.online_rtt = 1
        metrics.path_length_touched = 1
        metrics.stash_size_after = len(self.stash)

        if request.arrival_time is not None:
            timing.response_time = request.arrival_time
            timing.finalize()

        result = AccessResult(data=result_data, metrics=metrics, timing=timing)
        result.debug.current_bucket = (target_bucket.level, target_bucket.index)
        result.debug.note = (
            f"full AtomORAM logical access; "
            f"pending maintenance queue size={len(self.pending_maintenance)}; "
            f"offline_flush_executed={maintenance_executed}"
        )
        return result

    def tick(self, now: float) -> Optional[AccessResult]:
        # Not implemented in the current phase.
        return None

    @property
    def pending_flush_count(self) -> int:
        return len(self.pending_maintenance)

    def _validate_logical_id(self, logical_id: int) -> None:
        if logical_id < 0 or logical_id >= self.logical_block_capacity:
            raise ValueError(
                f"logical_id={logical_id} is outside the supported range "
                f"[0, {self.logical_block_capacity - 1}]."
            )

    def _sample_leaf(self) -> int:
        return self._rng.randrange(self.leaf_count)

    def _sample_uniform_bucket_address(self) -> BucketAddress:
        level = self._rng.randrange(self.num_levels)
        index = self._rng.randrange(1 << level)
        return BucketAddress(level=level, index=index)

    def _resolve_real_target_bucket(self, logical_id: int) -> BucketAddress:
        current_bucket = self.position_map[logical_id]
        if current_bucket is not None:
            return current_bucket
        return self._sample_uniform_bucket_address()

    def _online_access_bucket(
        self,
        *,
        target_bucket: BucketAddress,
        metrics: AccessMetrics,
    ) -> None:
        bucket = self.backend.read_bucket(target_bucket)

        metrics.record_bucket_read(
            online=True,
            byte_count=self.backend.bucket_storage_bytes,
            rtt_count=1,
            dummy_blocks=bucket.dummy_count(),
        )

        bucket_key = self._bucket_key(target_bucket)

        # If this bucket has already been logically pulled into the stash and not yet
        # written back, its server content is stale and must not be re-imported.
        if bucket_key in self.invalidated_buckets:
            return

        for block in bucket.non_dummy_blocks():
            if block.block_id is None:
                continue
            self.stash[block.block_id] = block.clone()
            self.position_map[block.block_id] = None

        self.invalidated_buckets.add(bucket_key)

    def _enqueue_pending_maintenance(
        self,
        *,
        target_bucket: BucketAddress,
        request_id: int,
        is_real: bool,
    ) -> None:
        parent_bucket = self.backend.parent_address(target_bucket)

        # Root is intentionally special-cased. There is no upward pairing for root.
        if parent_bucket is None:
            return

        self.pending_maintenance.append(
            PendingMaintenance(
                target_bucket=target_bucket,
                parent_bucket=parent_bucket,
                request_id=request_id,
                is_real=is_real,
            )
        )

    def _run_one_pending_maintenance(self, *, metrics: AccessMetrics) -> bool:
        if not self.pending_maintenance:
            return False

        item = self.pending_maintenance.pop(0)
        self._offline_flush_edge(
            parent_bucket=item.parent_bucket,
            target_bucket=item.target_bucket,
            metrics=metrics,
        )
        return True

    def _offline_flush_edge(
        self,
        *,
        parent_bucket: BucketAddress,
        target_bucket: BucketAddress,
        metrics: AccessMetrics,
    ) -> None:
        parent_key = self._bucket_key(parent_bucket)
        target_key = self._bucket_key(target_bucket)

        # Read parent bucket as the offline maintenance input.
        parent_contents = self.backend.read_bucket(parent_bucket)
        metrics.record_bucket_read(
            online=False,
            byte_count=self.backend.bucket_storage_bytes,
            rtt_count=1,
            dummy_blocks=parent_contents.dummy_count(),
        )

        # If the parent is still fresh on server, import it into stash first.
        if parent_key not in self.invalidated_buckets:
            for block in parent_contents.non_dummy_blocks():
                if block.block_id is None:
                    continue
                self.stash[block.block_id] = block.clone()
                self.position_map[block.block_id] = None

        # From this point until writeback completes, both buckets are treated as stale.
        self.invalidated_buckets.add(parent_key)
        self.invalidated_buckets.add(target_key)

        # Deeper-first local greedy writeback.
        target_blocks = self._pop_eligible_blocks_for_bucket(target_bucket)
        parent_blocks = self._pop_eligible_blocks_for_bucket(parent_bucket)

        self._write_bucket_from_blocks(
            address=target_bucket,
            blocks=target_blocks,
            metrics=metrics,
            rtt_count=1,
        )
        self._write_bucket_from_blocks(
            address=parent_bucket,
            blocks=parent_blocks,
            metrics=metrics,
            rtt_count=0,
        )

        # After fresh writeback, these buckets are no longer invalidated.
        self.invalidated_buckets.discard(target_key)
        self.invalidated_buckets.discard(parent_key)

    def _pop_eligible_blocks_for_bucket(self, address: BucketAddress) -> list[DataBlock]:
        eligible_ids = []
        for block_id, block in self.stash.items():
            if block.leaf is None:
                continue
            if self.backend.is_bucket_on_path(address, block.leaf):
                eligible_ids.append(block_id)

        eligible_ids.sort()
        selected_ids = eligible_ids[: self.bucket_size]

        selected_blocks: list[DataBlock] = []
        for block_id in selected_ids:
            block = self.stash.pop(block_id)
            self.position_map[block_id] = address
            selected_blocks.append(block.clone())

        return selected_blocks

    def _write_bucket_from_blocks(
        self,
        *,
        address: BucketAddress,
        blocks: list[DataBlock],
        metrics: AccessMetrics,
        rtt_count: int,
    ) -> None:
        bucket = Bucket(address=address, blocks=[block.clone() for block in blocks])
        normalized_bucket = self.backend.normalize_bucket(bucket)
        self.backend.write_bucket(normalized_bucket)

        metrics.record_bucket_write(
            online=False,
            byte_count=self.backend.bucket_storage_bytes,
            rtt_count=rtt_count,
            dummy_blocks=normalized_bucket.dummy_count(),
        )

    @staticmethod
    def _bucket_key(address: BucketAddress) -> tuple[int, int]:
        return (address.level, address.index)

    # ---------- debug / test helper ----------
    def debug_seed_bucket(
        self,
        *,
        bucket_address: BucketAddress,
        blocks: list[DataBlock],
    ) -> None:
        """
        Helper for unit tests only.
        It writes the supplied blocks into a bucket and updates position_map
        so that each non-dummy block is considered server-resident at bucket_address.
        """
        bucket = Bucket(address=bucket_address, blocks=[block.clone() for block in blocks])
        self.backend.write_bucket(bucket)

        for block in blocks:
            if block.is_dummy or block.block_id is None:
                continue
            self._validate_logical_id(block.block_id)
            self.position_map[block.block_id] = bucket_address
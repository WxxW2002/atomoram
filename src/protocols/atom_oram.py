from __future__ import annotations

import random
from typing import Optional

from src.backend.tree_backend import TreeBackend
from src.common.config import AtomConfig, StorageConfig
from src.common.interfaces import AbstractORAM
from src.common.metrics import AccessMetrics, AccessResult, TimingRecord
from src.common.utils import truncate_payload
from src.common.types import (
    Bucket,
    BucketAddress,
    DataBlock,
    OperationType,
    ProtocolKind,
    Request,
    RequestKind,
    BlockAddress,
)


class AtomORAM(AbstractORAM):
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

        self.position_map: list[Optional[BucketAddress]] = [None] * self.logical_block_capacity

        self.stash: dict[int, DataBlock] = {}
        self.invalidated_buckets: set[tuple[int, int]] = set()

        self.current_epoch_leaf: Optional[int] = None
        self.current_epoch_step: int = 0

        self.carried_epoch_bucket: Optional[BucketAddress] = None

    def reset(self) -> None:
        self.backend.reset()
        self.position_map = [None] * self.logical_block_capacity
        self.stash.clear()
        self.invalidated_buckets.clear()
        self.current_epoch_leaf = None
        self.current_epoch_step = 0
        self.carried_epoch_bucket = None

    def _local_cutoff_level(self) -> int:
        if self.atom_config.local_cutoff_level is not None:
            return self.atom_config.local_cutoff_level
        return self.backend.tree_height // 2

    def _is_local_bucket(self, address: BucketAddress) -> bool:
        if not self.atom_config.local_top_half_enabled:
            return False
        return address.level < self._local_cutoff_level()

    def _record_atom_bucket_read(
        self,
        metrics: AccessMetrics,
        address: BucketAddress,
        *,
        online: bool,
    ) -> None:
        if self._is_local_bucket(address):
            metrics.record_bucket_read(
                online=online,
                byte_count=0,
                rtt_count=0,
                dummy_blocks=0,
            )
        else:
            metrics.record_bucket_read(
                online=online,
                byte_count=self.backend.bucket_storage_bytes,
                rtt_count=1,
                dummy_blocks=0,
            )

    def _record_atom_bucket_write(
        self,
        metrics: AccessMetrics,
        address: BucketAddress,
        *,
        online: bool,
    ) -> None:
        if self._is_local_bucket(address):
            metrics.record_bucket_write(
                online=online,
                byte_count=0,
                rtt_count=0,
                dummy_blocks=0,
            )
        else:
            metrics.record_bucket_write(
                online=online,
                byte_count=self.backend.bucket_storage_bytes,
                rtt_count=0,
                dummy_blocks=0,
            )

    @staticmethod
    def _update_stash_peak(metrics: AccessMetrics, stash_size: int) -> None:
        current_peak = getattr(metrics, "stash_peak_during_access", stash_size)
        if stash_size > current_peak:
            metrics.stash_peak_during_access = stash_size

    def access(self, request: Request) -> AccessResult:
        metrics = AccessMetrics(protocol=ProtocolKind.ATOM.value)
        timing = TimingRecord(
            arrival_time=request.arrival_time,
            service_start_time=request.arrival_time,
        )

        metrics.stash_size_before = len(self.stash)
        metrics.stash_peak_during_access = len(self.stash)
        metrics.queue_length_before = 0
        metrics.queue_length_after = 0

        if request.kind == RequestKind.REAL:
            metrics.real_requests_served = 1
        else:
            metrics.virtual_requests_executed = 1

        if request.kind == RequestKind.REAL:
            if not isinstance(request.address, BlockAddress):
                raise ValueError("Real AtomORAM request requires a BlockAddress.")
            logical_id = request.address.logical_id
            self._validate_logical_id(logical_id)
            target_bucket = self._resolve_real_target_bucket(logical_id)
        else:
            logical_id = None
            if request.address is None:
                target_bucket = self._sample_uniform_bucket_address()
            elif isinstance(request.address, BucketAddress):
                self._validate_bucket_address(request.address)
                target_bucket = request.address
            else:
                raise ValueError(
                    "Virtual AtomORAM request address must be None or BucketAddress."
                )

        target_bucket_image = self._read_bucket_raw(
            address=target_bucket,
            metrics=metrics,
            online=True,
        )

        result_data: Optional[bytes] = None
        target_bucket_resident_blocks: list[DataBlock] = []

        if request.kind == RequestKind.REAL:
            assert logical_id is not None

            working_block, target_bucket_resident_blocks = self._split_real_target_bucket(
                bucket=target_bucket_image,
                bucket_address=target_bucket,
                logical_id=logical_id,
            )

            if request.op == OperationType.READ:
                if working_block is not None:
                    working_block.leaf = self._sample_leaf()
                    self.stash[logical_id] = working_block
                    self._update_stash_peak(metrics, len(self.stash))
                    self.position_map[logical_id] = None
                    result_data = truncate_payload(working_block)
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

                if working_block is None:
                    working_block = DataBlock(
                        block_id=logical_id,
                        payload=bytes(request.data),
                        is_dummy=False,
                        leaf=self._sample_leaf(),
                        metadata={"logical_payload_size": len(request.data)},
                    )
                else:
                    working_block.payload = bytes(request.data)
                    working_block.leaf = self._sample_leaf()
                    working_block.metadata["logical_payload_size"] = len(request.data)

                self.stash[logical_id] = working_block
                self._update_stash_peak(metrics, len(self.stash))
                self.position_map[logical_id] = None
                result_data = bytes(request.data)

            else:
                raise ValueError(f"Unsupported operation: {request.op}")

        else:
            target_bucket_resident_blocks = self._split_virtual_target_bucket(
                bucket=target_bucket_image,
                bucket_address=target_bucket,
            )

        protected_block_ids: set[int] = set()
        if request.kind == RequestKind.REAL and logical_id is not None:
            protected_block_ids.add(logical_id)

        maintenance_executed = self._run_one_epoch_micro_eviction(
            target_bucket=target_bucket,
            target_bucket_resident_blocks=target_bucket_resident_blocks,
            metrics=metrics,
            protected_block_ids=protected_block_ids,
        )

        metrics.path_length_touched = 1
        metrics.stash_size_after = len(self.stash)

        if request.arrival_time is not None:
            timing.response_time = request.arrival_time
            timing.finalize()

        result = AccessResult(data=result_data, metrics=metrics, timing=timing)
        result.debug.current_bucket = (target_bucket.level, target_bucket.index)
        result.debug.note = (
            "full AtomORAM logical access; "
            f"epoch_leaf={self.current_epoch_leaf}; "
            f"epoch_step={self.current_epoch_step}; "
            f"epoch_remaining={self.pending_flush_count}; "
            f"carried_epoch_bucket={self.carried_epoch_bucket}; "
            f"offline_micro_eviction_executed={maintenance_executed}"
        )
        return result

    def tick(self, now: float) -> Optional[AccessResult]:
        return None

    @property
    def pending_flush_count(self) -> int:
        if self.current_epoch_leaf is None:
            return 0
        return max(self.tree_height - self.current_epoch_step, 0)

    def _validate_logical_id(self, logical_id: int) -> None:
        if logical_id < 0 or logical_id >= self.logical_block_capacity:
            raise ValueError(
                f"logical_id={logical_id} is outside the supported range "
                f"[0, {self.logical_block_capacity - 1}]."
            )

    def _sample_leaf(self) -> int:
        return self._rng.randrange(self.leaf_count)

    def sample_virtual_bucket_address(self) -> BucketAddress:
        return self._sample_uniform_bucket_address()

    def _sample_uniform_bucket_address(self) -> BucketAddress:
        level = self._rng.randrange(self.num_levels)
        index = self._rng.randrange(1 << level)
        return BucketAddress(level=level, index=index)

    def _validate_bucket_address(self, address: BucketAddress) -> None:
        if address.level < 0 or address.level >= self.num_levels:
            raise ValueError(
                f"Bucket level {address.level} is outside [0, {self.num_levels - 1}]."
            )
        if address.index < 0 or address.index >= (1 << address.level):
            raise ValueError(
                f"Bucket index {address.index} is outside level {address.level}."
            )

    def _resolve_real_target_bucket(self, logical_id: int) -> BucketAddress:
        current_bucket = self.position_map[logical_id]
        if current_bucket is not None:
            return current_bucket
        return self._sample_uniform_bucket_address()

    def _read_bucket_raw(
        self,
        *,
        address: BucketAddress,
        metrics: AccessMetrics,
        online: bool,
    ) -> Bucket:
        bucket = self.backend.read_bucket(address)
        self._record_atom_bucket_read(metrics, address, online=online)

        if self._bucket_key(address) in self.invalidated_buckets:
            return self.backend.make_empty_bucket(address)

        return bucket

    def _read_bucket_into_stash(
        self,
        *,
        address: BucketAddress,
        metrics: AccessMetrics,
        online: bool,
    ) -> None:
        bucket = self.backend.read_bucket(address)
        self._record_atom_bucket_read(metrics, address, online=online)

        bucket_key = self._bucket_key(address)
        if bucket_key in self.invalidated_buckets:
            return

        for block in bucket.non_dummy_blocks():
            if block.block_id is None:
                continue
            self.stash[block.block_id] = block.clone()
            self.position_map[block.block_id] = None

        self._update_stash_peak(metrics, len(self.stash))
        self.invalidated_buckets.add(bucket_key)

    def _split_real_target_bucket(
        self,
        *,
        bucket: Bucket,
        bucket_address: BucketAddress,
        logical_id: int,
    ) -> tuple[Optional[DataBlock], list[DataBlock]]:
        # If the target block is already in the stash, keep using that version.
        target_block = self.stash.pop(logical_id, None)
        resident_blocks: list[DataBlock] = []

        for block in bucket.non_dummy_blocks():
            if block.block_id is None:
                continue

            if block.block_id == logical_id:
                if target_block is None:
                    target_block = block.clone()
                self.position_map[logical_id] = None
                continue

            resident_blocks.append(block.clone())
            self.position_map[block.block_id] = bucket_address

        return target_block, resident_blocks

    def _split_virtual_target_bucket(
        self,
        *,
        bucket: Bucket,
        bucket_address: BucketAddress,
    ) -> list[DataBlock]:
        resident_blocks: list[DataBlock] = []

        for block in bucket.non_dummy_blocks():
            if block.block_id is None:
                continue
            resident_blocks.append(block.clone())
            self.position_map[block.block_id] = bucket_address

        return resident_blocks

    def _stage_resident_blocks_into_stash(
        self,
        *,
        blocks: list[DataBlock],
        metrics: AccessMetrics,
    ) -> None:
        for block in blocks:
            if block.block_id is None or block.is_dummy:
                continue
            self.stash[block.block_id] = block.clone()
            self.position_map[block.block_id] = None

        self._update_stash_peak(metrics, len(self.stash))

    def _ensure_epoch(self) -> None:
        if self.current_epoch_leaf is None or self.current_epoch_step >= self.tree_height:
            self.current_epoch_leaf = self._sample_leaf()
            self.current_epoch_step = 0
            self.carried_epoch_bucket = None

    def _bucket_address_on_leaf(self, leaf: int, level: int) -> BucketAddress:
        index = leaf >> (self.tree_height - level)
        return BucketAddress(level=level, index=index)

    def _current_epoch_pair(self) -> tuple[BucketAddress, BucketAddress]:
        self._ensure_epoch()
        assert self.current_epoch_leaf is not None

        lower_level = self.tree_height - self.current_epoch_step
        upper_level = lower_level - 1

        upper = self._bucket_address_on_leaf(self.current_epoch_leaf, upper_level)
        lower = self._bucket_address_on_leaf(self.current_epoch_leaf, lower_level)
        return upper, lower

    def _run_one_epoch_micro_eviction(
        self,
        *,
        target_bucket: BucketAddress,
        target_bucket_resident_blocks: list[DataBlock],
        metrics: AccessMetrics,
        protected_block_ids: Optional[set[int]] = None,
    ) -> bool:
        if self.tree_height == 0:
            self._write_bucket_direct(
                address=target_bucket,
                blocks=target_bucket_resident_blocks,
                metrics=metrics,
            )
            return False

        self._ensure_epoch()
        assert self.current_epoch_leaf is not None

        step = self.current_epoch_step
        lower_level = self.tree_height - step
        upper_level = lower_level - 1

        lower_bucket = self._bucket_address_on_leaf(self.current_epoch_leaf, lower_level)
        upper_bucket = self._bucket_address_on_leaf(self.current_epoch_leaf, upper_level)

        target_key = self._bucket_key(target_bucket)
        lower_key = self._bucket_key(lower_bucket)
        upper_key = self._bucket_key(upper_bucket)

        is_first_step = step == 0
        is_last_step = step == self.tree_height - 1

        if is_first_step:
            self.carried_epoch_bucket = None
        else:
            if self.carried_epoch_bucket != lower_bucket:
                raise RuntimeError(
                    "Invalid AtomORAM epoch pipeline state: "
                    f"expected carried bucket {lower_bucket}, "
                    f"found {self.carried_epoch_bucket}."
                )

        # First step reads both L and L-1.
        if is_first_step:
            buckets_to_read = [lower_bucket, upper_bucket]
        else:
            buckets_to_read = [upper_bucket]

        # Non-final steps write only the lower/carried bucket and keep upper.
        if is_last_step:
            buckets_to_write = [lower_bucket, upper_bucket]
        else:
            buckets_to_write = [lower_bucket]

        edge_keys = {lower_key, upper_key}
        target_is_epoch_bucket = target_key in edge_keys

        if target_is_epoch_bucket:
            self._stage_resident_blocks_into_stash(
                blocks=target_bucket_resident_blocks,
                metrics=metrics,
            )
            target_bucket_resident_blocks = []
            self.invalidated_buckets.add(target_key)

        for bucket in buckets_to_read:
            bucket_key = self._bucket_key(bucket)
            if bucket_key == target_key:
                self.invalidated_buckets.add(bucket_key)
                continue

            self._read_bucket_into_stash(
                address=bucket,
                metrics=metrics,
                online=False,
            )

        for bucket in buckets_to_write:
            bucket_key = self._bucket_key(bucket)

            exclude_ids = None
            if bucket_key == target_key and protected_block_ids:
                exclude_ids = protected_block_ids

            self._write_bucket_from_stash(
                address=bucket,
                metrics=metrics,
                exclude_block_ids=exclude_ids,
            )

        if not target_is_epoch_bucket:
            self._write_bucket_direct(
                address=target_bucket,
                blocks=target_bucket_resident_blocks,
                metrics=metrics,
            )

        if is_last_step:
            self.current_epoch_leaf = None
            self.current_epoch_step = 0
            self.carried_epoch_bucket = None
        else:
            self.current_epoch_step += 1
            self.carried_epoch_bucket = upper_bucket

        return True

    def _pop_eligible_blocks_for_bucket(
        self,
        address: BucketAddress,
        *,
        exclude_block_ids: Optional[set[int]] = None,
    ) -> list[DataBlock]:
        eligible_ids = []
        blocked_ids = exclude_block_ids or set()

        for block_id, block in self.stash.items():
            if block_id in blocked_ids:
                continue
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

    def _write_bucket_from_stash(
        self,
        *,
        address: BucketAddress,
        metrics: AccessMetrics,
        exclude_block_ids: Optional[set[int]],
    ) -> None:
        bucket_key = self._bucket_key(address)
        blocks = self._pop_eligible_blocks_for_bucket(
            address,
            exclude_block_ids=exclude_block_ids,
        )

        bucket = Bucket(address=address, blocks=[block.clone() for block in blocks])
        normalized_bucket = self.backend.normalize_bucket(bucket)
        self.backend.write_bucket(normalized_bucket)
        self._record_atom_bucket_write(metrics, address, online=False)
        self.invalidated_buckets.discard(bucket_key)

    def _write_bucket_direct(
        self,
        *,
        address: BucketAddress,
        blocks: list[DataBlock],
        metrics: AccessMetrics,
    ) -> None:
        for block in blocks:
            if block.block_id is None or block.is_dummy:
                continue
            self.position_map[block.block_id] = address

        bucket = Bucket(address=address, blocks=[block.clone() for block in blocks])
        normalized_bucket = self.backend.normalize_bucket(bucket)
        self.backend.write_bucket(normalized_bucket)
        self._record_atom_bucket_write(metrics, address, online=False)
        self.invalidated_buckets.discard(self._bucket_key(address))

    @staticmethod
    def _bucket_key(address: BucketAddress) -> tuple[int, int]:
        return (address.level, address.index)

    def debug_seed_bucket(
        self,
        *,
        bucket_address: BucketAddress,
        blocks: list[DataBlock],
    ) -> None:
        bucket = Bucket(address=bucket_address, blocks=[block.clone() for block in blocks])
        self.backend.write_bucket(bucket)

        for block in blocks:
            if block.is_dummy or block.block_id is None:
                continue
            self._validate_logical_id(block.block_id)
            self.position_map[block.block_id] = bucket_address
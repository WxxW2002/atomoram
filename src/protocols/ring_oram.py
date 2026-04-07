from __future__ import annotations

import random
from dataclasses import replace
from typing import Optional

from src.backend.tree_backend import TreeBackend
from src.common.config import RingConfig, StorageConfig
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


class RingORAM(AbstractORAM):
    """
    Ring ORAM adapted from the V-ORAM reference implementation.

    Important conventions:
    - storage_config.bucket_size is interpreted as the number of real slots Z.
    - The backend bucket capacity becomes Z + S, where S is the number of dummy slots.
    - Communication accounting follows Ring ORAM semantics:
        * read_ring_path() communicates only one block (server-side XOR abstraction)
        * reshuffle / eviction communicate full buckets
    - Server-I/O accounting explicitly counts actual bucket reads/writes.
    """

    SLOT_EMPTY_REAL = -1
    SLOT_DUMMY_CONSUMED = 0
    SLOT_DUMMY_AVAILABLE = 1

    def __init__(
        self,
        storage_config: StorageConfig,
        *,
        ring_config: Optional[RingConfig] = None,
        rng_seed: Optional[int] = None,
    ) -> None:
        self.storage_config = storage_config
        self.ring_config = ring_config or RingConfig()

        if self.storage_config.bucket_size <= 0:
            raise ValueError("storage_config.bucket_size must be positive.")
        if self.ring_config.s_num <= 0:
            raise ValueError("ring_config.s_num must be positive.")
        if self.ring_config.a_num <= 0:
            raise ValueError("ring_config.a_num must be positive.")

        self.real_bucket_size = self.storage_config.bucket_size
        self.s_num = self.ring_config.s_num
        self.a_num = self.ring_config.a_num
        self.total_bucket_slots = self.real_bucket_size + self.s_num

        backend_config = replace(
            self.storage_config,
            bucket_size=self.total_bucket_slots,
        )
        self.backend = TreeBackend(config=backend_config)

        self.leaf_count = self.backend.leaf_count
        self.num_levels = self.backend.num_levels
        self.bucket_count = self.backend.bucket_count

        self.logical_block_capacity = self.leaf_count * self.real_bucket_size

        self._rng = random.Random(rng_seed)

        self.stash: dict[int, DataBlock] = {}
        self.position_map: list[Optional[int]] = [None] * self.logical_block_capacity

        # Per bucket slot state:
        #   real slots [0:Z): -1 means empty, otherwise logical block id
        #   dummy slots [Z:Z+S): 1 means available dummy, 0 means already consumed
        self.address_map: list[list[int]] = []
        for _ in range(self.bucket_count):
            row = [self.SLOT_EMPTY_REAL] * self.real_bucket_size
            row.extend([self.SLOT_DUMMY_AVAILABLE] * self.s_num)
            self.address_map.append(row)

        self.round = 0
        self.big_g = 0
        self.count: list[int] = [0] * self.bucket_count

    def reset(self) -> None:
        self.backend.reset()
        self.stash.clear()
        self.position_map = [None] * self.logical_block_capacity

        self.address_map = []
        for _ in range(self.bucket_count):
            row = [self.SLOT_EMPTY_REAL] * self.real_bucket_size
            row.extend([self.SLOT_DUMMY_AVAILABLE] * self.s_num)
            self.address_map.append(row)

        self.round = 0
        self.big_g = 0
        self.count = [0] * self.bucket_count

    def access(self, request: Request) -> AccessResult:
        if request.address is None:
            raise ValueError("RingORAM.access() requires request.address.")

        logical_id = request.address.logical_id
        self._validate_logical_id(logical_id)

        metrics = AccessMetrics(protocol=ProtocolKind.RING.value)
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
        first_access = current_leaf is None
        if current_leaf is None:
            current_leaf = self._sample_leaf()

        new_leaf = self._sample_leaf()
        self.position_map[logical_id] = new_leaf

        block = self._read_ring_path(leaf=current_leaf, logical_id=logical_id, metrics=metrics)

        if block is None and not first_access:
            block = self.stash.get(logical_id)

        if request.op == OperationType.WRITE:
            if request.data is None:
                raise ValueError("WRITE request must carry request.data.")
            if len(request.data) > self.storage_config.block_size:
                raise ValueError(
                    f"Write payload exceeds block size: "
                    f"{len(request.data)} > {self.storage_config.block_size}"
                )

            block = DataBlock(
                block_id=logical_id,
                payload=bytes(request.data),
                is_dummy=False,
                leaf=new_leaf,
                metadata={"logical_payload_size": len(request.data)},
            )
            self.stash[logical_id] = block
            result_data = bytes(request.data)

        elif request.op == OperationType.READ:
            if block is not None:
                block = block.clone()
                block.leaf = new_leaf
                self.stash[logical_id] = block
                result_data = _truncate_payload(block)
            else:
                result_data = None

        else:
            raise ValueError(f"Unsupported operation: {request.op}")

        if request.op == OperationType.WRITE and logical_id in self.stash:
            self.stash[logical_id].leaf = new_leaf

        self.round = (self.round + 1) % self.a_num

        did_periodic_eviction = False
        if self.round == 0:
            self._evict_path(metrics=metrics)
            did_periodic_eviction = True
            metrics.eviction_count += 1

        did_reshuffle = self._early_reshuffle(leaf=current_leaf, metrics=metrics)

        # RTT accounting follows the V-ORAM Ring_ORAM implementation:
        #   +1 RTT for read_ring_path
        #   +1 RTT if either periodic eviction or any early reshuffle happened
        metrics.online_rtt += 1
        if did_periodic_eviction or did_reshuffle:
            metrics.online_rtt += 1

        metrics.stash_size_after = len(self.stash)
        metrics.path_length_touched = self.num_levels

        if request.arrival_time is not None:
            timing.response_time = request.arrival_time
            timing.finalize()

        result = AccessResult(data=result_data, metrics=metrics, timing=timing)
        result.debug.current_leaf = current_leaf
        result.debug.note = (
            f"ring access on logical block {logical_id}, "
            f"reassigned to leaf {new_leaf}"
        )
        return result

    def _validate_logical_id(self, logical_id: int) -> None:
        if logical_id < 0 or logical_id >= self.logical_block_capacity:
            raise ValueError(
                f"logical_id={logical_id} is outside the supported range "
                f"[0, {self.logical_block_capacity - 1}]."
            )

    def _sample_leaf(self) -> int:
        return self._rng.randrange(self.leaf_count)

    def _read_ring_path(
        self,
        *,
        leaf: int,
        logical_id: int,
        metrics: AccessMetrics,
    ) -> Optional[DataBlock]:
        path = self.backend.path_to_leaf(leaf)
        found_block: Optional[DataBlock] = None

        for address in path:
            flat_index = self.backend.flatten_address(address)
            slots = self.address_map[flat_index]

            # Actual bucket touch happens, but communication cost is not a full-bucket download.
            _ = self.backend.read_bucket(address)
            metrics.record_bucket_read(
                online=True,
                byte_count=0,
                rtt_count=0,
                dummy_blocks=0,
            )

            found = False
            for j in range(self.real_bucket_size):
                if slots[j] == logical_id:
                    bucket = self.backend.read_bucket(address)
                    found_block = bucket.blocks[j].clone()
                    slots[j] = self.SLOT_EMPTY_REAL
                    found = True
                    break

            if not found:
                dummy_found = False
                for j in range(self.real_bucket_size, self.total_bucket_slots):
                    if slots[j] == self.SLOT_DUMMY_AVAILABLE:
                        slots[j] = self.SLOT_DUMMY_CONSUMED
                        dummy_found = True
                        break
                if not dummy_found:
                    raise RuntimeError(
                        "Ring ORAM overflow: no valid dummy block is available."
                    )

            self.count[flat_index] += 1

        # Communication-wise, Ring ORAM returns only one block for the whole path.
        metrics.online_bytes_down += self.storage_config.block_size
        return found_block

    def _evict_path(self, *, metrics: AccessMetrics) -> None:
        leaf = self._g_to_l(self.big_g)
        self.big_g += 1

        path = self.backend.path_to_leaf(leaf)

        # Read all buckets on the eviction path into stash.
        for address in path:
            flat_index = self.backend.flatten_address(address)
            slots = self.address_map[flat_index]
            bucket = self.backend.read_bucket(address)

            metrics.record_bucket_read(
                online=True,
                byte_count=self.backend.bucket_storage_bytes,
                rtt_count=0,
                dummy_blocks=0,
            )

            for j in range(self.real_bucket_size):
                block_id = slots[j]
                if block_id != self.SLOT_EMPTY_REAL:
                    self.stash[block_id] = bucket.blocks[j].clone()
                    slots[j] = self.SLOT_EMPTY_REAL

        for address in reversed(path):
            self._write_bucket(address=address, metrics=metrics)
            flat_index = self.backend.flatten_address(address)
            self.count[flat_index] = 0

    def _early_reshuffle(self, *, leaf: int, metrics: AccessMetrics) -> bool:
        did_reshuffle = False
        path = self.backend.path_to_leaf(leaf)

        for address in path:
            flat_index = self.backend.flatten_address(address)
            if self.count[flat_index] < self.s_num:
                continue

            did_reshuffle = True
            bucket = self.backend.read_bucket(address)
            metrics.record_bucket_read(
                online=True,
                byte_count=self.backend.bucket_storage_bytes,
                rtt_count=0,
                dummy_blocks=0,
            )

            slots = self.address_map[flat_index]
            for j in range(self.real_bucket_size):
                block_id = slots[j]
                if block_id != self.SLOT_EMPTY_REAL:
                    self.stash[block_id] = bucket.blocks[j].clone()
                    slots[j] = self.SLOT_EMPTY_REAL

            self._write_bucket(address=address, metrics=metrics)
            self.count[flat_index] = 0
            metrics.reshuffle_count += 1

        return did_reshuffle

    def _write_bucket(self, *, address, metrics: AccessMetrics) -> None:
        flat_index = self.backend.flatten_address(address)
        eligible_ids = [
            block_id
            for block_id, block in self.stash.items()
            if block.leaf is not None and self.backend.is_bucket_on_path(address, block.leaf)
        ]

        if eligible_ids:
            chosen_count = min(len(eligible_ids), self.real_bucket_size)
            if chosen_count == len(eligible_ids):
                selected_ids = list(eligible_ids)
            else:
                selected_ids = self._rng.sample(eligible_ids, chosen_count)
        else:
            selected_ids = []

        bucket = self.backend.make_empty_bucket(address)

        for j, block_id in enumerate(selected_ids):
            block = self.stash.pop(block_id)
            bucket.blocks[j] = block.clone()

        # Real slots not used remain dummy blocks, with SLOT_EMPTY_REAL in address_map.
        slots = self.address_map[flat_index]
        for j in range(self.real_bucket_size):
            if j < len(selected_ids):
                slots[j] = selected_ids[j]
            else:
                slots[j] = self.SLOT_EMPTY_REAL

        # Dummy region is refreshed to "available dummies".
        for j in range(self.real_bucket_size, self.total_bucket_slots):
            slots[j] = self.SLOT_DUMMY_AVAILABLE

        self.backend.write_bucket(bucket)

        metrics.record_bucket_write(
            online=True,
            byte_count=self.backend.bucket_storage_bytes,
            rtt_count=0,
            dummy_blocks=self.total_bucket_slots - len(selected_ids),
        )

    def _g_to_l(self, big_g: int) -> int:
        """
        Same mapping logic as V-ORAM's BTree.g_to_l().
        """
        tmp_l = big_g % self.leaf_count
        leaf = 0
        for _ in range(self.num_levels - 1):
            if tmp_l % 2 == 1:
                leaf = leaf * 2 + 2
            else:
                leaf = leaf * 2 + 1
            tmp_l //= 2
        leaf = leaf - self.leaf_count + 1
        return leaf
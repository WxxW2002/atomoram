from __future__ import annotations

from dataclasses import replace
from typing import Optional

from src.backend.memory_store import InMemoryBucketStore
from src.backend.file_store import FileBucketStore
from src.common.config import StorageConfig
from src.common.interfaces import AbstractBucketStore
from src.common.types import Bucket, BucketAddress, DataBlock
from src.common.utils import (
    bucket_address_on_path,
    bucket_count_from_height,
    flatten_bucket_address,
    is_bucket_on_leaf_path,
    leaf_count_from_height,
    path_to_leaf,
    unflatten_bucket_index,
)


class TreeBackend:
    """
    Logical binary tree with flattened-array physical addressing.

    Levels are numbered from 0 (root) to tree_height (leaf level).
    A bucket at (level, index) is stored at flat index:
        flat = 2^level - 1 + index
    """

    DEFAULT_BLOCK_METADATA_BYTES = 32

    def __init__(
        self,
        config: StorageConfig,
        *,
        bucket_store: Optional[AbstractBucketStore] = None,
        block_metadata_bytes: int = DEFAULT_BLOCK_METADATA_BYTES,
    ) -> None:
        if config.block_size <= 0:
            raise ValueError("block_size must be positive.")
        if config.bucket_size <= 0:
            raise ValueError("bucket_size must be positive.")
        if config.tree_height < 0:
            raise ValueError("tree_height must be non-negative.")
        if block_metadata_bytes < 0:
            raise ValueError("block_metadata_bytes must be non-negative.")

        self.config = config
        self.block_size = config.block_size
        self.bucket_size = config.bucket_size
        self.tree_height = config.tree_height
        self.num_levels = self.tree_height + 1
        self.leaf_count = leaf_count_from_height(self.tree_height)
        self.bucket_count = bucket_count_from_height(self.tree_height)
        self.block_metadata_bytes = block_metadata_bytes
        if bucket_store is not None:
            self.bucket_store = bucket_store
        elif config.use_file_backend:
            self.bucket_store = FileBucketStore(
                data_dir=config.data_dir,
                bucket_count=self.bucket_count,
                bucket_size=self.bucket_size,
                block_size=self.block_size,
                block_metadata_bytes=self.block_metadata_bytes,
                data_file_size=config.data_file_size,
            )
        else:
            self.bucket_store = InMemoryBucketStore()

    @property
    def block_storage_bytes(self) -> int:
        return self.block_size + self.block_metadata_bytes

    @property
    def bucket_storage_bytes(self) -> int:
        return self.bucket_size * self.block_storage_bytes

    def reset(self) -> None:
        self.bucket_store.reset()

    def flatten_address(self, address: BucketAddress) -> int:
        self._validate_bucket_address(address)
        return flatten_bucket_address(address)

    def unflatten_index(self, flat_index: int) -> BucketAddress:
        address = unflatten_bucket_index(flat_index)
        self._validate_bucket_address(address)
        return address

    def _validate_bucket_address(self, address: BucketAddress) -> None:
        if address.level < 0 or address.level > self.tree_height:
            raise ValueError(
                f"Bucket level {address.level} is outside [0, {self.tree_height}]."
            )
        if address.index < 0 or address.index >= (1 << address.level):
            raise ValueError(
                f"Bucket index {address.index} is invalid for level {address.level}."
            )

    def _normalize_block(self, block: DataBlock) -> DataBlock:
        copied = block.clone()

        if copied.is_dummy:
            copied.block_id = None
            copied.leaf = None
            copied.payload = bytes(self.block_size)
            return copied

        payload_len = len(copied.payload)
        if payload_len > self.block_size:
            raise ValueError(
                f"Block payload exceeds fixed block size: "
                f"{payload_len} > {self.block_size}"
            )

        if payload_len < self.block_size:
            copied.metadata = dict(copied.metadata)
            copied.metadata.setdefault("logical_payload_size", payload_len)
            copied.payload = copied.payload + bytes(self.block_size - payload_len)

        return copied

    def make_dummy_block(self) -> DataBlock:
        return DataBlock(
            block_id=None,
            payload=bytes(self.block_size),
            is_dummy=True,
            leaf=None,
            metadata={},
        )

    def make_empty_bucket(self, address: BucketAddress) -> Bucket:
        self._validate_bucket_address(address)
        return Bucket(
            address=address,
            blocks=[self.make_dummy_block() for _ in range(self.bucket_size)],
        )

    def normalize_bucket(self, bucket: Bucket) -> Bucket:
        self._validate_bucket_address(bucket.address)

        if len(bucket.blocks) > self.bucket_size:
            raise ValueError(
                f"Bucket at {bucket.address} contains {len(bucket.blocks)} blocks, "
                f"which exceeds bucket_size={self.bucket_size}."
            )

        normalized_blocks = [self._normalize_block(block) for block in bucket.blocks]
        if len(normalized_blocks) < self.bucket_size:
            normalized_blocks.extend(
                self.make_dummy_block()
                for _ in range(self.bucket_size - len(normalized_blocks))
            )

        return Bucket(address=bucket.address, blocks=normalized_blocks)

    def read_bucket(self, address: BucketAddress) -> Bucket:
        self._validate_bucket_address(address)
        flat_index = self.flatten_address(address)

        bucket = self.bucket_store.read(flat_index)
        if bucket is None:
            bucket = self.make_empty_bucket(address)
            self.bucket_store.write(flat_index, bucket)
            return bucket.clone()

        normalized = self.normalize_bucket(bucket)
        if len(bucket.blocks) != len(normalized.blocks):
            self.bucket_store.write(flat_index, normalized)
        return normalized.clone()

    def write_bucket(self, bucket: Bucket) -> None:
        normalized = self.normalize_bucket(bucket)
        flat_index = self.flatten_address(normalized.address)
        self.bucket_store.write(flat_index, normalized)

    # return the bucket address at a given level on a leaf path
    def bucket_address_on_path(self, leaf: int, level: int) -> BucketAddress:

        return bucket_address_on_path(
            tree_height=self.tree_height,
            leaf=leaf,
            level=level,
        )

    def path_to_leaf(self, leaf: int) -> list[BucketAddress]:
        return path_to_leaf(tree_height=self.tree_height, leaf=leaf)

    def is_bucket_on_path(self, address: BucketAddress, leaf: int) -> bool:
        return is_bucket_on_leaf_path(
            tree_height=self.tree_height,
            address=address,
            leaf=leaf,
        )

    def parent_address(self, address: BucketAddress) -> Optional[BucketAddress]:
        self._validate_bucket_address(address)
        if address.level == 0:
            return None
        return BucketAddress(level=address.level - 1, index=address.index // 2)

    def children_addresses(self, address: BucketAddress) -> list[BucketAddress]:
        self._validate_bucket_address(address)
        if address.level == self.tree_height:
            return []

        next_level = address.level + 1
        left = BucketAddress(level=next_level, index=address.index * 2)
        right = BucketAddress(level=next_level, index=address.index * 2 + 1)
        return [left, right]

    def bucket_fill_count(self, address: BucketAddress) -> int:
        bucket = self.read_bucket(address)
        return len(bucket.non_dummy_blocks())
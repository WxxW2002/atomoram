from __future__ import annotations

import math
import shutil
import struct
from pathlib import Path
from typing import Optional

from src.common.interfaces import AbstractBucketStore
from src.common.types import Bucket, DataBlock
from src.common.utils import ensure_dir, unflatten_bucket_index


class FileBucketStore(AbstractBucketStore):
    """
    Fixed-offset file-backed bucket store.

    Layout:
      bucket(flat_index)
        -> file_index = flat_index // buckets_per_file
        -> bucket_slot = flat_index % buckets_per_file
        -> byte_offset = bucket_slot * bucket_storage_bytes

    Each block is serialized as:
      [32-byte header][payload padded to block_size]

    Header format:
      - block_id: int64
      - leaf: int64
      - logical_payload_size: uint32
      - flags: uint8
      - pad: 11 bytes
    """

    HEADER = struct.Struct("<qqIB11x")   # 32 bytes
    FLAG_DUMMY = 0x01

    def __init__(
        self,
        *,
        data_dir: str,
        bucket_count: int,
        bucket_size: int,
        block_size: int,
        block_metadata_bytes: int,
        data_file_size: int,
    ) -> None:
        if block_metadata_bytes < self.HEADER.size:
            raise ValueError(
                f"block_metadata_bytes must be >= {self.HEADER.size} for file backend."
            )
        if data_file_size <= 0:
            raise ValueError("data_file_size must be positive.")

        self.root_dir = Path(data_dir)
        self.tree_dir = ensure_dir(self.root_dir / "tree_data")

        self.bucket_count = bucket_count
        self.bucket_size = bucket_size
        self.block_size = block_size
        self.block_metadata_bytes = block_metadata_bytes
        self.data_file_size = data_file_size

        self.block_storage_bytes = self.block_metadata_bytes + self.block_size
        self.bucket_storage_bytes = self.bucket_size * self.block_storage_bytes

        self.buckets_per_file = max(1, self.data_file_size // self.bucket_storage_bytes)
        self.total_tree_bytes = self.bucket_count * self.bucket_storage_bytes
        self.total_files = math.ceil(self.bucket_count / self.buckets_per_file)

        self._prepare_sparse_files()

    def reset(self) -> None:
        if self.tree_dir.exists():
            shutil.rmtree(self.tree_dir)
        self.tree_dir = ensure_dir(self.root_dir / "tree_data")
        self._prepare_sparse_files()

    def exists(self, flat_index: int) -> bool:
        raw = self._read_raw_bucket(flat_index)
        return raw is not None

    def read(self, flat_index: int) -> Optional[Bucket]:
        raw = self._read_raw_bucket(flat_index)
        if raw is None:
            return None
        return self._decode_bucket(flat_index, raw)

    def write(self, flat_index: int, bucket: Bucket) -> None:
        raw = self._encode_bucket(bucket)
        file_path, byte_offset = self._locate(flat_index)
        with file_path.open("r+b") as f:
            f.seek(byte_offset)
            f.write(raw)

    # return the number of bucket slots assigned to a data file
    def _buckets_in_file(self, file_index: int) -> int:
        if file_index < 0 or file_index >= self.total_files:
            return 0
        start_bucket = file_index * self.buckets_per_file
        remaining = self.bucket_count - start_bucket
        if remaining <= 0:
            return 0
        return min(self.buckets_per_file, remaining)


    def _logical_size_for_file(self, file_index: int) -> int:
        return self._buckets_in_file(file_index) * self.bucket_storage_bytes

    def _prepare_sparse_files(self) -> None:
        for file_index in range(self.total_files):
            file_path = self.tree_dir / f"tree_data_{file_index}.bin"
            logical_size = self._logical_size_for_file(file_index)
            if logical_size <= 0:
                continue

            if file_path.exists() and file_path.stat().st_size == logical_size:
                continue

            with file_path.open("wb") as f:
                f.seek(logical_size - 1)
                f.write(b"\0")

    # map a flat bucket index to its file path, local index, and byte offset
    def _locate(self, flat_index: int) -> tuple[Path, int]:
        if flat_index < 0 or flat_index >= self.bucket_count:
            raise ValueError(f"flat_index {flat_index} out of range.")

        file_index = flat_index // self.buckets_per_file
        bucket_slot = flat_index % self.buckets_per_file
        byte_offset = bucket_slot * self.bucket_storage_bytes
        file_path = self.tree_dir / f"tree_data_{file_index}.bin"
        return file_path, byte_offset

    def _read_raw_bucket(self, flat_index: int) -> Optional[bytes]:
        """Read a raw serialized bucket from a fixed file offset."""
        file_path, byte_offset = self._locate(flat_index)
        with file_path.open("rb") as f:
            f.seek(byte_offset)
            raw = f.read(self.bucket_storage_bytes)

        if len(raw) != self.bucket_storage_bytes:
            return None

        if raw == b"\0" * self.bucket_storage_bytes:
            return None

        return raw

    def _encode_bucket(self, bucket: Bucket) -> bytes:
        if len(bucket.blocks) != self.bucket_size:
            raise ValueError(
                f"Bucket must contain exactly {self.bucket_size} blocks before file write."
            )

        parts: list[bytes] = []
        for block in bucket.blocks:
            if block.is_dummy:
                header = self.HEADER.pack(-1, -1, 0, self.FLAG_DUMMY)
                extra_meta = bytes(self.block_metadata_bytes - self.HEADER.size)
                payload = bytes(self.block_size)
            else:
                block_id = int(block.block_id) if block.block_id is not None else -1
                leaf = int(block.leaf) if block.leaf is not None else -1
                logical_size = int(
                    block.metadata.get("logical_payload_size", len(block.payload))
                )
                if logical_size < 0 or logical_size > self.block_size:
                    raise ValueError("Invalid logical_payload_size.")

                payload = bytes(block.payload)
                if len(payload) != self.block_size:
                    raise ValueError(
                        f"Real block payload must already be padded to block_size={self.block_size}."
                    )

                header = self.HEADER.pack(block_id, leaf, logical_size, 0)
                extra_meta = bytes(self.block_metadata_bytes - self.HEADER.size)

            parts.append(header + extra_meta + payload)

        raw = b"".join(parts)
        if len(raw) != self.bucket_storage_bytes:
            raise ValueError("Encoded bucket size mismatch.")
        return raw

    def _decode_bucket(self, flat_index: int, raw: bytes) -> Bucket:
        address = unflatten_bucket_index(flat_index)
        blocks: list[DataBlock] = []

        offset = 0
        for _ in range(self.bucket_size):
            meta = raw[offset : offset + self.block_metadata_bytes]
            payload = raw[
                offset + self.block_metadata_bytes :
                offset + self.block_storage_bytes
            ]
            offset += self.block_storage_bytes

            block_id, leaf, logical_size, flags = self.HEADER.unpack(
                meta[: self.HEADER.size]
            )
            is_dummy = bool(flags & self.FLAG_DUMMY)

            if is_dummy:
                blocks.append(
                    DataBlock(
                        block_id=None,
                        payload=bytes(self.block_size),
                        is_dummy=True,
                        leaf=None,
                        metadata={},
                    )
                )
            else:
                blocks.append(
                    DataBlock(
                        block_id=block_id,
                        payload=bytes(payload),
                        is_dummy=False,
                        leaf=None if leaf < 0 else leaf,
                        metadata={"logical_payload_size": int(logical_size)},
                    )
                )

        return Bucket(address=address, blocks=blocks)
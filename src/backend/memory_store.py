from __future__ import annotations

from typing import Optional

from src.common.interfaces import AbstractBucketStore
from src.common.types import Bucket


class InMemoryBucketStore(AbstractBucketStore):
    def __init__(self) -> None:
        self._buckets: dict[int, Bucket] = {}

    def reset(self) -> None:
        self._buckets.clear()

    def exists(self, flat_index: int) -> bool:
        return flat_index in self._buckets

    def read(self, flat_index: int) -> Optional[Bucket]:
        bucket = self._buckets.get(flat_index)
        if bucket is None:
            return None
        return bucket.clone()

    def write(self, flat_index: int, bucket: Bucket) -> None:
        self._buckets[flat_index] = bucket.clone()

    def __len__(self) -> int:
        return len(self._buckets)
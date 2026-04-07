from typing import Optional

from src.common.interfaces import AbstractBucketStore, AbstractORAM
from src.common.metrics import AccessResult
from src.common.types import (
    Bucket,
    BucketAddress,
    OperationType,
    Request,
    RequestKind,
)


class DummyStore(AbstractBucketStore):
    def __init__(self) -> None:
        self._data: dict[int, Bucket] = {}

    def reset(self) -> None:
        self._data.clear()

    def exists(self, flat_index: int) -> bool:
        return flat_index in self._data

    def read(self, flat_index: int) -> Optional[Bucket]:
        bucket = self._data.get(flat_index)
        return bucket.clone() if bucket is not None else None

    def write(self, flat_index: int, bucket: Bucket) -> None:
        self._data[flat_index] = bucket.clone()


class DummyORAM(AbstractORAM):
    def __init__(self) -> None:
        self.called = False
        self.queue: list[Request] = []

    def reset(self) -> None:
        self.called = False
        self.queue.clear()

    def access(self, request: Request) -> AccessResult:
        self.called = True
        result = AccessResult.empty(protocol="dummy")
        result.debug.note = f"handled request {request.request_id}"
        return result

    def enqueue(self, request: Request) -> None:
        self.queue.append(request)


def test_dummy_store_round_trip() -> None:
    store = DummyStore()
    bucket = Bucket(address=BucketAddress(level=0, index=0), blocks=[])

    assert not store.exists(0)
    store.write(0, bucket)
    assert store.exists(0)

    read_back = store.read(0)
    assert read_back is not None
    assert read_back.address == bucket.address


def test_dummy_oram_access_and_enqueue() -> None:
    oram = DummyORAM()
    req = Request(
        request_id=1,
        kind=RequestKind.REAL,
        op=OperationType.READ,
    )

    oram.enqueue(req)
    assert len(oram.queue) == 1

    result = oram.access(req)
    assert oram.called is True
    assert result.debug.note == "handled request 1"
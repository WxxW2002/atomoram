from src.common.config import StorageConfig
from src.common.types import BlockAddress, OperationType, Request, RequestKind
from src.protocols.direct_store import DirectStore

# construct a direct-store baseline instance for tests
def make_store() -> DirectStore:
    config = StorageConfig(
        block_size=16,
        bucket_size=4,
        tree_height=3,
        use_file_backend=False,
        data_dir="data/tmp",
    )
    return DirectStore(config=config)


def test_direct_store_write_then_read() -> None:
    store = make_store()

    write_req = Request(
        request_id=1,
        kind=RequestKind.REAL,
        op=OperationType.WRITE,
        address=BlockAddress(logical_id=7),
        data=b"abc",
    )
    write_result = store.access(write_req)
    assert write_result.data == b"abc"
    assert write_result.metrics.online_bucket_writes == 1
    assert write_result.metrics.online_bucket_reads == 0
    assert write_result.metrics.total_rtt == 1

    read_req = Request(
        request_id=2,
        kind=RequestKind.REAL,
        op=OperationType.READ,
        address=BlockAddress(logical_id=7),
    )
    read_result = store.access(read_req)
    assert read_result.data == b"abc"
    assert read_result.metrics.online_bucket_reads == 1
    assert read_result.metrics.online_bucket_writes == 0
    assert read_result.metrics.total_rtt == 1


def test_direct_store_read_missing_returns_none() -> None:
    store = make_store()

    read_req = Request(
        request_id=1,
        kind=RequestKind.REAL,
        op=OperationType.READ,
        address=BlockAddress(logical_id=99),
    )
    result = store.access(read_req)
    assert result.data is None
    assert result.metrics.online_bucket_reads == 1
    assert result.metrics.path_length_touched == 1
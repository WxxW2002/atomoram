import random

from src.common.config import StorageConfig
from src.common.types import BlockAddress, OperationType, Request, RequestKind
from src.protocols.path_oram import PathORAM


def make_oram(seed: int = 7) -> PathORAM:
    config = StorageConfig(
        block_size=32,
        bucket_size=4,
        tree_height=4,
        use_file_backend=False,
        data_dir="data/tmp",
    )
    return PathORAM(config=config, rng_seed=seed)


def test_path_oram_write_then_read() -> None:
    oram = make_oram()

    write_req = Request(
        request_id=1,
        kind=RequestKind.REAL,
        op=OperationType.WRITE,
        address=BlockAddress(logical_id=3),
        data=b"hello",
    )
    write_result = oram.access(write_req)

    assert write_result.data == b"hello"
    assert write_result.metrics.online_bucket_reads == oram.num_levels
    assert write_result.metrics.online_bucket_writes == oram.num_levels
    assert write_result.metrics.total_rtt == 2

    read_req = Request(
        request_id=2,
        kind=RequestKind.REAL,
        op=OperationType.READ,
        address=BlockAddress(logical_id=3),
    )
    read_result = oram.access(read_req)

    assert read_result.data == b"hello"
    assert read_result.metrics.online_bucket_reads == oram.num_levels
    assert read_result.metrics.online_bucket_writes == oram.num_levels
    assert read_result.metrics.total_rtt == 2


def test_path_oram_first_read_of_missing_block_returns_none() -> None:
    oram = make_oram()

    read_req = Request(
        request_id=1,
        kind=RequestKind.REAL,
        op=OperationType.READ,
        address=BlockAddress(logical_id=5),
    )
    result = oram.access(read_req)
    assert result.data is None
    assert result.metrics.real_requests_served == 1
    assert result.metrics.path_length_touched == oram.num_levels


def test_path_oram_random_read_write_correctness() -> None:
    oram = make_oram(seed=123)
    rng = random.Random(12345)

    truth: dict[int, bytes] = {}

    for req_id in range(1, 401):
        logical_id = rng.randrange(oram.logical_block_capacity)

        if not truth or rng.random() < 0.55:
            payload = bytes([rng.randrange(0, 256) for _ in range(rng.randrange(1, 17))])
            req = Request(
                request_id=req_id,
                kind=RequestKind.REAL,
                op=OperationType.WRITE,
                address=BlockAddress(logical_id=logical_id),
                data=payload,
            )
            result = oram.access(req)
            truth[logical_id] = payload
            assert result.data == payload
        else:
            logical_id = rng.choice(list(truth.keys()))
            req = Request(
                request_id=req_id,
                kind=RequestKind.REAL,
                op=OperationType.READ,
                address=BlockAddress(logical_id=logical_id),
            )
            result = oram.access(req)
            assert result.data == truth[logical_id]


def test_path_oram_reset_clears_state() -> None:
    oram = make_oram()

    req = Request(
        request_id=1,
        kind=RequestKind.REAL,
        op=OperationType.WRITE,
        address=BlockAddress(logical_id=1),
        data=b"xyz",
    )
    oram.access(req)

    assert len(oram.stash) >= 0
    assert any(leaf is not None for leaf in oram.position_map)

    oram.reset()

    assert len(oram.stash) == 0
    assert all(leaf is None for leaf in oram.position_map)
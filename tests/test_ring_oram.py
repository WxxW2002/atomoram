import random

from src.common.config import RingConfig, StorageConfig
from src.common.types import BlockAddress, OperationType, Request, RequestKind
from src.protocols.ring_oram import RingORAM


def make_oram(
    *,
    seed: int = 7,
    tree_height: int = 4,
    bucket_size: int = 4,
    s_num: int = 12,
    a_num: int = 8,
) -> RingORAM:
    storage = StorageConfig(
        block_size=32,
        bucket_size=bucket_size,
        tree_height=tree_height,
        use_file_backend=False,
        data_dir="data/tmp",
    )
    ring = RingConfig(s_num=s_num, a_num=a_num)
    return RingORAM(storage_config=storage, ring_config=ring, rng_seed=seed)


def test_ring_oram_write_then_read_basic_correctness() -> None:
    # Disable eviction / reshuffle side effects for this basic test.
    oram = make_oram(s_num=100, a_num=100)

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
    assert write_result.metrics.online_bucket_writes == 0
    assert write_result.metrics.total_rtt == 1

    read_req = Request(
        request_id=2,
        kind=RequestKind.REAL,
        op=OperationType.READ,
        address=BlockAddress(logical_id=3),
    )
    read_result = oram.access(read_req)
    assert read_result.data == b"hello"
    assert read_result.metrics.online_bucket_reads == oram.num_levels
    assert read_result.metrics.online_bucket_writes == 0
    assert read_result.metrics.total_rtt == 1


def test_ring_oram_periodic_eviction_trigger() -> None:
    # Disable early reshuffle, force periodic eviction every 2 accesses.
    oram = make_oram(s_num=100, a_num=2)

    req1 = Request(
        request_id=1,
        kind=RequestKind.REAL,
        op=OperationType.WRITE,
        address=BlockAddress(logical_id=1),
        data=b"a",
    )
    result1 = oram.access(req1)
    assert result1.metrics.eviction_count == 0
    assert result1.metrics.online_bucket_reads == oram.num_levels
    assert result1.metrics.online_bucket_writes == 0
    assert result1.metrics.total_rtt == 1

    req2 = Request(
        request_id=2,
        kind=RequestKind.REAL,
        op=OperationType.WRITE,
        address=BlockAddress(logical_id=2),
        data=b"b",
    )
    result2 = oram.access(req2)

    # read_ring_path + full-path eviction
    assert result2.metrics.eviction_count == 1
    assert result2.metrics.online_bucket_reads == 2 * oram.num_levels
    assert result2.metrics.online_bucket_writes == oram.num_levels
    assert result2.metrics.total_rtt == 2


def test_ring_oram_early_reshuffle_trigger() -> None:
    # With S=1, every bucket on the accessed path reaches the threshold after one access.
    oram = make_oram(s_num=1, a_num=100)

    req = Request(
        request_id=1,
        kind=RequestKind.REAL,
        op=OperationType.WRITE,
        address=BlockAddress(logical_id=0),
        data=b"x",
    )
    result = oram.access(req)

    # read_ring_path + reshuffle for every bucket on the path
    assert result.metrics.reshuffle_count == oram.num_levels
    assert result.metrics.online_bucket_reads == 2 * oram.num_levels
    assert result.metrics.online_bucket_writes == oram.num_levels
    assert result.metrics.total_rtt == 2


def test_ring_oram_random_read_write_correctness() -> None:
    oram = make_oram(seed=123, s_num=12, a_num=8)
    rng = random.Random(2026)

    truth: dict[int, bytes] = {}

    for req_id in range(1, 301):
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


def test_ring_oram_reset_clears_state() -> None:
    oram = make_oram()

    req = Request(
        request_id=1,
        kind=RequestKind.REAL,
        op=OperationType.WRITE,
        address=BlockAddress(logical_id=1),
        data=b"xyz",
    )
    oram.access(req)

    assert any(leaf is not None for leaf in oram.position_map)

    oram.reset()

    assert len(oram.stash) == 0
    assert all(leaf is None for leaf in oram.position_map)
    assert all(count == 0 for count in oram.count)
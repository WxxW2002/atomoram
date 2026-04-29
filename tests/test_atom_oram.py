from src.common.config import AtomConfig, StorageConfig
from src.common.types import (
    BlockAddress,
    BucketAddress,
    DataBlock,
    OperationType,
    Request,
    RequestKind,
)
from src.protocols.atom_oram import AtomORAM


def make_oram(seed: int = 7) -> AtomORAM:
    storage = StorageConfig(
        block_size=32,
        bucket_size=4,
        tree_height=4,
        use_file_backend=False,
        data_dir="data/tmp",
    )
    atom = AtomConfig(
        lambda1=1.0,
        tick_interval_sec=0.001,
        queue_limit=100000,
    )
    return AtomORAM(storage_config=storage, atom_config=atom, rng_seed=seed)


def _pin_epoch(oram: AtomORAM, *, leaf: int, step: int = 0) -> None:
    oram.current_epoch_leaf = leaf
    oram.current_epoch_step = step

    if step == 0:
        oram.carried_epoch_bucket = None
    else:
        carried_level = oram.tree_height - step
        oram.carried_epoch_bucket = oram._bucket_address_on_leaf(
            leaf,
            carried_level,
        )


def test_atom_oram_write_then_read_counts_match_new_epoch_semantics() -> None:
    oram = make_oram()

    target_bucket = BucketAddress(level=2, index=1)
    oram.position_map[3] = target_bucket

    oram._sample_leaf = lambda: 5  # type: ignore[method-assign]
    _pin_epoch(oram, leaf=13, step=0)

    write_req = Request(
        request_id=1,
        kind=RequestKind.REAL,
        op=OperationType.WRITE,
        address=BlockAddress(logical_id=3),
        data=b"hello",
    )
    write_result = oram.access(write_req)

    assert write_result.data == b"hello"
    assert write_result.metrics.online_bucket_reads == 1
    assert write_result.metrics.online_bucket_writes == 0
    assert write_result.metrics.online_rtt == 1

    assert write_result.metrics.offline_bucket_reads == 2
    assert write_result.metrics.offline_bucket_writes == 2
    assert write_result.metrics.offline_rtt == 2

    assert write_result.metrics.path_length_touched == 1
    assert oram.position_map[3] is None
    assert 3 in oram.stash
    assert oram.pending_flush_count == 3
    assert oram.carried_epoch_bucket == BucketAddress(level=3, index=6)

    oram._sample_uniform_bucket_address = lambda: target_bucket  # type: ignore[method-assign]
    _pin_epoch(oram, leaf=13, step=1)

    read_req = Request(
        request_id=2,
        kind=RequestKind.REAL,
        op=OperationType.READ,
        address=BlockAddress(logical_id=3),
    )
    read_result = oram.access(read_req)

    assert read_result.data == b"hello"

    assert read_result.metrics.online_bucket_reads == 1
    assert read_result.metrics.offline_bucket_reads == 1
    assert read_result.metrics.offline_bucket_writes == 2
    assert read_result.metrics.offline_rtt == 1

    assert 3 in oram.stash
    assert oram.carried_epoch_bucket == BucketAddress(level=2, index=3)


def test_atom_oram_epoch_pair_writeback_is_deeper_first() -> None:
    oram = make_oram()

    lower_bucket = BucketAddress(level=4, index=13)
    upper_bucket = BucketAddress(level=3, index=6)
    target_bucket = BucketAddress(level=2, index=1)

    block_a = DataBlock(
        block_id=1,
        payload=b"a",
        is_dummy=False,
        leaf=13,
        metadata={"logical_payload_size": 1},
    )
    block_b = DataBlock(
        block_id=2,
        payload=b"bc",
        is_dummy=False,
        leaf=12,
        metadata={"logical_payload_size": 2},
    )

    oram.debug_seed_bucket(bucket_address=target_bucket, blocks=[])
    oram.debug_seed_bucket(bucket_address=lower_bucket, blocks=[block_a])
    oram.debug_seed_bucket(bucket_address=upper_bucket, blocks=[block_b])

    oram.position_map[9] = target_bucket
    oram._sample_leaf = lambda: 5  # type: ignore[method-assign]
    _pin_epoch(oram, leaf=13, step=0)

    req = Request(
        request_id=1,
        kind=RequestKind.REAL,
        op=OperationType.WRITE,
        address=BlockAddress(logical_id=9),
        data=b"xyz",
    )
    result = oram.access(req)

    assert result.metrics.online_bucket_reads == 1
    assert result.metrics.offline_bucket_reads == 2
    assert result.metrics.offline_bucket_writes == 2
    assert result.metrics.offline_rtt == 2

    # First pipeline step writes only the lower bucket.
    assert oram.position_map[1] == lower_bucket

    assert oram.position_map[2] is None
    assert 2 in oram.stash
    assert oram.carried_epoch_bucket == upper_bucket

    assert oram.position_map[9] is None
    assert 9 in oram.stash
    assert len(oram.stash) == 2

    assert (lower_bucket.level, lower_bucket.index) not in oram.invalidated_buckets
    assert (upper_bucket.level, upper_bucket.index) in oram.invalidated_buckets
    assert (target_bucket.level, target_bucket.index) not in oram.invalidated_buckets

    second_target = BucketAddress(level=2, index=1)
    req2 = Request(
        request_id=2,
        kind=RequestKind.VIRTUAL,
        op=OperationType.READ,
        address=second_target,
    )
    result2 = oram.access(req2)

    assert result2.metrics.online_bucket_reads == 1
    assert result2.metrics.offline_bucket_reads == 1
    assert result2.metrics.offline_bucket_writes == 2
    assert result2.metrics.offline_rtt == 1

    assert oram.position_map[2] == upper_bucket
    assert 2 not in oram.stash
    assert oram.carried_epoch_bucket == BucketAddress(level=2, index=3)


def test_atom_oram_virtual_access_executes_epoch_micro_eviction() -> None:
    oram = make_oram()

    non_root_bucket = BucketAddress(level=2, index=2)
    oram._sample_uniform_bucket_address = lambda: non_root_bucket  # type: ignore[method-assign]
    _pin_epoch(oram, leaf=13, step=0)

    req = Request(
        request_id=1,
        kind=RequestKind.VIRTUAL,
        op=OperationType.READ,
        address=None,
    )
    result = oram.access(req)

    assert result.data is None
    assert result.metrics.virtual_requests_executed == 1
    assert result.metrics.online_bucket_reads == 1
    assert result.metrics.online_bucket_writes == 0
    assert result.metrics.online_rtt == 1
    assert result.metrics.offline_bucket_reads == 2
    assert result.metrics.offline_bucket_writes == 2
    assert result.metrics.offline_rtt == 2
    assert oram.carried_epoch_bucket == BucketAddress(level=3, index=6)


def test_atom_oram_virtual_access_can_execute_explicit_bucket_address() -> None:
    oram = make_oram()

    explicit_bucket = BucketAddress(level=2, index=1)
    _pin_epoch(oram, leaf=13, step=0)

    req = Request(
        request_id=1,
        kind=RequestKind.VIRTUAL,
        op=OperationType.READ,
        address=explicit_bucket,
    )
    result = oram.access(req)

    assert result.data is None
    assert result.debug.current_bucket == (2, 1)
    assert result.metrics.virtual_requests_executed == 1
    assert result.metrics.online_bucket_reads == 1


def test_atom_oram_missing_read_non_root_returns_none_with_new_counts() -> None:
    oram = make_oram()

    non_root_bucket = BucketAddress(level=3, index=4)
    oram._sample_uniform_bucket_address = lambda: non_root_bucket  # type: ignore[method-assign]
    _pin_epoch(oram, leaf=13, step=0)

    req = Request(
        request_id=1,
        kind=RequestKind.REAL,
        op=OperationType.READ,
        address=BlockAddress(logical_id=7),
    )
    result = oram.access(req)

    assert result.data is None
    assert result.metrics.real_requests_served == 1
    assert result.metrics.online_bucket_reads == 1
    assert result.metrics.offline_bucket_reads == 2
    assert result.metrics.offline_bucket_writes == 2
    assert result.metrics.offline_rtt == 2
    assert oram.carried_epoch_bucket == BucketAddress(level=3, index=6)


def make_file_oram(tmp_path, seed: int = 7) -> AtomORAM:
    storage = StorageConfig(
        block_size=32,
        bucket_size=4,
        tree_height=4,
        use_file_backend=True,
        data_dir=str(tmp_path),
        data_file_size=512,
    )
    atom = AtomConfig(
        lambda1=1.0,
        tick_interval_sec=0.001,
        queue_limit=100000,
    )
    return AtomORAM(storage_config=storage, atom_config=atom, rng_seed=seed)


def test_atom_oram_file_backend_matches_memory_backend(tmp_path) -> None:
    mem_oram = make_oram(seed=7)
    file_oram = make_file_oram(tmp_path, seed=7)

    requests = []

    for i in range(12):
        requests.append(
            Request(
                request_id=1000 + i,
                kind=RequestKind.REAL,
                op=OperationType.WRITE,
                address=BlockAddress(logical_id=i),
                data=bytes([i + 1]) * (5 + (i % 7)),
            )
        )
        for j in range(3):
            requests.append(
                Request(
                    request_id=2000 + i * 10 + j,
                    kind=RequestKind.VIRTUAL,
                    op=OperationType.READ,
                    address=None,
                    data=None,
                )
            )

    for i in range(20):
        requests.append(
            Request(
                request_id=3000 + i,
                kind=RequestKind.REAL,
                op=OperationType.READ,
                address=BlockAddress(logical_id=i % 12),
                data=None,
            )
        )
        requests.append(
            Request(
                request_id=4000 + i,
                kind=RequestKind.REAL,
                op=OperationType.WRITE,
                address=BlockAddress(logical_id=(i * 3) % 12),
                data=bytes([50 + i]) * (6 + (i % 5)),
            )
        )
        for j in range(2):
            requests.append(
                Request(
                    request_id=5000 + i * 10 + j,
                    kind=RequestKind.VIRTUAL,
                    op=OperationType.READ,
                    address=None,
                    data=None,
                )
            )

    mem_stash_sizes = []
    file_stash_sizes = []

    for req in requests:
        mem_res = mem_oram.access(req)
        file_res = file_oram.access(req)

        mem_stash_sizes.append(mem_res.metrics.stash_size_after)
        file_stash_sizes.append(file_res.metrics.stash_size_after)

        assert mem_res.metrics.online_bucket_reads == file_res.metrics.online_bucket_reads
        assert mem_res.metrics.offline_bucket_reads == file_res.metrics.offline_bucket_reads
        assert mem_res.metrics.offline_bucket_writes == file_res.metrics.offline_bucket_writes
        assert mem_res.metrics.stash_size_after == file_res.metrics.stash_size_after

    assert mem_stash_sizes == file_stash_sizes
    assert sorted(mem_oram.stash.keys()) == sorted(file_oram.stash.keys())

    mem_pos = [x for x in mem_oram.position_map[:12]]
    file_pos = [x for x in file_oram.position_map[:12]]
    assert mem_pos == file_pos


def test_atom_oram_local_top_half_online_read_has_no_network_cost() -> None:
    storage = StorageConfig(
        block_size=32,
        bucket_size=4,
        tree_height=4,
        use_file_backend=False,
        data_dir="data/tmp",
    )
    atom_cfg = AtomConfig(
        lambda1=1.0,
        tick_interval_sec=0.001,
        queue_limit=100000,
        local_top_half_enabled=True,
        local_cutoff_level=None,
    )
    oram = AtomORAM(storage_config=storage, atom_config=atom_cfg, rng_seed=7)

    oram.position_map[0] = BucketAddress(level=1, index=0)
    _pin_epoch(oram, leaf=13, step=0)

    req = Request(
        request_id=1,
        kind=RequestKind.REAL,
        op=OperationType.WRITE,
        address=BlockAddress(logical_id=0),
        data=b"\x11" * storage.block_size,
        arrival_time=0.0,
        issued_time=0.0,
        tag="test",
    )
    res = oram.access(req)

    assert res.metrics.online_bucket_reads == 1
    assert res.metrics.online_bytes_down == 0
    assert res.metrics.online_rtt == 0


def test_atom_oram_remote_lower_half_online_read_keeps_network_cost() -> None:
    storage = StorageConfig(
        block_size=32,
        bucket_size=4,
        tree_height=4,
        use_file_backend=False,
        data_dir="data/tmp",
    )
    atom_cfg = AtomConfig(
        lambda1=1.0,
        tick_interval_sec=0.001,
        queue_limit=100000,
        local_top_half_enabled=True,
        local_cutoff_level=None,
    )
    oram = AtomORAM(storage_config=storage, atom_config=atom_cfg, rng_seed=7)

    oram.position_map[0] = BucketAddress(level=3, index=0)
    _pin_epoch(oram, leaf=13, step=0)

    req = Request(
        request_id=1,
        kind=RequestKind.REAL,
        op=OperationType.WRITE,
        address=BlockAddress(logical_id=0),
        data=b"\x22" * storage.block_size,
        arrival_time=0.0,
        issued_time=0.0,
        tag="test",
    )
    res = oram.access(req)

    assert res.metrics.online_bucket_reads == 1
    assert res.metrics.online_bytes_down == oram.backend.bucket_storage_bytes
    assert res.metrics.online_rtt == 1

def test_atom_oram_pipeline_io_counts_across_one_epoch() -> None:
    oram = make_oram()

    target_bucket = BucketAddress(level=4, index=0)
    _pin_epoch(oram, leaf=13, step=0)

    expected = [
        # first step: read L and L-1, write L and target
        (2, 2, 2, BucketAddress(level=3, index=6)),
        # middle step: read new upper, write carried/lower and target
        (1, 2, 1, BucketAddress(level=2, index=3)),
        (1, 2, 1, BucketAddress(level=1, index=1)),
        # last step: read root, write level 1, root, and target
        (1, 3, 1, None),
    ]

    for i, (expected_reads, expected_writes, expected_rtt, expected_carry) in enumerate(expected):
        req = Request(
            request_id=100 + i,
            kind=RequestKind.VIRTUAL,
            op=OperationType.READ,
            address=target_bucket,
        )
        result = oram.access(req)

        assert result.metrics.online_bucket_reads == 1
        assert result.metrics.online_bucket_writes == 0
        assert result.metrics.online_rtt == 1

        assert result.metrics.offline_bucket_reads == expected_reads
        assert result.metrics.offline_bucket_writes == expected_writes
        assert result.metrics.offline_rtt == expected_rtt
        assert oram.carried_epoch_bucket == expected_carry

    assert oram.current_epoch_leaf is None
    assert oram.current_epoch_step == 0
    assert oram.pending_flush_count == 0
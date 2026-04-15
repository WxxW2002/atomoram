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


def test_atom_oram_write_then_read_full_access() -> None:
    oram = make_oram()

    target_bucket = BucketAddress(level=2, index=1)
    oram.position_map[3] = target_bucket

    # Force the reassigned path to stay inside the target subtree, so the block
    # can be written back to the target bucket.
    oram._sample_leaf = lambda: 5  # type: ignore[method-assign]

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
    assert write_result.metrics.offline_bucket_reads == 1
    assert write_result.metrics.offline_bucket_writes == 2
    assert write_result.metrics.offline_rtt == 1
    assert write_result.metrics.path_length_touched == 1
    assert oram.pending_flush_count == 0
    assert oram.position_map[3] == target_bucket
    assert 3 not in oram.stash

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
    assert oram.pending_flush_count == 0
    assert oram.position_map[3] == target_bucket
    assert 3 not in oram.stash


def test_atom_oram_local_greedy_writeback_target_then_parent() -> None:
    oram = make_oram()

    target_bucket = BucketAddress(level=2, index=1)
    parent_bucket = BucketAddress(level=1, index=0)

    # block_a fits target (and parent); block_b fits parent only.
    block_a = DataBlock(
        block_id=1,
        payload=b"a",
        is_dummy=False,
        leaf=5,  # path includes level-2 index-1 and level-1 index-0
        metadata={"logical_payload_size": 1},
    )
    block_b = DataBlock(
        block_id=2,
        payload=b"bc",
        is_dummy=False,
        leaf=1,  # path includes parent only, not target
        metadata={"logical_payload_size": 2},
    )

    oram.debug_seed_bucket(bucket_address=target_bucket, blocks=[block_a])
    oram.debug_seed_bucket(bucket_address=parent_bucket, blocks=[block_b])

    oram._sample_leaf = lambda: 5 

    req = Request(
        request_id=1,
        kind=RequestKind.REAL,
        op=OperationType.READ,
        address=BlockAddress(logical_id=1),
    )
    result = oram.access(req)

    assert result.data == b"a"
    assert result.metrics.online_bucket_reads == 1
    assert result.metrics.offline_bucket_reads == 1
    assert result.metrics.offline_bucket_writes == 2
    assert result.metrics.offline_rtt == 1

    assert oram.position_map[1] == target_bucket
    assert oram.position_map[2] == parent_bucket
    assert len(oram.stash) == 0
    assert (target_bucket.level, target_bucket.index) not in oram.invalidated_buckets
    assert (parent_bucket.level, parent_bucket.index) not in oram.invalidated_buckets
    assert oram.pending_flush_count == 0


def test_atom_oram_root_target_skips_offline_flush() -> None:
    oram = make_oram()

    root_bucket = BucketAddress(level=0, index=0)
    block = DataBlock(
        block_id=4,
        payload=b"root",
        is_dummy=False,
        leaf=0,
        metadata={"logical_payload_size": 4},
    )
    oram.debug_seed_bucket(bucket_address=root_bucket, blocks=[block])

    oram._sample_leaf = lambda: 0  # type: ignore[method-assign]

    req = Request(
        request_id=1,
        kind=RequestKind.REAL,
        op=OperationType.READ,
        address=BlockAddress(logical_id=4),
    )
    result = oram.access(req)

    assert result.data == b"root"
    assert result.metrics.online_bucket_reads == 1
    assert result.metrics.online_rtt == 1
    assert result.metrics.offline_bucket_reads == 0
    assert result.metrics.offline_bucket_writes == 0
    assert result.metrics.offline_rtt == 0

    assert oram.pending_flush_count == 0
    assert (0, 0) in oram.invalidated_buckets
    assert 4 in oram.stash
    assert oram.position_map[4] is None


def test_atom_oram_virtual_access_non_root_executes_offline_flush() -> None:
    oram = make_oram()

    non_root_bucket = BucketAddress(level=2, index=2)
    oram._sample_uniform_bucket_address = lambda: non_root_bucket  # type: ignore[method-assign]

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
    assert result.metrics.offline_bucket_reads == 1
    assert result.metrics.offline_bucket_writes == 2
    assert result.metrics.offline_rtt == 1
    assert oram.pending_flush_count == 0


def test_atom_oram_missing_read_non_root_returns_none_and_flushes() -> None:
    oram = make_oram()

    non_root_bucket = BucketAddress(level=3, index=4)
    oram._sample_uniform_bucket_address = lambda: non_root_bucket  # type: ignore[method-assign]

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
    assert result.metrics.offline_bucket_reads == 1
    assert result.metrics.offline_bucket_writes == 2
    assert result.metrics.offline_rtt == 1
    assert oram.pending_flush_count == 0

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

    # 先做一批写入，建立非平凡状态
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

    # 再混合读写
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

    # level=1 < tree_height//2=2 => local
    oram.position_map[0] = BucketAddress(level=1, index=0)

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

    # level=3 >= tree_height//2=2 => remote
    oram.position_map[0] = BucketAddress(level=3, index=0)

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
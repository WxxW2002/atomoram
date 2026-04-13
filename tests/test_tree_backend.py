from pathlib import Path

from src.backend.tree_backend import TreeBackend
from src.common.config import StorageConfig
from src.common.types import Bucket, BucketAddress, DataBlock


def make_memory_backend() -> TreeBackend:
    config = StorageConfig(
        block_size=16,
        bucket_size=4,
        tree_height=3,
        use_file_backend=False,
        data_dir="data/tmp",
        data_file_size=4096,
    )
    return TreeBackend(config=config)


def make_file_backend(tmp_path: Path, *, data_file_size: int = 512) -> TreeBackend:
    config = StorageConfig(
        block_size=16,
        bucket_size=4,
        tree_height=3,
        use_file_backend=True,
        data_dir=str(tmp_path),
        data_file_size=data_file_size,
    )
    return TreeBackend(config=config)


def test_tree_backend_basic_properties_memory() -> None:
    backend = make_memory_backend()

    assert backend.block_size == 16
    assert backend.bucket_size == 4
    assert backend.tree_height == 3
    assert backend.num_levels == 4
    assert backend.leaf_count == 8
    assert backend.bucket_count == 15
    assert backend.bucket_storage_bytes == 4 * (16 + backend.block_metadata_bytes)


def test_read_empty_bucket_returns_padded_dummy_bucket_memory() -> None:
    backend = make_memory_backend()
    address = BucketAddress(level=2, index=1)

    bucket = backend.read_bucket(address)

    assert bucket.address == address
    assert len(bucket.blocks) == backend.bucket_size
    assert all(block.is_dummy for block in bucket.blocks)
    assert all(len(block.payload) == backend.block_size for block in bucket.blocks)


def test_write_bucket_and_read_back_round_trip_memory() -> None:
    backend = make_memory_backend()
    address = BucketAddress(level=1, index=1)

    real_block = DataBlock(
        block_id=7,
        payload=b"abc",
        is_dummy=False,
        leaf=6,
        metadata={},
    )
    bucket = Bucket(address=address, blocks=[real_block])

    backend.write_bucket(bucket)
    read_back = backend.read_bucket(address)

    assert read_back.address == address
    assert len(read_back.blocks) == backend.bucket_size

    non_dummy = read_back.non_dummy_blocks()
    assert len(non_dummy) == 1
    assert non_dummy[0].block_id == 7
    assert non_dummy[0].leaf == 6
    assert non_dummy[0].payload.startswith(b"abc")
    assert len(non_dummy[0].payload) == backend.block_size
    assert non_dummy[0].metadata["logical_payload_size"] == 3

    dummy_count = sum(1 for block in read_back.blocks if block.is_dummy)
    assert dummy_count == backend.bucket_size - 1


def test_file_backend_round_trip_and_fixed_offset(tmp_path: Path) -> None:
    backend = make_file_backend(tmp_path, data_file_size=512)
    address = BucketAddress(level=2, index=1)

    real_block = DataBlock(
        block_id=11,
        payload=b"hello",
        is_dummy=False,
        leaf=5,
        metadata={},
    )
    bucket = Bucket(address=address, blocks=[real_block])

    backend.write_bucket(bucket)
    read_back = backend.read_bucket(address)

    assert read_back.address == address
    assert len(read_back.blocks) == backend.bucket_size

    non_dummy = read_back.non_dummy_blocks()
    assert len(non_dummy) == 1
    assert non_dummy[0].block_id == 11
    assert non_dummy[0].leaf == 5
    assert non_dummy[0].payload.startswith(b"hello")
    assert len(non_dummy[0].payload) == backend.block_size
    assert non_dummy[0].metadata["logical_payload_size"] == 5

    store = backend.bucket_store
    flat_index = backend.flatten_address(address)
    file_path, byte_offset = store._locate(flat_index)

    assert file_path.exists()
    assert file_path.parent == Path(tmp_path) / "tree_data"
    assert byte_offset % backend.bucket_storage_bytes == 0

    with file_path.open("rb") as f:
        f.seek(byte_offset)
        raw = f.read(backend.bucket_storage_bytes)

    assert len(raw) == backend.bucket_storage_bytes
    assert raw != bytes(backend.bucket_storage_bytes)


def test_file_backend_creates_multiple_tree_files_when_needed(tmp_path: Path) -> None:
    backend = make_file_backend(tmp_path, data_file_size=512)

    tree_dir = Path(tmp_path) / "tree_data"
    files = sorted(tree_dir.glob("tree_data_*.bin"))

    assert len(files) >= 2

    for p in files:
        assert p.exists()
        assert p.stat().st_size > 0


def test_file_backend_reset_clears_previous_contents(tmp_path: Path) -> None:
    backend = make_file_backend(tmp_path, data_file_size=512)
    address = BucketAddress(level=1, index=1)

    backend.write_bucket(
        Bucket(
            address=address,
            blocks=[
                DataBlock(
                    block_id=23,
                    payload=b"xyz",
                    is_dummy=False,
                    leaf=7,
                    metadata={},
                )
            ],
        )
    )

    before = backend.read_bucket(address)
    assert len(before.non_dummy_blocks()) == 1

    backend.reset()

    after = backend.read_bucket(address)
    assert after.address == address
    assert len(after.blocks) == backend.bucket_size
    assert len(after.non_dummy_blocks()) == 0
    assert all(block.is_dummy for block in after.blocks)


def test_path_helpers() -> None:
    backend = make_memory_backend()
    leaf = 5

    path = backend.path_to_leaf(leaf)

    assert path == [
        BucketAddress(level=0, index=0),
        BucketAddress(level=1, index=1),
        BucketAddress(level=2, index=2),
        BucketAddress(level=3, index=5),
    ]

    assert backend.bucket_address_on_path(leaf=5, level=2) == BucketAddress(level=2, index=2)
    assert backend.is_bucket_on_path(BucketAddress(level=2, index=2), leaf=5) is True
    assert backend.is_bucket_on_path(BucketAddress(level=2, index=3), leaf=5) is False


def test_parent_and_children_addresses() -> None:
    backend = make_memory_backend()

    node = BucketAddress(level=2, index=3)
    parent = backend.parent_address(node)
    children = backend.children_addresses(node)

    assert parent == BucketAddress(level=1, index=1)
    assert children == [
        BucketAddress(level=3, index=6),
        BucketAddress(level=3, index=7),
    ]

    assert backend.parent_address(BucketAddress(level=0, index=0)) is None
    assert backend.children_addresses(BucketAddress(level=3, index=7)) == []

def test_file_backend_multi_real_blocks_round_trip(tmp_path):
    from src.backend.tree_backend import TreeBackend
    from src.common.config import StorageConfig
    from src.common.types import Bucket, BucketAddress, DataBlock

    cfg = StorageConfig(
        block_size=16,
        bucket_size=4,
        tree_height=3,
        use_file_backend=True,
        data_dir=str(tmp_path),
        data_file_size=4096,
    )
    backend = TreeBackend(config=cfg)
    address = BucketAddress(level=2, index=3)

    bucket = Bucket(
        address=address,
        blocks=[
            DataBlock(block_id=1, payload=b"A" * 3, is_dummy=False, leaf=4, metadata={}),
            DataBlock(block_id=2, payload=b"B" * 5, is_dummy=False, leaf=5, metadata={}),
            DataBlock(block_id=3, payload=b"C" * 7, is_dummy=False, leaf=6, metadata={}),
        ],
    )

    backend.write_bucket(bucket)
    read_back = backend.read_bucket(address)

    non_dummy = sorted(read_back.non_dummy_blocks(), key=lambda b: b.block_id)
    assert len(non_dummy) == 3

    assert non_dummy[0].block_id == 1
    assert non_dummy[0].leaf == 4
    assert non_dummy[0].payload.startswith(b"A" * 3)
    assert non_dummy[0].metadata["logical_payload_size"] == 3

    assert non_dummy[1].block_id == 2
    assert non_dummy[1].leaf == 5
    assert non_dummy[1].payload.startswith(b"B" * 5)
    assert non_dummy[1].metadata["logical_payload_size"] == 5

    assert non_dummy[2].block_id == 3
    assert non_dummy[2].leaf == 6
    assert non_dummy[2].payload.startswith(b"C" * 7)
    assert non_dummy[2].metadata["logical_payload_size"] == 7
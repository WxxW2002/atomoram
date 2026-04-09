from src.backend.tree_backend import TreeBackend
from src.common.config import StorageConfig
from src.common.types import Bucket, BucketAddress, DataBlock


def make_backend() -> TreeBackend:
    config = StorageConfig(
        block_size=16,
        bucket_size=4,
        tree_height=3,
        use_file_backend=False,
        data_dir="data/tmp",
    )
    return TreeBackend(config=config)


def test_tree_backend_basic_properties() -> None:
    backend = make_backend()

    assert backend.block_size == 16
    assert backend.bucket_size == 4
    assert backend.tree_height == 3
    assert backend.num_levels == 4
    assert backend.leaf_count == 8
    assert backend.bucket_count == 15
    assert backend.bucket_storage_bytes == 4 * (16 + backend.block_metadata_bytes)


def test_read_empty_bucket_returns_padded_dummy_bucket() -> None:
    backend = make_backend()
    address = BucketAddress(level=2, index=1)

    bucket = backend.read_bucket(address)

    assert bucket.address == address
    assert len(bucket.blocks) == backend.bucket_size
    assert all(block.is_dummy for block in bucket.blocks)
    assert all(len(block.payload) == backend.block_size for block in bucket.blocks)


def test_write_bucket_and_read_back_round_trip() -> None:
    backend = make_backend()
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


def test_path_helpers() -> None:
    backend = make_backend()
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
    backend = make_backend()

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
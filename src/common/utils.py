from __future__ import annotations

import random
import numpy as np
from pathlib import Path
from typing import Optional
from src.common.types import DataBlock
from src.common.types import BucketAddress

def truncate_payload(block: Optional[DataBlock]) -> Optional[bytes]:
    if block is None:
        return None
    logical_size = block.metadata.get("logical_payload_size", len(block.payload))
    return block.payload[:logical_size]

def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def leaf_count_from_height(tree_height: int) -> int:
    if tree_height < 0:
        raise ValueError("tree_height must be non-negative.")
    return 1 << tree_height


def bucket_count_from_height(tree_height: int) -> int:
    if tree_height < 0:
        raise ValueError("tree_height must be non-negative.")
    return (1 << (tree_height + 1)) - 1


def flatten_bucket_address(address: BucketAddress) -> int:
    if address.level < 0:
        raise ValueError("Bucket level must be non-negative.")
    if address.index < 0 or address.index >= (1 << address.level):
        raise ValueError(
            f"Invalid bucket index {address.index} for level {address.level}."
        )
    return (1 << address.level) - 1 + address.index


def unflatten_bucket_index(flat_index: int) -> BucketAddress:
    if flat_index < 0:
        raise ValueError("flat_index must be non-negative.")

    level = 0
    while (1 << (level + 1)) - 1 <= flat_index:
        level += 1

    first_index_at_level = (1 << level) - 1
    return BucketAddress(level=level, index=flat_index - first_index_at_level)


def bucket_address_on_path(tree_height: int, leaf: int, level: int) -> BucketAddress:
    if level < 0 or level > tree_height:
        raise ValueError("Requested level is outside the tree.")
    leaf_count = leaf_count_from_height(tree_height)
    if leaf < 0 or leaf >= leaf_count:
        raise ValueError("Leaf index is outside the valid range.")

    shift = tree_height - level
    index = leaf >> shift
    return BucketAddress(level=level, index=index)


def path_to_leaf(tree_height: int, leaf: int) -> list[BucketAddress]:
    return [
        bucket_address_on_path(tree_height=tree_height, leaf=leaf, level=level)
        for level in range(tree_height + 1)
    ]


def is_bucket_on_leaf_path(
    tree_height: int,
    address: BucketAddress,
    leaf: int,
) -> bool:
    expected = bucket_address_on_path(
        tree_height=tree_height,
        leaf=leaf,
        level=address.level,
    )
    return expected == address
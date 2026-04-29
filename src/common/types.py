from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class OperationType(str, Enum):
    READ = "read"
    WRITE = "write"


class RequestKind(str, Enum):
    REAL = "real"
    VIRTUAL = "virtual"


class ProtocolKind(str, Enum):
    DIRECT = "direct_store"
    PATH = "path_oram"
    RING = "ring_oram"
    ATOM = "atom_oram"


@dataclass(slots=True, frozen=True)
class BlockAddress:
    logical_id: int


@dataclass(slots=True, frozen=True)
class BucketAddress:
    level: int
    index: int


#  access request with timing, operation, address, and optional data
@dataclass(slots=True)
class Request:
    request_id: int
    kind: RequestKind
    op: OperationType
    address: Optional[BlockAddress | BucketAddress] = None
    data: Optional[bytes] = None
    arrival_time: Optional[float] = None
    issued_time: Optional[float] = None
    tag: Optional[str] = None


@dataclass(slots=True)
class Response:
    request_id: int
    data: Optional[bytes]
    success: bool = True
    error: Optional[str] = None


@dataclass(slots=True)
class DataBlock:
    block_id: Optional[int]
    payload: bytes
    is_dummy: bool = False
    leaf: Optional[int] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def clone(self) -> "DataBlock":
        return DataBlock(
            block_id=self.block_id,
            payload=bytes(self.payload),
            is_dummy=self.is_dummy,
            leaf=self.leaf,
            metadata=dict(self.metadata),
        )

    @property
    def payload_size(self) -> int:
        return len(self.payload)


@dataclass(slots=True)
class Bucket:
    address: BucketAddress
    blocks: list[DataBlock] = field(default_factory=list)

    def clone(self) -> "Bucket":
        return Bucket(
            address=self.address,
            blocks=[block.clone() for block in self.blocks],
        )

    def non_dummy_blocks(self) -> list[DataBlock]:
        return [block for block in self.blocks if not block.is_dummy]

    def dummy_count(self) -> int:
        return sum(1 for block in self.blocks if block.is_dummy)
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from src.common.metrics import AccessResult
from src.common.types import Bucket, Request

# minimal bucket-store interface used by tree backends
class AbstractBucketStore(ABC):
    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def exists(self, flat_index: int) -> bool:
        pass

    @abstractmethod
    def read(self, flat_index: int) -> Optional[Bucket]:
        pass

    @abstractmethod
    def write(self, flat_index: int, bucket: Bucket) -> None:
        pass


class AbstractORAM(ABC):
    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def access(self, request: Request) -> AccessResult:
        pass

    def tick(self, now: float) -> Optional[AccessResult]:
        return None

    def enqueue(self, request: Request) -> None:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement enqueue()."
        )
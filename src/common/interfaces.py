from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from src.common.metrics import AccessResult
from src.common.types import Bucket, Request


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
        """
        Execute a single protocol-visible access.
        Path/Ring/DirectStore mainly use this entry.
        AtomORAM can also expose it for direct protocol-level tests.
        """
        pass

    def tick(self, now: float) -> Optional[AccessResult]:
        """
        Timer-driven service hook.
        Path/Ring/DirectStore may simply return None.
        AtomORAM will override this.
        """
        return None

    def enqueue(self, request: Request) -> None:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement enqueue()."
        )
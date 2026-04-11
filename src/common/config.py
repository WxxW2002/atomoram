from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class StorageConfig:
    block_size: int = 4096
    bucket_size: int = 8
    tree_height: int = 20
    use_file_backend: bool = False
    data_dir: str = "data/tmp"


@dataclass(slots=True)
class NetworkConfig:
    down_bw_bytes_per_sec: float = 10 * 2**20
    up_bw_bytes_per_sec: float = 10 * 2**20
    rtt_sec: float = 0.010


@dataclass(slots=True)
class CryptoConfig:
    enc_bytes_per_sec: float = 2e9
    dec_bytes_per_sec: float = 2e9


@dataclass(slots=True)
class ServerIOConfig:
    bucket_read_sec: float = 2e-3
    bucket_write_sec: float = 2e-3


@dataclass(slots=True)
class RingConfig:
    s_num: int = 12
    a_num: int = 8


@dataclass(slots=True)
class AtomConfig:
    lambda1: float = 2.0
    tick_interval_sec: float = 0.002
    queue_limit: int = 100000


@dataclass(slots=True)
class ExperimentConfig:
    storage: StorageConfig = field(default_factory=StorageConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    crypto: CryptoConfig = field(default_factory=CryptoConfig)
    server_io: ServerIOConfig = field(default_factory=ServerIOConfig)
    ring: RingConfig = field(default_factory=RingConfig)
    atom: AtomConfig = field(default_factory=AtomConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ExperimentConfig":
        return cls(
            storage=StorageConfig(**raw.get("storage", {})),
            network=NetworkConfig(**raw.get("network", {})),
            crypto=CryptoConfig(**raw.get("crypto", {})),
            server_io=ServerIOConfig(**raw.get("server_io", {})),
            ring=RingConfig(**raw.get("ring", {})),
            atom=AtomConfig(**raw.get("atom", {})),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        yaml_path = Path(path)
        with yaml_path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return cls.from_dict(raw)
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.config import ExperimentConfig, RingConfig, ServerIOConfig, StorageConfig
from src.common.latency_model import LatencyModel
from src.common.types import BlockAddress, OperationType, Request, RequestKind
from src.protocols.direct_store import DirectStore
from src.protocols.path_oram import PathORAM
from src.protocols.ring_oram import RingORAM


def print_summary(name: str, phase: str, result, estimate) -> None:
    print(f"[{name}][{phase}]")
    print(
        f"  online_bucket_reads={result.metrics.online_bucket_reads}, "
        f"online_bucket_writes={result.metrics.online_bucket_writes}, "
        f"online_rtt={result.metrics.online_rtt}"
    )
    print(
        f"  online_latency={estimate.online_latency:.6f}s, "
        f"offline_latency={estimate.offline_latency:.6f}s, "
        f"total_latency={estimate.total_latency:.6f}s"
    )
    print(
        f"  network_time={result.timing.network_time:.6f}s, "
        f"server_io_time={result.timing.server_io_time:.6f}s, "
        f"crypto_time={result.timing.crypto_time:.6f}s"
    )
    print()


def main() -> None:
    storage_cfg = StorageConfig(
        block_size=32,
        bucket_size=4,
        tree_height=4,
        use_file_backend=False,
        data_dir="data/tmp",
    )

    exp_cfg = ExperimentConfig(
        storage=storage_cfg,
        server_io=ServerIOConfig(
            bucket_read_sec=50e-6,
            bucket_write_sec=50e-6,
        ),
    )
    latency_model = LatencyModel(exp_cfg)

    direct = DirectStore(config=storage_cfg)
    path = PathORAM(config=storage_cfg, rng_seed=7)
    ring = RingORAM(storage_config=storage_cfg,ring_config=RingConfig(s_num=12, a_num=8),rng_seed=7)

    protocols = [
        ("DirectStore", direct),
        ("PathORAM", path),
        ("RingORAM", ring),
    ]

    for name, proto in protocols:
        write_req = Request(
            request_id=1,
            kind=RequestKind.REAL,
            op=OperationType.WRITE,
            address=BlockAddress(logical_id=3),
            data=b"hello",
            arrival_time=0.0,
        )
        write_result = proto.access(write_req)
        write_estimate = latency_model.annotate(write_result)
        print_summary(name, "WRITE", write_result, write_estimate)

        read_req = Request(
            request_id=2,
            kind=RequestKind.REAL,
            op=OperationType.READ,
            address=BlockAddress(logical_id=3),
            arrival_time=1.0,
        )
        read_result = proto.access(read_req)
        read_estimate = latency_model.annotate(read_result)
        print_summary(name, "READ", read_result, read_estimate)

        assert read_result.data == b"hello", f"{name} read-back failed."


if __name__ == "__main__":
    main()
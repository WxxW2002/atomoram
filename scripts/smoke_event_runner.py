from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.config import ExperimentConfig
from src.common.latency_model import LatencyModel
from src.protocols.atom_oram import AtomORAM
from src.sim.atom_event_runner import AtomEventRunner
from src.traces.synthetic import generate_constant_interval_trace


def main() -> None:
    cfg = ExperimentConfig.from_yaml(REPO_ROOT / "configs" / "default.yaml")

    protocol = AtomORAM(
        storage_config=cfg.storage,
        atom_config=cfg.atom,
        rng_seed=7,
    )
    runner = AtomEventRunner(
        latency_model=LatencyModel(cfg),
        atom_config=cfg.atom,
    )

    records = generate_constant_interval_trace(
        num_requests=5,
        address_space=32,
        interval_sec=cfg.atom.tick_interval_sec * 0.4,
        read_ratio=1.0,
        seed=1,
        request_size_bytes=cfg.storage.block_size,
    )

    df = runner.run(
        protocol=protocol,
        records=records,
        block_size=cfg.storage.block_size,
    )

    print(df[
        [
            "tick_index",
            "service_kind",
            "trace_id",
            "arrival_time",
            "queueing_delay",
            "end_to_end_latency",
            "online_latency",
            "offline_latency",
            "fallback_flag",
            "queue_length_before",
            "queue_length_after",
        ]
    ].to_string(index=False))


if __name__ == "__main__":
    main()
import os
import pandas as pd
import matplotlib.pyplot as plt
from src.sim.atom_event_runner import AtomEventRunner
from src.protocols.atom_oram import AtomORAM
from src.common.config import ExperimentConfig
from src.common.latency_model import LatencyModel
from src.traces.schema import TraceRecord
from src.common.types import OperationType
from src.common.exp_utils import instantiate_protocol, prepare_storage_config

os.makedirs('artifacts/figs', exist_ok=True)
os.makedirs('artifacts/csv', exist_ok=True)
plt.rcParams.update({'font.family': 'serif', 'font.size': 12, 'pdf.fonttype': 42, 'axes.linewidth': 1.2})

def generate_burst_trace(
    t_virt,
    required_virtual_ticks,
    block_size,
    burst_size=100,
    num_bursts=3,
    burst_gap_factor=0.1,
    idle_margin_sec=0.5,
    rtt_sec=0.02,
    bucket_read_sec=0.005,
    bucket_write_sec=0.005,
):
    records = []
    current_time = 0.0
    trace_id = 0

    base_gap = t_virt * required_virtual_ticks
    burst_gap = base_gap * burst_gap_factor
    tail = 2 * rtt_sec + 2 * bucket_read_sec + 2 * bucket_write_sec

    recovery_gap = tail + max(0, burst_size - 1) * max(0.0, base_gap - burst_gap)
    idle_gap = recovery_gap + idle_margin_sec

    for burst_idx in range(num_bursts):
        for _ in range(burst_size):
            records.append(
                TraceRecord(
                    trace_id=trace_id,
                    timestamp=current_time,
                    op=OperationType.READ,
                    logical_id=trace_id % 1000,
                    size_bytes=block_size,
                    source=f"b{burst_idx + 1}",
                    original_index=trace_id,
                    original_offset=0,
                    request_group=burst_idx,
                )
            )
            current_time += burst_gap
            trace_id += 1

        if burst_idx < num_bursts - 1:
            current_time += idle_gap

    return records

def run_e5():
    cfg = ExperimentConfig.load_default()
    L = cfg.storage.tree_height
    t_virt = cfg.atom.tick_interval_sec
    lambda_1 = cfg.atom.lambda1
    block_size = cfg.storage.block_size
    required_virtual_ticks = int(lambda_1 * L)

    records = generate_burst_trace(
        t_virt=t_virt,
        required_virtual_ticks=required_virtual_ticks,
        block_size=block_size,
        burst_size=100,
        num_bursts=3,
        burst_gap_factor=0.25,
        idle_margin_sec=0.25,
        rtt_sec=cfg.network.rtt_sec,
        bucket_read_sec=cfg.server_io.bucket_read_sec,
        bucket_write_sec=cfg.server_io.bucket_write_sec,
    )

    storage_cfg = prepare_storage_config(
        cfg.storage,
        exp_name="e5",
        protocol_name="AtomORAM",
        run_tag="burst_recovery",
    )
    protocol = instantiate_protocol(AtomORAM, cfg, storage_cfg, rng_seed=0)

    runner = AtomEventRunner(latency_model=LatencyModel(config=cfg), atom_config=cfg.atom)
    df = runner.run(
        protocol=protocol,
        records=records,
        block_size=block_size,
        required_virtual_ticks=required_virtual_ticks,
        max_idle_ticks_after_last_arrival=0,
        record_virtuals=False,
    )

    real_df = df[df["service_kind"] == "real"][["arrival_time", "queue_length_after"]]
    real_df.to_csv("artifacts/csv/E5_burst_recovery.csv", index=False)

    plt.figure(figsize=(7, 4))
    plt.plot(real_df["arrival_time"], real_df["queue_length_after"], marker=".", linestyle="-", color="tab:blue", linewidth=2)
    plt.xlabel("Time (s)")
    plt.ylabel("Queue Length")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig("artifacts/figs/E5_burst_recovery.pdf", format="pdf", bbox_inches="tight")

if __name__ == '__main__':
    run_e5()
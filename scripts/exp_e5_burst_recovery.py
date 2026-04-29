import os
import random
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
    *,
    base_gap_sec,
    block_size,
    burst_sizes,
    expected_service_gap_sec,
    service_tail_sec,
    drain_safety,
    seed=7,
    burst_interarrival_range=(0.04, 0.28),
):
    rng = random.Random(seed)

    records = []
    burst_meta = []

    current_time = 0.0
    trace_id = 0

    for burst_idx, burst_size in enumerate(burst_sizes):
        burst_start = current_time

        for j in range(burst_size):
            if j > 0:
                current_time += base_gap_sec * rng.uniform(
                    burst_interarrival_range[0],
                    burst_interarrival_range[1],
                )

            records.append(
                TraceRecord(
                    trace_id=trace_id,
                    timestamp=current_time,
                    op=OperationType.READ if trace_id % 3 else OperationType.WRITE,
                    logical_id=(trace_id * 17) % 10000,
                    size_bytes=block_size,
                    source=f"burst_{burst_idx + 1}_n{burst_size}",
                    original_index=trace_id,
                    original_offset=0,
                    request_group=burst_idx,
                )
            )
            trace_id += 1

        burst_end = current_time

        burst_meta.append(
            {
                "Burst": burst_idx + 1,
                "Burst_Size": burst_size,
                "Burst_Start_Time": burst_start,
                "Burst_End_Time": burst_end,
            }
        )

        if burst_idx < len(burst_sizes) - 1:
            drain_gap = drain_safety * (
                burst_size * expected_service_gap_sec + service_tail_sec
            )
            drain_gap *= rng.uniform(0.9, 1.1)
            current_time += drain_gap

    return records, pd.DataFrame(burst_meta)

def run_atom(records, cfg, block_size, *, run_tag):
    storage_cfg = prepare_storage_config(
        cfg.storage,
        exp_name="e5",
        protocol_name="AtomORAM",
        run_tag=run_tag,
    )
    protocol = instantiate_protocol(AtomORAM, cfg, storage_cfg, rng_seed=0)

    runner = AtomEventRunner(
        latency_model=LatencyModel(config=cfg),
        atom_config=cfg.atom,
    )

    df = runner.run(
        protocol=protocol,
        records=records,
        block_size=block_size,
        max_idle_ticks_after_last_arrival=0,
        record_virtuals=False,
    )

    return df

def build_queue_timeline(real_df):
    events = []

    for row in real_df.itertuples(index=False):
        events.append((float(row.arrival_time), 0, +1))
        events.append((float(row.service_start_time), 1, -1))

    events.sort(key=lambda x: (x[0], x[1]))

    q = 0
    timeline = []

    if events:
        timeline.append({"time": events[0][0], "queue_length": 0})

    for t, _, delta in events:
        timeline.append({"time": t, "queue_length": q})
        q += delta
        q = max(q, 0)
        timeline.append({"time": t, "queue_length": q})

    return pd.DataFrame(timeline)

def previous_burst_drained_before_next(queue_df, burst_start_times):
    for start_time in burst_start_times[1:]:
        before = queue_df[queue_df["time"] <= start_time]

        if before.empty:
            return False

        q_before_next_burst = int(before.iloc[-1]["queue_length"])

        if q_before_next_burst != 0:
            return False

    return True

def run_e5():
    cfg = ExperimentConfig.load_default()

    L = cfg.storage.tree_height
    t_virt = cfg.atom.tick_interval_sec
    block_size = cfg.storage.block_size

    # This is only a workload scale estimate
    # the runner uses the compensation scheduler.
    base_gap_sec = L * t_virt

    burst_sizes = [10, 30, 50, 70, 90]

    # compensation is approximately (L + 1) timer ticks.
    expected_service_gap_sec = (L + 1) * t_virt

    service_tail_sec = (
        2 * cfg.network.rtt_sec
        + 2 * cfg.server_io.bucket_read_sec
        + 2 * cfg.server_io.bucket_write_sec
    )

    final_df = None
    final_records = None
    final_burst_meta = None
    final_queue_df = None

    # Start with a relatively tight gap. 
    drain_safety = 0.85

    for attempt in range(10):
        records, burst_meta = generate_burst_trace(
            base_gap_sec=base_gap_sec,
            block_size=block_size,
            burst_sizes=burst_sizes,
            expected_service_gap_sec=expected_service_gap_sec,
            service_tail_sec=service_tail_sec,
            drain_safety=drain_safety,
            seed=7,
            burst_interarrival_range=(0.04, 0.28),
        )

        df = run_atom(
            records=records,
            cfg=cfg,
            block_size=block_size,
            run_tag=f"burst_recovery_attempt_{attempt}",
        )

        real_df = df[df["service_kind"] == "real"].copy()
        queue_df = build_queue_timeline(real_df)

        burst_start_times = burst_meta["Burst_Start_Time"].tolist()

        if previous_burst_drained_before_next(queue_df, burst_start_times):
            final_df = df
            final_records = records
            final_burst_meta = burst_meta
            final_queue_df = queue_df
            break

        drain_safety *= 1.15

    if (
        final_df is None
        or final_records is None
        or final_burst_meta is None
        or final_queue_df is None
    ):
        raise RuntimeError(
            "Could not construct an E5 trace where each previous burst drains "
        )

    real_df = final_df[final_df["service_kind"] == "real"].copy()

    real_df[
        [
            "arrival_time",
            "service_start_time",
            "response_time",
            "request_group",
            "source",
            "queue_length_before",
            "queue_length_after",
            "queueing_delay",
            "end_to_end_latency",
        ]
    ].to_csv("artifacts/csv/E5_burst_recovery.csv", index=False)

    final_burst_meta.to_csv("artifacts/csv/E5_burst_metadata.csv", index=False)
    final_queue_df.to_csv("artifacts/csv/E5_queue_timeline.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.plot(final_queue_df["time"], final_queue_df["queue_length"], drawstyle="steps-post", linewidth=1.8)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Queue Length")
    ax.grid(True, linestyle="--", alpha=0.5)

    fig.savefig("artifacts/figs/E5_burst_recovery.pdf", format="pdf", bbox_inches="tight")

if __name__ == '__main__':
    run_e5()
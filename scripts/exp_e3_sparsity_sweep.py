import os
import numpy as np
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

def generate_synthetic_trace(alpha, reference_gap_sec, block_size, num_reqs=800):
    """Generate a trace with alpha = Delta t_real / (L * t_virt)."""
    dt = alpha * reference_gap_sec

    return [
        TraceRecord(
            trace_id=i,
            timestamp=i * dt,
            op=OperationType.WRITE if i % 2 == 0 else OperationType.READ,
            logical_id=i % 1000,
            size_bytes=block_size,
            source="synthetic",
            original_index=i,
            original_offset=0,
            request_group=0,
        )
        for i in range(num_reqs)
    ]

def run_e3():
    cfg = ExperimentConfig.load_default()
    L = cfg.storage.tree_height
    t_virt = cfg.atom.tick_interval_sec
    block_size = cfg.storage.block_size

    reference_gap_sec = L * t_virt

    alphas = [0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0]
    data_out = []

    for alpha in alphas:
        records = generate_synthetic_trace(
            alpha=alpha,
            reference_gap_sec=reference_gap_sec,
            block_size=block_size,
            num_reqs=800,
        )

        storage_cfg = prepare_storage_config(
            cfg.storage,
            exp_name="e3",
            protocol_name="AtomORAM",
            run_tag=f"alpha_{alpha}",
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

        real_df = df[df["service_kind"] == "real"]

        data_out.append(
            {
                "Alpha": alpha,
                "Load_Intensity": 1.0 / alpha,
                "Interarrival_Sec": alpha * reference_gap_sec,
                "Mean_Latency": real_df["end_to_end_latency"].mean(),
                "P95_Latency": real_df["end_to_end_latency"].quantile(0.95),
                "P99_Latency": real_df["end_to_end_latency"].quantile(0.99),
                "Mean_Queue": real_df["queue_length_after"].mean(),
                "Max_Queue": real_df["queue_length_after"].max(),
                "Compensation_Virtual_Ticks": runner.required_virtual_ticks_executed,
            }
        )

    df_out = pd.DataFrame(data_out)
    df_out.to_csv("artifacts/csv/E3_sparsity_sweep.csv", index=False)

    df_plot = df_out.sort_values("Load_Intensity")

    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.set_xlabel(
        r"Workload Intensity $\rho = 1/\alpha = L \cdot t_{virt} / \Delta t_{real}$"
    )
    ax1.set_ylabel("End-to-End Latency (s)")

    line1 = ax1.plot(
        df_plot["Load_Intensity"],
        df_plot["Mean_Latency"],
        marker="o",
        label="Mean Latency",
        linewidth=2,
        color="tab:orange",
        linestyle="--",
        alpha=0.9,
    )
    line2 = ax1.plot(
        df_plot["Load_Intensity"],
        df_plot["P95_Latency"],
        marker="^",
        label="P95 Latency",
        linewidth=2,
        alpha=0.9,
        linestyle="--",
        color="tab:orange",
    )

    boundary = ax1.axvline(
        x=1.0,
        linestyle=":",
        label=r"Boundary ($\rho=1$)",
    )

    ax2 = ax1.twinx()
    ax2.set_ylabel("Max Queue Length")

    line3 = ax2.plot(
        df_plot["Load_Intensity"],
        df_plot["Max_Queue"],
        marker="s",
        label="Max Queue",
        linewidth=2,
        color="tab:blue",
        alpha=0.7,
    )

    lines = line1 + line2 + line3 + [boundary]
    ax1.legend(
        lines,
        [l.get_label() for l in lines],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=4,
        frameon=False,
    )

    ax1.grid(True, linestyle="--", alpha=0.5)

    plt.savefig(
        "artifacts/figs/E3_sparsity_sweep.pdf",
        format="pdf",
        bbox_inches="tight",
    )

if __name__ == '__main__':
    run_e3()

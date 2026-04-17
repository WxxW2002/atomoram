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

def generate_synthetic_trace(alpha, base_gap, block_size, num_reqs=800):
    dt = alpha * base_gap
    return [TraceRecord(trace_id=i, timestamp=i*dt, op=OperationType.WRITE if i % 2 == 0 else OperationType.READ, logical_id=i%1000, size_bytes=block_size, source='synthetic', original_index=i, original_offset=0, request_group=0) for i in range(num_reqs)]

def run_e3():
    cfg = ExperimentConfig.load_default()
    L = cfg.storage.tree_height
    t_virt = cfg.atom.tick_interval_sec
    lambda_1 = cfg.atom.lambda1
    block_size = cfg.storage.block_size

    required_virtual_ticks = int(lambda_1 * L)
    base_gap = required_virtual_ticks * t_virt
    alphas = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0]
    data_out = []

    for alpha in alphas:
        records = generate_synthetic_trace(alpha, base_gap, block_size, num_reqs=800)

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
            required_virtual_ticks=required_virtual_ticks,
            max_idle_ticks_after_last_arrival=0,
            record_virtuals=False,
        )
        real_df = df[df["service_kind"] == "real"]
        data_out.append({
            "Alpha": alpha,
            "Mean_Latency": real_df["end_to_end_latency"].mean(),
            "P99_Latency": real_df["end_to_end_latency"].quantile(0.99),
            "Max_Queue": real_df["queue_length_after"].max(),
        })

    df_out = pd.DataFrame(data_out)
    df_out.to_csv("artifacts/csv/E3_sparsity_sweep.csv", index=False)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    color1, color2 = "tab:blue", "tab:orange"
    ax1.set_xlabel(r"Sparsity Ratio $\beta = \Delta t_{real} / (\lambda_1 \cdot L \cdot t_{virt})$")
    ax1.set_ylabel("End-to-End Latency (s)", color=color1)
    line1 = ax1.plot(df_out["Alpha"], df_out["Mean_Latency"], marker="o", color=color1, label="Mean Latency", linewidth=2)
    line2 = ax1.plot(df_out["Alpha"], df_out["P99_Latency"], marker="^", color="tab:cyan", label="P99 Latency", linewidth=2, linestyle="--")
    ax1.tick_params(axis="y", labelcolor=color1)
    line_bound = ax1.axvline(x=1.0, color="r", linestyle=":", label=r"Boundary ($\beta=1$)")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Max Queue Length", color=color2)
    line3 = ax2.plot(df_out["Alpha"], df_out["Max_Queue"], marker="s", color=color2, label="Max Queue", linewidth=2, alpha=0.5)
    ax2.tick_params(axis="y", labelcolor=color2)

    lines = line1 + line2 + line3 + [line_bound]
    ax1.legend(lines, [l.get_label() for l in lines], loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=4, frameon=False)
    ax1.grid(True, linestyle="--", alpha=0.5)
    plt.savefig("artifacts/figs/E3_sparsity_sweep.pdf", format="pdf", bbox_inches="tight")

if __name__ == '__main__':
    run_e3()
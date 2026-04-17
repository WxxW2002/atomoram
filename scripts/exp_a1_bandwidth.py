import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.sim.atom_event_runner import AtomEventRunner
from src.protocols.atom_oram import AtomORAM
from src.protocols.path_oram import PathORAM
from src.protocols.ring_oram import RingORAM
from src.common.config import ExperimentConfig
from src.common.latency_model import LatencyModel
from src.traces.schema import TraceRecord
from src.common.types import OperationType, Request, RequestKind, BlockAddress
from src.common.exp_utils import instantiate_protocol, prepare_storage_config

os.makedirs('artifacts/figs', exist_ok=True)
os.makedirs('artifacts/csv', exist_ok=True)
plt.rcParams.update({'font.family': 'serif', 'font.size': 12, 'pdf.fonttype': 42, 'axes.linewidth': 1.2})

def load_trace(file_path, limit=1000):
    df = pd.read_csv(file_path, nrows=limit)
    return [TraceRecord(trace_id=int(row['trace_id']), timestamp=float(row['timestamp']), op=OperationType.WRITE if row['op'] == 'W' else OperationType.READ, logical_id=int(row['logical_id']), size_bytes=int(row['size_bytes']), source='trace', original_index=int(row['trace_id']), original_offset=0, request_group=0) for _, row in df.iterrows()]

def run_baseline_bw(protocol_class, cfg, records, block_size, run_tag):
    storage_cfg = prepare_storage_config(
        cfg.storage,
        exp_name="a1",
        protocol_name=protocol_class.__name__,
        run_tag=run_tag,
    )
    protocol = instantiate_protocol(protocol_class, cfg, storage_cfg, rng_seed=0)

    total_bytes = 0
    for rec in records:
        req = Request(
            request_id=rec.trace_id,
            kind=RequestKind.REAL,
            op=rec.op,
            address=BlockAddress(logical_id=rec.logical_id),
            data=b"\x00" * block_size if rec.op == OperationType.WRITE else None,
            arrival_time=rec.timestamp,
            issued_time=0.0,
            tag="baseline",
        )
        res = protocol.access(req)
        total_bytes += res.metrics.total_bytes_down + res.metrics.total_bytes_up
    return total_bytes

def run_a1():
    cfg = ExperimentConfig.load_default()
    L = cfg.storage.tree_height
    lambda_1 = cfg.atom.lambda1
    block_size = cfg.storage.block_size
    required_virtual_ticks = int(lambda_1 * L)

    traces = {
        "MSRC": "data/processed/msrc_src1_0_trace.csv",
        "AliCloud": "data/processed/alicloud_device32_trace.csv",
    }
    plot_data = []

    for name, path in traces.items():
        records = load_trace(path, limit=1000)

        bw_path = run_baseline_bw(PathORAM, cfg, records, block_size, run_tag=f"{name}_path")
        bw_ring = run_baseline_bw(RingORAM, cfg, records, block_size, run_tag=f"{name}_ring")

        atom_storage_cfg = prepare_storage_config(
            cfg.storage,
            exp_name="a1",
            protocol_name="AtomORAM",
            run_tag=f"{name}_atom",
        )
        atom_protocol = instantiate_protocol(AtomORAM, cfg, atom_storage_cfg, rng_seed=0)

        runner = AtomEventRunner(latency_model=LatencyModel(config=cfg), atom_config=cfg.atom)
        df_atom = runner.run(
            protocol=atom_protocol,
            records=records,
            block_size=block_size,
            required_virtual_ticks=required_virtual_ticks,
            max_idle_ticks_after_last_arrival=0,
            record_virtuals=False,
        )

        real_df = df_atom[df_atom['service_kind'] == 'real']
        bw_atom_real = (
            real_df['online_bytes_down'].sum()
            + real_df['online_bytes_up'].sum()
            + real_df['offline_bytes_down'].sum()
            + real_df['offline_bytes_up'].sum()
        )

        bw_atom_required_virtual = (
            runner.required_virtual_bytes_down
            + runner.required_virtual_bytes_up
        )

        bw_atom_total = bw_atom_real + bw_atom_required_virtual

        for prot, bw in [("Path ORAM", bw_path), ("Ring ORAM", bw_ring), ("AtomORAM", bw_atom_total)]:
            plot_data.append({
                "Trace": name,
                "Protocol": prot,
                "Total_Bandwidth_MB": bw / (1024 * 1024),
            })

    df_plot = pd.DataFrame(plot_data)
    df_plot.to_csv("artifacts/csv/A1_bandwidth.csv", index=False)

    df_single_trace = df_plot[df_plot["Trace"] == "MSRC"]

    fig, ax = plt.subplots(figsize=(6, 5))
    protocols = df_single_trace["Protocol"].values
    y_vals = df_single_trace["Total_Bandwidth_MB"].values
    
    x = np.arange(len(protocols))
    width = 0.5

    colors = ['C0', 'C1', 'C2']
    ax.bar(x, y_vals, width, color=colors)

    ax.set_ylabel("Amortized Total Bandwidth (MB)")
    ax.set_xticks(x)
    ax.set_xticklabels(protocols)
    
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.set_yscale("log")
    plt.savefig("artifacts/figs/A1_bandwidth.pdf", format="pdf", bbox_inches="tight")

if __name__ == '__main__':
    run_a1()
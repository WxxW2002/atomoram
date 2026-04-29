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

# load the real-trace subset
def load_trace(file_path, limit=2000):
    from src.traces.schema import normalize_operation

    df = pd.read_csv(file_path, nrows=limit)
    records = []

    for _, row in df.iterrows():
        records.append(
            TraceRecord(
                trace_id=int(row["trace_id"]),
                timestamp=float(row["timestamp"]),
                op=normalize_operation(row["op"]),
                logical_id=int(row["logical_id"]),
                size_bytes=int(row["size_bytes"]),
                source="real_trace",
                original_index=int(row["trace_id"]),
                original_offset=0,
                request_group=0,
            )
        )

    return records

# run protocols on the real trace
def run_baseline(protocol_class, cfg, records, latency_model, block_size, run_tag):
    storage_cfg = prepare_storage_config(
        cfg.storage,
        exp_name="e4",
        protocol_name=protocol_class.__name__,
        run_tag=run_tag,
    )
    protocol = instantiate_protocol(protocol_class, cfg, storage_cfg, rng_seed=0)

    current_time = 0.0
    latencies = []

    for rec in sorted(records, key=lambda r: (r.timestamp, r.trace_id)):
        start_service = max(current_time, rec.timestamp)
        queueing_delay = start_service - rec.timestamp

        req = Request(
            request_id=rec.trace_id,
            kind=RequestKind.REAL,
            op=rec.op,
            address=BlockAddress(logical_id=rec.logical_id),
            data=b"\x00" * block_size if rec.op == OperationType.WRITE else None,
            arrival_time=rec.timestamp,
            issued_time=start_service,
            tag="baseline",
        )

        res = protocol.access(req)
        res.timing.service_start_time = start_service

        est = latency_model.annotate(res, queueing_delay=queueing_delay)

        service_time_excluding_queue = est.total_latency - est.queueing_delay
        current_time = start_service + service_time_excluding_queue

        latencies.append(est.total_latency)

    return latencies

def run_e4():
    cfg = ExperimentConfig.load_default()
    block_size = cfg.storage.block_size

    traces = {
        "MSRC": "data/processed/msrc_src1_0_trace.csv",
        "AliCloud": "data/processed/alicloud_device32_trace.csv",
    }
    latency_model = LatencyModel(config=cfg)
    plot_data = []

    for name, path in traces.items():
        records = load_trace(path, limit=2000)

        path_lats = run_baseline(PathORAM, cfg, records, latency_model, block_size, run_tag=f"{name}_path")
        ring_lats = run_baseline(RingORAM, cfg, records, latency_model, block_size, run_tag=f"{name}_ring")

        atom_storage_cfg = prepare_storage_config(cfg.storage, exp_name="e4", protocol_name="AtomORAM", run_tag=f"{name}_atom")
        atom_protocol = instantiate_protocol(AtomORAM, cfg, atom_storage_cfg, rng_seed=0)

        runner = AtomEventRunner(latency_model=latency_model, atom_config=cfg.atom)
        df_atom = runner.run(
            protocol=atom_protocol,
            records=records,
            block_size=block_size,
            max_idle_ticks_after_last_arrival=0,
            record_virtuals=False,
        )
        atom_lats = df_atom[df_atom["service_kind"] == "real"]["end_to_end_latency"].values

        for prot, lats in [("Path ORAM", path_lats), ("Ring ORAM", ring_lats), ("AtomORAM", atom_lats)]:
            plot_data.append({
                "Trace": name,
                "Protocol": prot,
                "P5": np.percentile(lats, 5),
                "P50": np.percentile(lats, 50),
                "P95": np.percentile(lats, 95),
                "Mean": np.mean(lats),
            })

    df_plot = pd.DataFrame(plot_data)
    df_plot.to_csv("artifacts/csv/E4_real_trace_comparison.csv", index=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    traces_labels = df_plot["Trace"].unique()
    x = np.arange(len(traces_labels))
    width = 0.25

    for i, prot in enumerate(df_plot["Protocol"].unique()):
        y_vals_p5 = df_plot[df_plot["Protocol"] == prot]["P5"]
        ax1.bar(x + (i - 1) * width, y_vals_p5, width, label=prot)

        y_vals_p95 = df_plot[df_plot["Protocol"] == prot]["P95"]
        ax2.bar(x + (i - 1) * width, y_vals_p95, width, label=prot)

    ax1.set_ylabel("P5 End-to-End Latency (s)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(traces_labels)
    ax1.set_xlabel("(a) P5 Latency Comparison", labelpad=15, fontsize=14) 
    ax1.grid(axis="y", linestyle="--", alpha=0.7)
    ax1.set_yscale("log")

    ax2.set_ylabel("P95 End-to-End Latency (s)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(traces_labels)
    ax2.set_xlabel("(b) P95 Latency Comparison", labelpad=15, fontsize=14)
    ax2.grid(axis="y", linestyle="--", alpha=0.7)
    ax2.set_yscale("log")

    ax1.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False)
    ax2.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False)

    plt.tight_layout()
    plt.savefig("artifacts/figs/E4_real_trace_comparison.pdf", format="pdf", bbox_inches="tight")

if __name__ == '__main__':
    run_e4()
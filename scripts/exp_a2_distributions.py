import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import MaxNLocator
from src.sim.atom_event_runner import AtomEventRunner
from src.protocols.atom_oram import AtomORAM
from src.common.config import ExperimentConfig
from src.common.latency_model import LatencyModel
from src.traces.schema import TraceRecord
from src.common.types import OperationType, Request, RequestKind, BlockAddress
from src.common.exp_utils import instantiate_protocol, prepare_storage_config

os.makedirs('artifacts/figs', exist_ok=True)
os.makedirs('artifacts/csv', exist_ok=True)
plt.rcParams.update({'font.family': 'serif', 'font.size': 12, 'pdf.fonttype': 42, 'axes.linewidth': 1.2})

def load_trace(file_path, limit=5000):
    df = pd.read_csv(file_path, nrows=limit)
    return [TraceRecord(trace_id=int(row['trace_id']), timestamp=float(row['timestamp']), op=OperationType.WRITE if row['op'] == 'W' else OperationType.READ, logical_id=int(row['logical_id']), size_bytes=int(row['size_bytes']), source='real_trace', original_index=int(row['trace_id']), original_offset=0, request_group=0) for _, row in df.iterrows()]

def warmup_protocol(protocol, warmup_virtual_ticks, block_size, num_blocks=50000):
    for i in range(num_blocks):
        fill_byte = (i % 251) + 1
        protocol.access(
            Request(
                request_id=-(i + 1),
                kind=RequestKind.REAL,
                op=OperationType.WRITE,
                address=BlockAddress(logical_id=i),
                data=bytes([fill_byte]) * block_size,
                arrival_time=0.0,
                issued_time=0.0,
                tag="warmup",
            )
        )
        for j in range(warmup_virtual_ticks):
            protocol.access(
                Request(
                    request_id=-(i * warmup_virtual_ticks + j + 1000000),
                    kind=RequestKind.VIRTUAL,
                    op=OperationType.READ,
                    address=None,
                    data=None,
                    arrival_time=0.0,
                    issued_time=0.0,
                    tag="warmup_virtual",
                )
            )

def run_a2():
    cfg = ExperimentConfig.load_default()
    L = cfg.storage.tree_height
    lambda_1 = cfg.atom.lambda1
    block_size = cfg.storage.block_size

    warmup_virtual_ticks = L
    required_virtual_ticks = int(lambda_1 * L)

    traces = {
        "MSRC": "data/processed/msrc_src1_0_trace.csv",
        "AliCloud": "data/processed/alicloud_device32_trace.csv",
        "Google": "data/processed/google_cluster2_20240118_trace.csv",
    }

    stash_data, queue_data = {}, {}

    for name, path in traces.items():
        records = load_trace(path, limit=5000)

        storage_cfg = prepare_storage_config(
            cfg.storage,
            exp_name="a2",
            protocol_name="AtomORAM",
            run_tag=f"{name}_steady_state",
        )
        protocol = instantiate_protocol(AtomORAM, cfg, storage_cfg, rng_seed=0)

        warmup_protocol(protocol, warmup_virtual_ticks, block_size, num_blocks=50000)

        runner = AtomEventRunner(latency_model=LatencyModel(config=cfg), atom_config=cfg.atom)
        df = runner.run(
            protocol=protocol,
            records=records,
            block_size=block_size,
            required_virtual_ticks=required_virtual_ticks,
            max_idle_ticks_after_last_arrival=0,
            record_virtuals=False,
        )

        stash_data[name] = np.sort(df["stash_size_after"].values)
        queue_data[name] = np.sort(df["queue_length_after"].values)
        yvals = np.arange(len(stash_data[name])) / float(len(stash_data[name]) - 1)
        slug_map = {
            "MSRC": "msrc",
            "AliCloud": "alicloud",
            "Google": "google",
        }
        pd.DataFrame({
            "Stash_Size": stash_data[name],
            "Queue_Length": queue_data[name],
            "CDF": yvals,
        }).to_csv(f"artifacts/csv/A2_{slug_map[name]}_Distribution.csv", index=False)

    for data_dict, xlabel, is_log, out_name in [
        (stash_data, "Stash Size (Blocks)", False, "FigA2_Stash_Distribution.pdf"),
        (queue_data, "Queue Length", True, "FigA2_Queue_Distribution.pdf"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 5))
        style_map = {
            "MSRC": {"linestyle": "--", "zorder": 3, "alpha": 1.0},
            "AliCloud": {"linestyle": "-", "zorder": 2, "alpha": 0.85},
            "Google": {"linestyle": "-.", "zorder": 4, "alpha": 0.95},
        }
        plot_order = ["MSRC", "AliCloud", "Google"]

        max_x = 0
        for name in plot_order:
            data = data_dict[name]
            yvals = np.arange(len(data)) / float(len(data) - 1)
            ax.plot(
                data,
                yvals,
                label=name,
                linewidth=2,
                linestyle=style_map[name]["linestyle"],
                zorder=style_map[name]["zorder"],
            )
            if len(data) > 0:
                max_x = max(max_x, int(np.max(data)))

        ax.set(xlabel=xlabel, ylabel="CDF")

        if is_log:
            ax.set_xscale("symlog")
        else:
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.set_xlim(left=0, right=max(1, max_x) + 0.5)

        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False)
        fig.savefig(f"artifacts/figs/{out_name}", format="pdf", bbox_inches="tight")

if __name__ == '__main__':
    run_a2()
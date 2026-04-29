import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from src.sim.atom_event_runner import AtomEventRunner
from src.protocols.atom_oram import AtomORAM
from src.common.config import ExperimentConfig
from src.common.latency_model import LatencyModel
from src.traces.schema import TraceRecord
from src.common.types import OperationType
from src.common.exp_utils import instantiate_protocol, prepare_storage_config

os.makedirs("artifacts/figs", exist_ok=True)
os.makedirs("artifacts/csv", exist_ok=True)

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 12,
        "pdf.fonttype": 42,
        "axes.linewidth": 1.2,
    }
)

# load the trace subset used for distributional measurements
def load_trace(file_path, limit=5000):
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

#Prime the protocol state before collecting distribution statistics
def warmup_protocol(protocol, runner, block_size, reference_gap_sec, num_blocks=50000):
    records = []
    for i in range(num_blocks):
        records.append(
            TraceRecord(
                trace_id=-(i + 1),
                timestamp=i * reference_gap_sec,
                op=OperationType.WRITE,
                logical_id=i,
                size_bytes=block_size,
                source="warmup",
                original_index=i,
                original_offset=0,
                request_group=0,
            )
        )

    runner.run(
        protocol=protocol,
        records=records,
        block_size=block_size,
        max_idle_ticks_after_last_arrival=0,
        record_virtuals=False,
    )


def _cdf_yvals(n: int) -> np.ndarray:
    if n <= 1:
        return np.zeros(n, dtype=float)
    return np.arange(n) / float(n - 1)


def run_a2():
    cfg = ExperimentConfig.load_default()
    L = cfg.storage.tree_height
    lambda_1 = cfg.atom.lambda1
    block_size = cfg.storage.block_size
    reference_gap_sec = lambda_1 * L * cfg.atom.tick_interval_sec

    traces = {
        "MSRC": "data/processed/msrc_src1_0_trace.csv",
        "AliCloud": "data/processed/alicloud_device32_trace.csv",
    }

    stash_data = {}
    queue_data = {}

    slug_map = {
        "MSRC": "msrc",
        "AliCloud": "alicloud",
    }

    for name, path in traces.items():
        records = load_trace(path, limit=5000)

        storage_cfg = prepare_storage_config(
            cfg.storage,
            exp_name="a2",
            protocol_name="AtomORAM",
            run_tag=f"{name}_steady_state",
        )
        protocol = instantiate_protocol(AtomORAM, cfg, storage_cfg, rng_seed=0)

        warmup_runner = AtomEventRunner(
            latency_model=LatencyModel(config=cfg),
            atom_config=cfg.atom,
        )
        warmup_protocol(
            protocol=protocol,
            runner=warmup_runner,
            block_size=block_size,
            reference_gap_sec=reference_gap_sec,
            num_blocks=50000,
        )

        runner = AtomEventRunner(
            latency_model=LatencyModel(config=cfg),
            atom_config=cfg.atom,
        )

        df = runner.run(
            protocol=protocol,
            records=records,
            block_size=block_size,
            max_idle_ticks_after_last_arrival=0,
            record_virtuals=True,
        )

        stash_series = df["stash_peak_during_access"].astype(int).to_numpy()
        queue_series = df["queue_length_after"].astype(int).to_numpy()

        stash_data[name] = np.sort(stash_series)
        queue_data[name] = np.sort(queue_series)

        yvals = _cdf_yvals(len(stash_data[name]))

        pd.DataFrame(
            {
                "Stash_Peak_During_Access": stash_data[name],
                "Queue_Length": queue_data[name],
                "CDF": yvals,
            }
        ).to_csv(f"artifacts/csv/A2_{slug_map[name]}_distribution.csv", index=False)

    for data_dict, xlabel, is_log, out_name in [
        (
            stash_data,
            "Stash Size (Blocks)",
            False,
            "stash_distribution.pdf",
        ),
        (
            queue_data,
            "Queue Length",
            True,
            "queue_distribution.pdf",
        ),
    ]:
        fig, ax = plt.subplots(figsize=(7, 5))
        style_map = {
            "MSRC": {"linestyle": "--", "zorder": 3, "alpha": 1.0},
            "AliCloud": {"linestyle": "-", "zorder": 2, "alpha": 0.85},
        }
        plot_order = ["MSRC", "AliCloud"]

        max_x = 0
        for name in plot_order:
            data = data_dict[name]
            yvals = _cdf_yvals(len(data))
            ax.plot(data, yvals, label=name, linewidth=2, linestyle=style_map[name]["linestyle"], zorder=style_map[name]["zorder"])
            if len(data) > 0:
                max_x = max(max_x, int(np.max(data)))

        ax.set(xlabel=xlabel, ylabel="CDF")

        if is_log:
            ax.set_xscale("symlog")
        else:
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.set_xlim(left=0, right=max(1, max_x) + 0.5)

        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False)
        fig.savefig(f"artifacts/figs/A2_{out_name}", format="pdf", bbox_inches="tight")


if __name__ == "__main__":
    run_a2()
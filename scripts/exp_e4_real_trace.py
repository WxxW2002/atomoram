import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.sim.atom_event_runner import AtomEventRunner
from src.protocols.atom_oram import AtomORAM
from src.protocols.path_oram import PathORAM
from src.protocols.ring_oram import RingORAM
from src.common.config import StorageConfig, AtomConfig, ExperimentConfig
from src.common.latency_model import LatencyModel
from src.traces.schema import TraceRecord
from src.common.types import OperationType, Request, RequestKind, BlockAddress

os.makedirs('artifacts/figs', exist_ok=True)
os.makedirs('artifacts/csv', exist_ok=True)
plt.rcParams.update({'font.family': 'serif', 'font.size': 12, 'pdf.fonttype': 42, 'axes.linewidth': 1.2})

def load_trace(file_path, limit=2000):
    df = pd.read_csv(file_path, nrows=limit)
    return [TraceRecord(trace_id=int(row['trace_id']), timestamp=float(row['timestamp']),
                        op=OperationType.WRITE if row['op'] == 'W' else OperationType.READ,
                        logical_id=int(row['logical_id']), size_bytes=int(row['size_bytes']),
                        source='real_trace', original_index=int(row['trace_id']), original_offset=0, request_group=0)
            for _, row in df.iterrows()]

def run_baseline(protocol, records, latency_model):
    current_time, latencies = 0.0, []
    for rec in records:
        start_service = max(current_time, rec.timestamp)
        queueing_delay = start_service - rec.timestamp
        req = Request(request_id=rec.trace_id, kind=RequestKind.REAL, op=rec.op,
                      address=BlockAddress(logical_id=rec.logical_id), data=b'\x00'*4096 if rec.op == OperationType.WRITE else None,
                      arrival_time=rec.timestamp, issued_time=start_service, tag="baseline")
        res = protocol.access(req)
        est = latency_model.annotate(res, queueing_delay=queueing_delay)
        current_time = rec.timestamp + est.total_latency
        latencies.append(est.total_latency)
    return latencies

def run_e4():
    L, t_virt, lambda_1 = 20, 0.005, 3
    required_virtual_ticks = int(lambda_1 * L)
    traces = {'MSRC (Sparse)': 'data/processed/msrc_src1_0_trace.csv', 'AliCloud (Dense)': 'data/processed/alicloud_device32_trace.csv'}
    
    storage_config = StorageConfig(tree_height=L, bucket_size=8, block_size=4096)
    atom_config = AtomConfig(tick_interval_sec=t_virt)
    exp_config = ExperimentConfig()
    latency_model = LatencyModel(config=exp_config)
    
    plot_data = []
    
    for name, path in traces.items():
        records = load_trace(path, limit=2000)
        
        path_lats = run_baseline(PathORAM(storage_config), records, latency_model)
        ring_lats = run_baseline(RingORAM(storage_config), records, latency_model)
        
        runner = AtomEventRunner(latency_model=latency_model, atom_config=atom_config)
        df_atom = runner.run(protocol=AtomORAM(storage_config, atom_config=atom_config), records=records, block_size=4096,
                             required_virtual_ticks=required_virtual_ticks, max_idle_ticks_after_last_arrival=0, record_virtuals=False)
        atom_lats = df_atom[df_atom['service_kind'] == 'real']['end_to_end_latency'].values
        
        for prot, lats in [('Path ORAM', path_lats), ('Ring ORAM', ring_lats), ('AtomORAM', atom_lats)]:
            plot_data.append({'Trace': name, 'Protocol': prot, 'P50': np.percentile(lats, 50), 'P95': np.percentile(lats, 95), 'P99': np.percentile(lats, 99)})

    df_plot = pd.DataFrame(plot_data)
    df_plot.to_csv('artifacts/csv/E4_Real_Trace_Comparison.csv', index=False)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    traces_labels = df_plot['Trace'].unique()
    x = np.arange(len(traces_labels))
    width = 0.25
    
    for i, prot in enumerate(df_plot['Protocol'].unique()):
        y_vals = df_plot[df_plot['Protocol'] == prot]['P95']
        ax.bar(x + (i - 1) * width, y_vals, width, label=prot)

    ax.set_ylabel('P95 End-to-End Latency (s)')
    ax.set_xticks(x)
    ax.set_xticklabels(traces_labels)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_yscale('log')
    ax.set_ylim(bottom=10)
    plt.savefig('artifacts/figs/Fig4_Real_Trace_Comparison.pdf', format='pdf', bbox_inches='tight')

if __name__ == '__main__':
    run_e4()
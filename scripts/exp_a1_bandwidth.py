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
    return [TraceRecord(trace_id=int(row['trace_id']), timestamp=float(row['timestamp']), op=OperationType.WRITE if row['op'] == 'W' else OperationType.READ, logical_id=int(row['logical_id']), size_bytes=int(row['size_bytes']), source='trace', original_index=int(row['trace_id']), original_offset=0, request_group=0) for _, row in df.iterrows()]

def run_baseline_bw(protocol, records):
    total_bytes = 0
    for rec in records:
        req = Request(request_id=rec.trace_id, kind=RequestKind.REAL, op=rec.op, address=BlockAddress(logical_id=rec.logical_id), data=b'\x00'*4096 if rec.op == OperationType.WRITE else None, arrival_time=rec.timestamp, issued_time=0.0, tag="baseline")
        res = protocol.access(req)
        total_bytes += res.metrics.total_bytes_down + res.metrics.total_bytes_up
    return total_bytes

def run_a1():
    L, t_virt, lambda_1 = 20, 0.002, 2
    required_virtual_ticks = int(lambda_1 * L)
    traces = {'MSRC (Sparse)': 'data/processed/msrc_src1_0_trace.csv', 'AliCloud (Dense)': 'data/processed/alicloud_device32_trace.csv'}
    storage_config = StorageConfig(tree_height=L, bucket_size=8, block_size=4096)
    atom_config = AtomConfig(tick_interval_sec=t_virt)
    
    plot_data = []
    
    for name, path in traces.items():
        records = load_trace(path, limit=2000)
        
        bw_path = run_baseline_bw(PathORAM(storage_config), records)
        bw_ring = run_baseline_bw(RingORAM(storage_config), records)
        
        runner = AtomEventRunner(latency_model=LatencyModel(config=ExperimentConfig()), atom_config=atom_config)
        df_atom = runner.run(protocol=AtomORAM(storage_config, atom_config=atom_config), records=records, block_size=4096, required_virtual_ticks=required_virtual_ticks, max_idle_ticks_after_last_arrival=0, record_virtuals=False)
        
        real_df = df_atom[df_atom['service_kind'] == 'real']
        bw_atom_real = real_df['online_bytes_down'].sum() + real_df['online_bytes_up'].sum() + real_df['offline_bytes_down'].sum() + real_df['offline_bytes_up'].sum()
        bw_atom_total = bw_atom_real + runner.global_virtual_bytes_down + runner.global_virtual_bytes_up
        
        for prot, bw in [('Path ORAM', bw_path), ('Ring ORAM', bw_ring), ('AtomORAM', bw_atom_total)]:
            plot_data.append({'Trace': name, 'Protocol': prot, 'Total_Bandwidth_MB': bw / (1024 * 1024)})

    df_plot = pd.DataFrame(plot_data)
    df_plot.to_csv('artifacts/csv/A1_Bandwidth.csv', index=False)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    traces_labels = df_plot['Trace'].unique()
    x = np.arange(len(traces_labels))
    width = 0.25
    
    for i, prot in enumerate(df_plot['Protocol'].unique()):
        y_vals = df_plot[df_plot['Protocol'] == prot]['Total_Bandwidth_MB']
        ax.bar(x + (i - 1) * width, y_vals, width, label=prot)

    ax.set_ylabel('Amortized Total Bandwidth (MB)')
    ax.set_xticks(x)
    ax.set_xticklabels(traces_labels)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_yscale('log')
    plt.savefig('artifacts/figs/FigA1_Bandwidth.pdf', format='pdf', bbox_inches='tight')

if __name__ == '__main__':
    run_a1()
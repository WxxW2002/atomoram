import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.sim.atom_event_runner import AtomEventRunner
from src.protocols.atom_oram import AtomORAM
from src.common.config import StorageConfig, AtomConfig, ExperimentConfig
from src.common.latency_model import LatencyModel
from src.traces.schema import TraceRecord
from src.common.types import OperationType, Request, RequestKind, BlockAddress

os.makedirs('artifacts/figs', exist_ok=True)
os.makedirs('artifacts/csv', exist_ok=True)
plt.rcParams.update({'font.family': 'serif', 'font.size': 12, 'pdf.fonttype': 42, 'axes.linewidth': 1.2})

def load_trace(file_path, limit=10000):
    df = pd.read_csv(file_path, nrows=limit)
    return [TraceRecord(trace_id=int(row['trace_id']), timestamp=float(row['timestamp']),
                        op=OperationType.WRITE if row['op'] == 'W' else OperationType.READ,
                        logical_id=int(row['logical_id']), size_bytes=int(row['size_bytes']), source='real_trace',
                        original_index=int(row['trace_id']), original_offset=0, request_group=0) for _, row in df.iterrows()]

def warmup_protocol(protocol, L, num_blocks=50000):
    for i in range(num_blocks):
        protocol.access(Request(request_id=-(i+1), kind=RequestKind.REAL, op=OperationType.WRITE, address=BlockAddress(logical_id=i), data=b'\x00'*4096, arrival_time=0.0, issued_time=0.0, tag="warmup"))
        for j in range(L): 
            protocol.access(Request(request_id=-(i*L + j + 1000000), kind=RequestKind.VIRTUAL, op=OperationType.READ, address=None, data=None, arrival_time=0.0, issued_time=0.0, tag="warmup_virtual"))

def run_appendix():
    L, t_virt, lambda_1 = 20, 0.005, 3
    required_virtual_ticks = int(lambda_1 * L)
    traces = {'MSRC (Sparse)': 'data/processed/msrc_src1_0_trace.csv', 'AliCloud (Dense)': 'data/processed/alicloud_device32_trace.csv'}
    
    stash_data, queue_data = {}, {}
    
    for name, path in traces.items():
        records = load_trace(path, limit=10000)
        protocol = AtomORAM(StorageConfig(tree_height=L, bucket_size=8, block_size=4096), atom_config=AtomConfig(tick_interval_sec=t_virt))
        warmup_protocol(protocol, L, num_blocks=50000)
        
        runner = AtomEventRunner(latency_model=LatencyModel(config=ExperimentConfig()), atom_config=AtomConfig(tick_interval_sec=t_virt))
        df = runner.run(protocol=protocol, records=records, block_size=4096, required_virtual_ticks=required_virtual_ticks, max_idle_ticks_after_last_arrival=0, record_virtuals=False)
        
        stash_data[name] = np.sort(df['stash_size_after'].values)
        queue_data[name] = np.sort(df['queue_length_after'].values)
        
        yvals = np.arange(len(stash_data[name])) / float(len(stash_data[name]) - 1)
        pd.DataFrame({'Stash_Size': stash_data[name], 'Queue_Length': queue_data[name], 'CDF': yvals}).to_csv(f"artifacts/csv/A1_A2_{name[:4]}_Distribution.csv", index=False)

    for data_dict, title, xlabel, is_log, out_name in [(stash_data, 'Stash Occupancy (Steady State)', 'Stash Size (Blocks)', False, 'FigA1_Stash_Distribution.pdf'),
                                                       (queue_data, 'Queue Length Distribution', 'Queue Length', True, 'FigA2_Queue_Distribution.pdf')]:
        fig, ax = plt.subplots(figsize=(7, 5))
        for name, data in data_dict.items():
            yvals = np.arange(len(data)) / float(len(data) - 1)
            ax.plot(data, yvals, label=name, linewidth=2)
        ax.set(xlabel=xlabel, ylabel='CDF')
        if is_log: ax.set_xscale('symlog')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()
        fig.savefig(f'artifacts/figs/{out_name}', format='pdf', bbox_inches='tight')

if __name__ == '__main__':
    run_appendix()
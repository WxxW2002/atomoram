import os
import pandas as pd
import matplotlib.pyplot as plt
from src.sim.atom_event_runner import AtomEventRunner
from src.protocols.atom_oram import AtomORAM
from src.common.config import ExperimentConfig
from src.common.latency_model import LatencyModel
from src.traces.schema import TraceRecord
from src.common.types import OperationType

os.makedirs('artifacts/figs', exist_ok=True)
os.makedirs('artifacts/csv', exist_ok=True)
plt.rcParams.update({'font.family': 'serif', 'font.size': 12, 'pdf.fonttype': 42, 'axes.linewidth': 1.2})

def generate_burst_trace(t_virt, required_virtual_ticks, block_size):
    records = []
    current_time, trace_id = 0.0, 0
    base_gap = t_virt * required_virtual_ticks
    burst_gap = base_gap * 0.1 
    
    for _ in range(100):
        records.append(TraceRecord(trace_id=trace_id, timestamp=current_time, op=OperationType.READ, logical_id=trace_id%1000, size_bytes=block_size, source='b1', original_index=trace_id, original_offset=0, request_group=0))
        current_time += burst_gap; trace_id += 1
    current_time += base_gap * 100
    for _ in range(100):
        records.append(TraceRecord(trace_id=trace_id, timestamp=current_time, op=OperationType.READ, logical_id=trace_id%1000, size_bytes=block_size, source='b2', original_index=trace_id, original_offset=0, request_group=0))
        current_time += burst_gap; trace_id += 1
    return records

def run_e5():
    cfg = ExperimentConfig.load_default()
    L = cfg.storage.tree_height
    t_virt = cfg.atom.tick_interval_sec
    lambda_1 = cfg.atom.lambda1
    block_size = cfg.storage.block_size
    required_virtual_ticks = int(lambda_1 * L)
    
    records = generate_burst_trace(t_virt, required_virtual_ticks, block_size)
    
    protocol = AtomORAM(cfg.storage, atom_config=cfg.atom)
    runner = AtomEventRunner(latency_model=LatencyModel(config=cfg), atom_config=cfg.atom)
    df = runner.run(protocol=protocol, records=records, block_size=block_size, required_virtual_ticks=required_virtual_ticks, max_idle_ticks_after_last_arrival=0, record_virtuals=False)
    
    real_df = df[df['service_kind'] == 'real'][['arrival_time', 'queue_length_after']]
    real_df.to_csv('artifacts/csv/E5_Burst_Recovery.csv', index=False)
    
    plt.figure(figsize=(10, 4))
    plt.plot(real_df['arrival_time'], real_df['queue_length_after'], marker='.', linestyle='-', color='tab:blue', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Queue Length')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig('artifacts/figs/Fig5_Burst_Recovery.pdf', format='pdf', bbox_inches='tight')

if __name__ == '__main__':
    run_e5()
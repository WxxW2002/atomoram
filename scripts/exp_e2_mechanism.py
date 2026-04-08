import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.common.config import StorageConfig, AtomConfig, ExperimentConfig
from src.common.latency_model import LatencyModel
from src.common.types import Request, RequestKind, OperationType, BlockAddress
from src.protocols.direct_store import DirectStore
from src.protocols.path_oram import PathORAM
from src.protocols.ring_oram import RingORAM
from src.protocols.atom_oram import AtomORAM

os.makedirs('artifacts/figs', exist_ok=True)
os.makedirs('artifacts/csv', exist_ok=True)
plt.rcParams.update({'font.family': 'serif', 'font.size': 12, 'pdf.fonttype': 42, 'axes.linewidth': 1.2})

def measure_online_cost(protocol_class, L, is_atom=False):
    storage_config = StorageConfig(tree_height=L, bucket_size=8, block_size=4096)
    exp_config = ExperimentConfig()
    latency_model = LatencyModel(config=exp_config)
    
    if is_atom:
        atom_config = AtomConfig(tick_interval_sec=0.005)
        protocol = protocol_class(storage_config, atom_config=atom_config)
    else:
        protocol = protocol_class(storage_config)
        
    protocol.reset()
    req = Request(request_id=1, kind=RequestKind.REAL, op=OperationType.WRITE,
                  address=BlockAddress(logical_id=0), data=b'\x00' * 4096,
                  arrival_time=0.0, issued_time=0.0, tag="e2")
    
    for _ in range(5): protocol.access(req)
        
    io_touches, latencies = [], []
    for _ in range(10):
        res = protocol.access(req)
        est = latency_model.annotate(res, queueing_delay=0.0)
        touches = 1 if protocol_class.__name__ == 'DirectStore' else res.metrics.online_bucket_reads + res.metrics.online_bucket_writes
        io_touches.append(touches)
        latencies.append(est.online_latency * 1000)

    return np.mean(io_touches), np.mean(latencies)

def run_e2():
    L_values = [10, 12, 14, 16, 18, 20, 22, 24]
    protocols = [('DirectStore', DirectStore, False), ('Path ORAM', PathORAM, False),
                 ('Ring ORAM', RingORAM, False), ('AtomORAM', AtomORAM, True)]
    
    results_io, results_lat = {'L': L_values}, {'L': L_values}
    
    for name, cls, is_atom in protocols:
        results_io[name], results_lat[name] = [], []
        for L in L_values:
            touches, lat = measure_online_cost(cls, L, is_atom)
            results_io[name].append(touches)
            results_lat[name].append(lat)
            
    # 导出 CSV
    pd.DataFrame(results_io).to_csv('artifacts/csv/E2_Online_IO.csv', index=False)
    pd.DataFrame(results_lat).to_csv('artifacts/csv/E2_Online_Latency.csv', index=False)
    
    # 绘图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    markers = ['x', 's', '^', 'o']
    for (name, _, _), marker in zip(protocols, markers):
        ax1.plot(L_values, results_io[name], marker=marker, label=name, linewidth=2)
        ax2.plot(L_values, results_lat[name], marker=marker, label=name, linewidth=2)
        
    ax1.set(xlabel='Tree Height $L$ ($\log N$)', ylabel='Online Server I/O (Buckets)')
    ax2.set(xlabel='Tree Height $L$ ($\log N$)', ylabel='Online Latency (ms)')
    for ax in (ax1, ax2):
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()
        
    plt.tight_layout()
    plt.savefig('artifacts/figs/Fig2_Mechanism_Validation.pdf', format='pdf', bbox_inches='tight')

if __name__ == '__main__':
    run_e2()
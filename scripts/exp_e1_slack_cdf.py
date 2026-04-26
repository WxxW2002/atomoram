import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.common.config import ExperimentConfig

os.makedirs('artifacts/figs', exist_ok=True)
os.makedirs('artifacts/csv', exist_ok=True)

plt.rcParams.update({'font.family': 'serif', 'font.size': 12, 'pdf.fonttype': 42, 'axes.linewidth': 1.2})

def run_e1():
    cfg = ExperimentConfig.load_default()
    L = cfg.storage.tree_height
    t_virt = cfg.atom.tick_interval_sec
    lambda_1 = cfg.atom.lambda1
    
    base_cost = L * t_virt

    fig, ax = plt.subplots(figsize=(8, 5))
    
    for file_path, label in [('data/processed/msrc_src1_0_trace.csv', 'MSRC'), 
                             ('data/processed/alicloud_device32_trace.csv', 'AliCloud')]:
        df = pd.read_csv(file_path)
        df['dt_real'] = df['timestamp'].diff().fillna(0)
        df['slack'] = df['dt_real'] / base_cost
        
        sorted_data = np.sort(df['slack'])
        yvals = np.arange(len(sorted_data)) / float(len(sorted_data) - 1)
        
        out_df = pd.DataFrame({'Slack_Alpha': sorted_data, 'CDF': yvals})
        out_df.to_csv(f"artifacts/csv/E1_{label}_cdf.csv", index=False)
        
        ax.plot(sorted_data, yvals, label=label, linewidth=2)

    ax.set_xscale('symlog', linthresh=0.1)
    ax.axvline(x=1, color='red', linestyle=':', label=rf'$\lambda_1 = 1.0$')

    ax.set_xlim(-0.01, 10000)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel(r'Sparse Slack $\alpha = \Delta t_{real} / (L \cdot t_{virt})$')
    ax.set_ylabel('CDF')
    
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False)
    ax.grid(True, linestyle='--', alpha=0.5)

    plt.savefig('artifacts/figs/E1_sparse_slack_cdf.pdf', format='pdf', bbox_inches='tight')

if __name__ == '__main__':
    run_e1()
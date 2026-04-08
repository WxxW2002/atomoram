import pandas as pd
from src.sim.atom_event_runner import AtomEventRunner
from src.protocols.atom_oram import AtomORAM
from src.common.config import StorageConfig, AtomConfig, ExperimentConfig
from src.common.latency_model import LatencyModel
from src.traces.schema import TraceRecord
from src.common.types import OperationType

# 1. 加载前 1000 条 AliCloud 数据
df_raw = pd.read_csv('data/processed/alicloud_device32_trace.csv', nrows=1000)
records = []
for idx, row in df_raw.iterrows():
    op = OperationType.WRITE if row['op'] == 'W' else OperationType.READ
    records.append(TraceRecord(
        trace_id=int(row['trace_id']),
        timestamp=float(row['timestamp']),
        op=op,
        logical_id=int(row['logical_id']),
        size_bytes=int(row['size_bytes']),
        source='alicloud_smoke',
        original_index=idx,
        original_offset=0,
        request_group="smoke_group"
    ))

# 2. 强行制造拥塞配置
# N=2^20, L=20. 假设 t_virt 为 1ms (0.001s)
atom_config = AtomConfig(tick_interval_sec=0.001) 
storage_config = StorageConfig(tree_height=20, bucket_size=8, block_size=4096)
exp_config = ExperimentConfig() # 使用默认网络与IO参数

protocol = AtomORAM(storage_config=storage_config, atom_config=atom_config)
latency_model = LatencyModel(config=exp_config)
runner = AtomEventRunner(latency_model=latency_model, atom_config=atom_config)

# 3. 运行 Runner
# 强制设定 required_virtual_ticks = 5 (模拟 \lambda_1 * \log N)
print("Running smoke test...")
result_df = runner.run(
    protocol=protocol,
    records=records,
    block_size=4096,
    required_virtual_ticks=5,  
    max_idle_ticks_after_last_arrival=10
)

# 4. 验证隔离与降级指标
real_df = result_df[result_df['service_kind'] == 'real']
virtual_df = result_df[result_df['service_kind'] == 'virtual']

print(f"Total Ticks: {len(result_df)}")
print(f"Real Accesses Processed: {len(real_df)}")
print(f"Virtual Accesses Processed: {len(virtual_df)}")

# 打印前 20 个服务序列，验证是否呈现 1 Real + 5 Virtual 的降级交替模式
print("\n--- Execution Sequence (First 20) ---")
print(result_df[['tick_index', 'service_kind', 'queue_length_after', 'stash_size_after']].head(20).to_string())

# 检查 Stash 是否爆炸 (Root 节点残留监控)
print("\n--- Max Stash Size ---")
print(result_df['stash_size_after'].max())
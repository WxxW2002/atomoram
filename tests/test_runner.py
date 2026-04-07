from src.common.config import (
    CryptoConfig,
    ExperimentConfig,
    NetworkConfig,
    ServerIOConfig,
    StorageConfig,
)
from src.common.latency_model import LatencyModel
from src.protocols.direct_store import DirectStore
from src.sim.runner import TraceRunner
from src.traces.synthetic import generate_constant_interval_trace


def make_config() -> ExperimentConfig:
    return ExperimentConfig(
        storage=StorageConfig(
            block_size=16,
            bucket_size=4,
            tree_height=3,
            use_file_backend=False,
            data_dir="data/tmp",
        ),
        network=NetworkConfig(
            down_bw_bytes_per_sec=100.0,
            up_bw_bytes_per_sec=100.0,
            rtt_sec=0.5,
        ),
        crypto=CryptoConfig(
            enc_bytes_per_sec=1e12,
            dec_bytes_per_sec=1e12,
        ),
        server_io=ServerIOConfig(
            bucket_read_sec=0.1,
            bucket_write_sec=0.1,
        ),
    )


def test_trace_runner_serializes_requests_and_records_queueing() -> None:
    cfg = make_config()
    protocol = DirectStore(config=cfg.storage)
    latency_model = LatencyModel(cfg)
    runner = TraceRunner(latency_model=latency_model)

    records = generate_constant_interval_trace(
        num_requests=2,
        address_space=4,
        interval_sec=0.0,
        seed=1,
        read_ratio=0.0,  # both writes
        request_size_bytes=16,
    )

    df = runner.run(protocol=protocol, records=records, block_size=cfg.storage.block_size)

    assert len(df) == 2
    assert df.loc[0, "queueing_delay"] == 0.0
    assert df.loc[1, "queueing_delay"] > 0.0
    assert df.loc[1, "service_start_time"] >= df.loc[0, "service_finish_time"]
    assert df.loc[0, "protocol"] == "direct_store"


def test_trace_runner_returns_structured_latency_columns() -> None:
    cfg = make_config()
    protocol = DirectStore(config=cfg.storage)
    latency_model = LatencyModel(cfg)
    runner = TraceRunner(latency_model=latency_model)

    records = generate_constant_interval_trace(
        num_requests=1,
        address_space=4,
        interval_sec=1.0,
        seed=2,
        read_ratio=1.0,  # read
        request_size_bytes=16,
    )

    df = runner.run(protocol=protocol, records=records, block_size=cfg.storage.block_size)

    expected_cols = {
        "trace_id",
        "timestamp",
        "service_start_time",
        "service_finish_time",
        "response_time",
        "end_to_end_latency",
        "online_latency",
        "offline_latency",
        "total_latency",
        "queueing_delay",
        "protocol",
        "online_bucket_reads",
        "offline_bucket_writes",
    }
    assert expected_cols.issubset(set(df.columns))
from src.common.config import (
    AtomConfig,
    CryptoConfig,
    ExperimentConfig,
    NetworkConfig,
    ServerIOConfig,
    StorageConfig,
)
from src.common.latency_model import LatencyModel
from src.protocols.atom_oram import AtomORAM
from src.sim.atom_event_runner import AtomEventRunner
from src.traces.synthetic import generate_constant_interval_trace


def make_env():
    storage = StorageConfig(
        block_size=32,
        bucket_size=4,
        tree_height=4,
        use_file_backend=False,
        data_dir="data/tmp",
    )
    atom_cfg = AtomConfig(
        lambda1=1.0,
        tick_interval_sec=1.0,
        queue_limit=100000,
    )
    exp_cfg = ExperimentConfig(
        storage=storage,
        network=NetworkConfig(
            down_bw_bytes_per_sec=1e9,
            up_bw_bytes_per_sec=1e9,
            rtt_sec=0.001,
        ),
        crypto=CryptoConfig(
            enc_bytes_per_sec=1e9,
            dec_bytes_per_sec=1e9,
        ),
        server_io=ServerIOConfig(
            bucket_read_sec=1e-5,
            bucket_write_sec=1e-5,
        ),
        atom=atom_cfg,
    )
    protocol = AtomORAM(storage_config=storage, atom_config=atom_cfg, rng_seed=7)
    runner = AtomEventRunner(
        latency_model=LatencyModel(exp_cfg),
        atom_config=atom_cfg,
    )
    return protocol, runner, storage


def test_atom_event_runner_inserts_virtual_ticks_when_idle() -> None:
    protocol, runner, storage = make_env()

    records = generate_constant_interval_trace(
        num_requests=2,
        address_space=8,
        interval_sec=2.0,
        read_ratio=1.0,
        seed=1,
        request_size_bytes=storage.block_size,
    )

    df = runner.run(
        protocol=protocol,
        records=records,
        block_size=storage.block_size,
    )

    assert len(df) == 3
    assert list(df["service_kind"]) == ["real", "virtual", "real"]
    assert list(df["tick_index"]) == [0, 1, 2]
    assert bool(df.loc[1, "executed_virtual_access"]) is True
    assert int(df.loc[1, "virtual_requests_executed"]) == 1


def test_atom_event_runner_queueing_and_fallback() -> None:
    protocol, runner, storage = make_env()

    records = generate_constant_interval_trace(
        num_requests=3,
        address_space=8,
        interval_sec=0.2,
        read_ratio=1.0,
        seed=2,
        request_size_bytes=storage.block_size,
    )

    df = runner.run(
        protocol=protocol,
        records=records,
        block_size=storage.block_size,
    )

    real_df = df[df["service_kind"] == "real"].reset_index(drop=True)

    assert len(real_df) == 3
    assert float(real_df.loc[0, "queueing_delay"]) == 0.0
    assert float(real_df.loc[1, "queueing_delay"]) > 0.0
    assert float(real_df.loc[2, "queueing_delay"]) > float(real_df.loc[1, "queueing_delay"])
    assert bool(real_df.loc[1, "fallback_flag"]) is True
    assert bool(real_df.loc[2, "fallback_flag"]) is True


def test_atom_event_runner_marks_one_virtual_tick_per_tick() -> None:
    protocol, runner, storage = make_env()

    records = generate_constant_interval_trace(
        num_requests=2,
        address_space=8,
        interval_sec=0.0,
        read_ratio=1.0,
        seed=3,
        request_size_bytes=storage.block_size,
    )

    df = runner.run(
        protocol=protocol,
        records=records,
        block_size=storage.block_size,
    )

    assert all(int(v) == 1 for v in df["virtual_ticks_generated"])
    real_df = df[df["service_kind"] == "real"]
    assert all(bool(v) is False for v in real_df["executed_virtual_access"])


def test_atom_event_runner_keeps_atom_metrics_shape() -> None:
    protocol, runner, storage = make_env()

    records = generate_constant_interval_trace(
        num_requests=1,
        address_space=8,
        interval_sec=1.0,
        read_ratio=1.0,
        seed=4,
        request_size_bytes=storage.block_size,
    )

    df = runner.run(
        protocol=protocol,
        records=records,
        block_size=storage.block_size,
    )

    row = df.iloc[0]
    assert row["protocol"] == "atom_oram"
    assert int(row["online_bucket_reads"]) == 1
    assert int(row["online_rtt"]) == 1
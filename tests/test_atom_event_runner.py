import pytest

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


def make_env(tick_interval_sec: float = 1.0):
    storage = StorageConfig(
        block_size=32,
        bucket_size=4,
        tree_height=4,
        use_file_backend=False,
        data_dir="data/tmp",
    )
    atom_cfg = AtomConfig(
        lambda1=1.0,
        tick_interval_sec=tick_interval_sec,
        queue_limit=100000,
        local_top_half_enabled=False,
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


def test_atom_event_runner_inserts_required_virtual_accesses_between_reals() -> None:
    protocol, runner, storage = make_env()

    records = generate_constant_interval_trace(
        num_requests=2,
        address_space=8,
        interval_sec=0.0,
        read_ratio=1.0,
        seed=1,
        request_size_bytes=storage.block_size,
    )

    df = runner.run(
        protocol=protocol,
        records=records,
        block_size=storage.block_size,
        required_virtual_ticks=2,
        record_virtuals=True,
    )

    assert list(df["service_kind"])[:4] == ["real", "virtual", "virtual", "real"]
    assert bool(df.loc[1, "executed_virtual_access"]) is True
    assert bool(df.loc[2, "executed_virtual_access"]) is True


def test_atom_event_runner_queueing_and_fallback_under_interval_runner() -> None:
    protocol, runner, storage = make_env()

    records = generate_constant_interval_trace(
        num_requests=3,
        address_space=8,
        interval_sec=0.0,
        read_ratio=1.0,
        seed=2,
        request_size_bytes=storage.block_size,
    )

    df = runner.run(
        protocol=protocol,
        records=records,
        block_size=storage.block_size,
        required_virtual_ticks=1,
        record_virtuals=True,
    )

    real_df = df[df["service_kind"] == "real"].reset_index(drop=True)

    assert len(real_df) == 3
    assert float(real_df.loc[0, "queueing_delay"]) == 0.0
    assert float(real_df.loc[1, "queueing_delay"]) > 0.0
    assert float(real_df.loc[2, "queueing_delay"]) > float(real_df.loc[1, "queueing_delay"])
    assert bool(real_df.loc[1, "fallback_flag"]) is True
    assert bool(real_df.loc[2, "fallback_flag"]) is True


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
        record_virtuals=False,
    )

    row = df.iloc[0]
    assert row["protocol"] == "atom_oram"
    assert int(row["online_bucket_reads"]) == 1
    assert int(row["offline_bucket_reads"]) == 2
    assert int(row["offline_bucket_writes"]) in {2, 3}
    assert int(row["online_rtt"]) == 1
    assert int(row["offline_rtt"]) == 2


def test_atom_event_runner_visible_latency_uses_online_only() -> None:
    protocol, runner, storage = make_env()

    records = generate_constant_interval_trace(
        num_requests=1,
        address_space=8,
        interval_sec=1.0,
        read_ratio=1.0,
        seed=5,
        request_size_bytes=storage.block_size,
    )

    df = runner.run(
        protocol=protocol,
        records=records,
        block_size=storage.block_size,
        required_virtual_ticks=0,
        record_virtuals=False,
    )

    row = df.iloc[0]
    assert float(row["queueing_delay"]) == pytest.approx(0.0)
    assert float(row["end_to_end_latency"]) == pytest.approx(float(row["online_latency"]))
    assert float(row["total_latency"]) >= float(row["online_latency"])


def test_atom_event_runner_real_without_queue_has_visible_latency_equal_online_latency() -> None:
    protocol, runner, storage = make_env(tick_interval_sec=0.01)

    records = generate_constant_interval_trace(
        num_requests=1,
        address_space=8,
        interval_sec=1.0,
        read_ratio=1.0,
        seed=11,
        request_size_bytes=storage.block_size,
    )

    df = runner.run(
        protocol=protocol,
        records=records,
        block_size=storage.block_size,
        required_virtual_ticks=2,
        record_virtuals=False,
    )

    row = df[df["service_kind"] == "real"].iloc[0]
    assert float(row["queueing_delay"]) == pytest.approx(0.0)
    assert float(row["end_to_end_latency"]) == pytest.approx(float(row["online_latency"]))


def test_atom_event_runner_second_real_queueing_matches_interval_plus_first_real_service_tail() -> None:
    protocol, runner, storage = make_env(tick_interval_sec=0.01)

    records = generate_constant_interval_trace(
        num_requests=2,
        address_space=8,
        interval_sec=0.0,
        read_ratio=1.0,
        seed=12,
        request_size_bytes=storage.block_size,
    )

    required_virtual_ticks = 3
    df = runner.run(
        protocol=protocol,
        records=records,
        block_size=storage.block_size,
        required_virtual_ticks=required_virtual_ticks,
        record_virtuals=True,
    )

    real_df = df[df["service_kind"] == "real"].reset_index(drop=True)
    first_real_service_tail = float(real_df.loc[0, "total_latency"] - real_df.loc[0, "queueing_delay"])
    interval_part = required_virtual_ticks * runner.atom_config.tick_interval_sec
    expected = interval_part + first_real_service_tail

    assert float(real_df.loc[0, "queueing_delay"]) == pytest.approx(0.0)
    assert float(real_df.loc[1, "queueing_delay"]) == pytest.approx(expected)


def test_atom_event_runner_burst_queueing_grows_with_interval_and_single_tail() -> None:
    protocol, runner, storage = make_env(tick_interval_sec=0.01)

    records = generate_constant_interval_trace(
        num_requests=3,
        address_space=8,
        interval_sec=0.0,
        read_ratio=1.0,
        seed=13,
        request_size_bytes=storage.block_size,
    )

    required_virtual_ticks = 2
    df = runner.run(
        protocol=protocol,
        records=records,
        block_size=storage.block_size,
        required_virtual_ticks=required_virtual_ticks,
        record_virtuals=True,
    )

    real_df = df[df["service_kind"] == "real"].reset_index(drop=True)
    first_real_service_tail = float(real_df.loc[0, "total_latency"] - real_df.loc[0, "queueing_delay"])
    interval_part = required_virtual_ticks * runner.atom_config.tick_interval_sec

    assert float(real_df.loc[0, "queueing_delay"]) == pytest.approx(0.0)
    assert float(real_df.loc[1, "queueing_delay"]) == pytest.approx(interval_part + first_real_service_tail)
    assert float(real_df.loc[2, "queueing_delay"]) == pytest.approx(2 * interval_part + first_real_service_tail)

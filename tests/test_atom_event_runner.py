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


def test_atom_event_runner_compensates_before_next_real() -> None:
    protocol, runner, storage = make_env(tick_interval_sec=0.01)

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
        record_virtuals=True,
    ).reset_index(drop=True)

    real_indices = list(df.index[df["service_kind"] == "real"])
    assert len(real_indices) == 2

    middle = df.iloc[real_indices[0] + 1: real_indices[1]]
    assert len(middle) >= 1
    assert set(middle["service_kind"]) == {"virtual"}
    assert bool(middle["required_virtual_access"].all()) is True
    assert bool(middle["compensation_wait_tick"].all()) is True
    assert bool(middle["compensation_satisfied"].any()) is True

    satisfied = middle[middle["compensation_satisfied"]].iloc[-1]
    assert int(satisfied["generated_dummy_level"]) == int(satisfied["compensating_real_level"])


def test_atom_event_runner_queueing_and_fallback_under_compensation_runner() -> None:
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
        record_virtuals=False,
    )

    row = df[df["service_kind"] == "real"].iloc[0]
    assert float(row["queueing_delay"]) == pytest.approx(0.0)
    assert float(row["end_to_end_latency"]) == pytest.approx(float(row["online_latency"]))


def test_atom_event_runner_no_fixed_virtual_count_between_reals() -> None:
    protocol, runner, storage = make_env(tick_interval_sec=0.01)

    records = generate_constant_interval_trace(
        num_requests=2,
        address_space=8,
        interval_sec=0.0,
        read_ratio=1.0,
        seed=12,
        request_size_bytes=storage.block_size,
    )

    df = runner.run(
        protocol=protocol,
        records=records,
        block_size=storage.block_size,
        record_virtuals=True,
    ).reset_index(drop=True)

    real_indices = list(df.index[df["service_kind"] == "real"])
    middle = df.iloc[real_indices[0] + 1: real_indices[1]]

    assert len(middle) >= 1
    assert bool(middle["compensation_satisfied"].iloc[-1]) is True
    # The count is determined by level-match waiting, not by a fixed
    # lambda*logN parameter passed to runner.run().
    assert runner.required_virtual_ticks_executed >= len(middle)


def test_atom_event_runner_compensation_match_uses_real_level() -> None:
    protocol, runner, storage = make_env(tick_interval_sec=0.01)

    records = generate_constant_interval_trace(
        num_requests=3,
        address_space=8,
        interval_sec=0.0,
        read_ratio=1.0,
        seed=13,
        request_size_bytes=storage.block_size,
    )

    df = runner.run(
        protocol=protocol,
        records=records,
        block_size=storage.block_size,
        record_virtuals=True,
    )

    satisfied = df[df["compensation_satisfied"] == True]  # noqa: E712
    assert len(satisfied) >= 3
    assert (
        satisfied["generated_dummy_level"].astype(int)
        == satisfied["compensating_real_level"].astype(int)
    ).all()

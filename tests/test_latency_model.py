from src.common.config import (
    CryptoConfig,
    ExperimentConfig,
    NetworkConfig,
    ServerIOConfig,
    StorageConfig,
)
from src.common.latency_model import LatencyModel
from src.common.metrics import AccessMetrics, AccessResult, TimingRecord


def make_config() -> ExperimentConfig:
    return ExperimentConfig(
        storage=StorageConfig(block_size=16, bucket_size=4, tree_height=3),
        network=NetworkConfig(
            down_bw_bytes_per_sec=100.0,
            up_bw_bytes_per_sec=200.0,
            rtt_sec=0.5,
        ),
        crypto=CryptoConfig(
            enc_bytes_per_sec=400.0,
            dec_bytes_per_sec=800.0,
        ),
        server_io=ServerIOConfig(
            bucket_read_sec=0.1,
            bucket_write_sec=0.2,
        ),
    )


def test_latency_model_formula() -> None:
    config = make_config()
    model = LatencyModel(config)

    metrics = AccessMetrics(protocol="path_oram")
    metrics.online_bytes_down = 80
    metrics.online_bytes_up = 40
    metrics.online_rtt = 2
    metrics.online_bucket_reads = 3
    metrics.online_bucket_writes = 1

    metrics.offline_bytes_down = 20
    metrics.offline_bytes_up = 60
    metrics.offline_rtt = 1
    metrics.offline_bucket_reads = 2
    metrics.offline_bucket_writes = 4

    estimate = model.estimate(metrics, queueing_delay=0.25)

    assert abs(estimate.online_network_time - 2.0) < 1e-12

    assert abs(estimate.offline_network_time - 1.0) < 1e-12

    assert abs(estimate.online_server_io_time - 0.5) < 1e-12

    assert abs(estimate.offline_server_io_time - 1.0) < 1e-12

    assert abs(estimate.online_crypto_time - 0.2) < 1e-12

    assert abs(estimate.offline_crypto_time - 0.175) < 1e-12

    assert abs(estimate.online_latency - 2.95) < 1e-12

    assert abs(estimate.offline_latency - 2.175) < 1e-12

    assert abs(estimate.total_latency - 5.125) < 1e-12


def test_latency_model_infers_queueing_delay_from_timing() -> None:
    config = make_config()
    model = LatencyModel(config)

    metrics = AccessMetrics(protocol="direct_store")
    timing = TimingRecord(
        arrival_time=1.0,
        service_start_time=1.4,
    )

    estimate = model.estimate(metrics, timing=timing)
    assert abs(estimate.queueing_delay - 0.4) < 1e-12


def test_latency_model_annotate_uses_online_latency_for_response_time() -> None:
    config = make_config()
    model = LatencyModel(config)

    metrics = AccessMetrics(protocol="atom_oram")
    metrics.online_bytes_down = 100
    metrics.online_rtt = 1
    metrics.online_bucket_reads = 1

    metrics.offline_bytes_up = 200
    metrics.offline_bucket_writes = 2
    metrics.offline_rtt = 1

    result = AccessResult(
        data=None,
        metrics=metrics,
        timing=TimingRecord(
            arrival_time=10.0,
            service_start_time=10.3,
        ),
    )

    estimate = model.annotate(result)

    assert abs(result.timing.end_to_end_latency - estimate.online_latency) < 1e-12
    assert abs(result.timing.total_modeled_latency - estimate.total_latency) < 1e-12
    assert abs(result.timing.response_time - (10.0 + estimate.online_latency)) < 1e-12
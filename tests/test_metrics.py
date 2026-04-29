from src.common.metrics import AccessMetrics, TimingRecord

def test_access_metrics_addition_and_bool_or() -> None:
    lhs = AccessMetrics(protocol="atom_oram")
    lhs.online_bucket_reads = 1
    lhs.online_bytes_down = 128
    lhs.online_rtt = 1
    lhs.fallback_flag = False
    lhs.virtual_ticks_generated = 2

    rhs = AccessMetrics(protocol="atom_oram")
    rhs.offline_bucket_writes = 3
    rhs.offline_bytes_up = 256
    rhs.offline_rtt = 2
    rhs.fallback_flag = True
    rhs.virtual_ticks_generated = 5

    lhs.add(rhs)

    assert lhs.online_bucket_reads == 1
    assert lhs.offline_bucket_writes == 3
    assert lhs.total_bytes_down == 128
    assert lhs.total_bytes_up == 256
    assert lhs.total_rtt == 3
    assert lhs.virtual_ticks_generated == 7
    assert lhs.fallback_flag is True


def test_access_metrics_reject_different_protocols() -> None:
    lhs = AccessMetrics(protocol="path_oram")
    rhs = AccessMetrics(protocol="atom_oram")

    try:
        lhs.add(rhs)
    except ValueError as exc:
        assert "different protocols" in str(exc)
    else:
        raise AssertionError("Expected ValueError for mismatched protocols.")


def test_timing_record_finalize() -> None:
    record = TimingRecord(
        arrival_time=1.0,
        service_start_time=1.2,
        response_time=1.8,
        network_time=0.2,
        server_io_time=0.3,
        crypto_time=0.1,
        client_cpu_time=0.05,
        queueing_delay=0.2,
    )

    record.finalize()

    assert abs(record.total_modeled_latency - 0.85) < 1e-12
    assert abs(record.end_to_end_latency - 0.8) < 1e-12
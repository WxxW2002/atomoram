from pathlib import Path

from src.common.types import OperationType
from src.traces.alicloud import load_alicloud_trace
from src.traces.msrc import load_msrc_trace
from src.traces.google import load_google_trace
from src.traces.synthetic import (
    generate_constant_interval_trace,
    generate_sparse_trace,
    generate_two_burst_trace,
)


def test_load_msrc_trace_splits_and_compacts(tmp_path: Path) -> None:
    csv_path = tmp_path / "msrc.csv"
    csv_path.write_text(
        "\n".join(
            [
                "10000000,hostA,0,Read,4096,4096,1",
                "20000000,hostA,0,Write,12288,8192,2",
            ]
        ),
        encoding="utf-8",
    )

    records = load_msrc_trace(csv_path, block_size=4096)

    assert len(records) == 2
    assert records[0].timestamp == 0.0
    assert records[0].op == OperationType.READ
    assert records[1].timestamp == 1.0
    assert records[1].op == OperationType.WRITE
    assert [r.logical_id for r in records] == [0, 1]
    assert all(r.size_bytes == 4096 for r in records)
    assert records[1].metadata["raw_size_bytes"] == 8192


def test_load_alicloud_trace_splits_and_normalizes_time(tmp_path: Path) -> None:
    csv_path = tmp_path / "alicloud.csv"
    csv_path.write_text(
        "\n".join(
            [
                "32,R,0,4096,5000000",
                "32,W,4096,8192,7000000",
            ]
        ),
        encoding="utf-8",
    )

    records = load_alicloud_trace(csv_path, block_size=4096)

    assert len(records) == 2
    assert records[0].timestamp == 0.0
    assert records[1].timestamp == 2.0
    assert records[0].op == OperationType.READ
    assert records[1].op == OperationType.WRITE
    assert all(r.size_bytes == 4096 for r in records)
    assert records[1].metadata["raw_size_bytes"] == 8192


def test_generate_constant_interval_trace() -> None:
    records = generate_constant_interval_trace(
        num_requests=4,
        address_space=8,
        interval_sec=0.5,
        seed=7,
    )
    assert len(records) == 4
    assert records[0].timestamp == 0.0
    assert records[1].timestamp == 0.5
    assert records[2].timestamp == 1.0


def test_generate_sparse_trace_uses_alpha_formula() -> None:
    records = generate_sparse_trace(
        num_requests=3,
        address_space=8,
        alpha=2.0,
        lambda1=1.5,
        tree_height=4,
        t_virtual_sec=0.01,
        seed=1,
    )
    assert abs(records[1].timestamp - 0.12) < 1e-12
    assert abs(records[2].timestamp - 0.24) < 1e-12


def test_generate_two_burst_trace() -> None:
    records = generate_two_burst_trace(
        burst1_size=2,
        burst2_size=2,
        address_space=8,
        intra_burst_interval_sec=0.1,
        idle_gap_sec=1.0,
        seed=1,
    )
    assert len(records) == 4
    assert abs(records[0].timestamp - 0.0) < 1e-12
    assert abs(records[1].timestamp - 0.1) < 1e-12
    assert abs(records[2].timestamp - 1.2) < 1e-12
    assert records[0].metadata["burst"] == 1
    assert records[2].metadata["burst"] == 2

def test_load_google_trace_preserves_one_request_one_record(tmp_path: Path) -> None:
    from src.traces.google import load_google_trace

    csv_path = tmp_path / "google.csv"
    csv_path.write_text(
        "\n".join(
            [
                "filename,file_offset,application,c_time,io_zone,redundancy_type,op_type,service_class,from_flash_cache,cache_hit,request_io_size_bytes,disk_io_size_bytes,response_io_size_bytes,start_time,disk_time,simulated_disk_start_time,simulated_latency",
                "fileA,0,appA,100,WARM,REPLICATED,READ,OTHER,0,1,1050624,0,0,10.0,0.0,0.0,0.001",
                "fileA,4096,appA,100,WARM,REPLICATED,WRITE,OTHER,0,-1,8192,0,0,12.0,0.0,0.0,0.002",
                "fileB,0,appB,200,COLD,ERASURE_CODED,READ,OTHER,0,0,4096,4096,4096,15.0,0.01,15.0,0.02",
            ]
        ),
        encoding="utf-8",
    )

    records = load_google_trace(csv_path, block_size=4096)

    assert len(records) == 3
    assert records[0].timestamp == 0.0
    assert records[1].timestamp == 2.0
    assert records[2].timestamp == 5.0

    assert records[0].op == OperationType.READ
    assert records[1].op == OperationType.WRITE
    assert records[2].op == OperationType.READ

    assert all(r.size_bytes == 4096 for r in records)
    assert records[0].metadata["raw_size_bytes"] == 1050624
    assert records[1].metadata["raw_size_bytes"] == 8192
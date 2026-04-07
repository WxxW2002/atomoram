from pathlib import Path

from src.common.types import OperationType
from src.traces.alicloud import load_alicloud_trace
from src.traces.msrc import load_msrc_trace
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

    # Row1 -> block 1 ; Row2 -> blocks 3 and 4 ; compacted -> 0,1,2
    assert len(records) == 3
    assert records[0].timestamp == 0.0
    assert records[0].op == OperationType.READ
    assert records[1].timestamp == 1.0
    assert records[1].op == OperationType.WRITE
    assert [r.logical_id for r in records] == [0, 1, 2]


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

    assert len(records) == 3
    assert records[0].timestamp == 0.0
    assert records[1].timestamp == 2.0
    assert records[2].timestamp == 2.0
    assert records[0].op == OperationType.READ
    assert records[1].op == OperationType.WRITE


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
    # interval = 2.0 * 1.5 * 4 * 0.01 = 0.12
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
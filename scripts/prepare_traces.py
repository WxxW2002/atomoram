from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.config import ExperimentConfig
from src.common.utils import ensure_dir
from src.traces.alicloud import load_alicloud_trace
from src.traces.msrc import load_msrc_trace
from src.traces.schema import records_to_dataframe


def main() -> None:
    cfg = ExperimentConfig.from_yaml(REPO_ROOT / "configs" / "default.yaml")
    processed_dir = ensure_dir(REPO_ROOT / "data" / "processed")

    msrc_path = REPO_ROOT / "data" / "raw" / "MSRC" / "src1_0_tripped.csv"
    alicloud_path = REPO_ROOT / "data" / "raw" / "AliCloud" / "io_traces_32.csv"

    print("[1/2] parsing MSRC ...")
    msrc_records = load_msrc_trace(
        msrc_path,
        block_size=cfg.storage.block_size,
        compact_addresses=True,
        split_multi_block_requests=True,
    )
    msrc_df = records_to_dataframe(msrc_records)
    msrc_out = processed_dir / "msrc_src1_0_trace.csv"
    msrc_df.to_csv(msrc_out, index=False)
    print(f"  saved {len(msrc_df)} records to {msrc_out}")

    print("[2/2] parsing AliCloud ...")
    alicloud_records = load_alicloud_trace(
        alicloud_path,
        block_size=cfg.storage.block_size,
        compact_addresses=True,
        split_multi_block_requests=True,
    )
    alicloud_df = records_to_dataframe(alicloud_records)
    alicloud_out = processed_dir / "alicloud_device32_trace.csv"
    alicloud_df.to_csv(alicloud_out, index=False)
    print(f"  saved {len(alicloud_df)} records to {alicloud_out}")


if __name__ == "__main__":
    main()
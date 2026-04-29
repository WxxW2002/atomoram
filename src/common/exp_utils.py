from __future__ import annotations

import re
import shutil

from dataclasses import replace
from pathlib import Path
from typing import Type

from src.common.config import ExperimentConfig, StorageConfig
from src.protocols.atom_oram import AtomORAM
from src.protocols.direct_store import DirectStore
from src.protocols.path_oram import PathORAM
from src.protocols.ring_oram import RingORAM


REPO_ROOT = Path(__file__).resolve().parents[2]


def _slug(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9._-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "run"


def prepare_storage_config(
    base_storage: StorageConfig,
    *,
    exp_name: str,
    protocol_name: str,
    run_tag: str,
) -> StorageConfig:
    run_dir = (
        REPO_ROOT
        / "data"
        / "runtime"
        / _slug(exp_name)
        / _slug(protocol_name)
        / _slug(run_tag)
    )

    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    return replace(
        base_storage,
        use_file_backend=True,
        data_dir=str(run_dir),
    )

# instantiate a protocol class with the configuration it expects
def instantiate_protocol(
    protocol_cls: Type,
    cfg: ExperimentConfig,
    storage_cfg: StorageConfig,
    *,
    rng_seed: int = 0,
):
    if protocol_cls is AtomORAM:
        return AtomORAM(storage_cfg, atom_config=cfg.atom, rng_seed=rng_seed)

    if protocol_cls is PathORAM:
        return PathORAM(storage_cfg, rng_seed=rng_seed)

    if protocol_cls is RingORAM:
        return RingORAM(storage_cfg, ring_config=cfg.ring, rng_seed=rng_seed)

    if protocol_cls is DirectStore:
        return DirectStore(storage_cfg)

    raise TypeError(f"Unsupported protocol class: {protocol_cls}")
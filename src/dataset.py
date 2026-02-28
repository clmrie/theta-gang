"""Data loading, spike-sequence reconstruction, and PyTorch dataset.

Usage
-----
    from src.dataset import load_data, SpikeSequenceDataset, collate_fn
    df_moving, nGroups, nChannelsPerGroup = load_data()
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.config import (
    DATA_DIR,
    JSON_NAME,
    MAX_CHANNELS,
    MAX_SEQ_LEN,
    PARQUET_NAME,
)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(data_dir=None):
    """Load the parquet dataset and JSON session parameters.

    Parameters
    ----------
    data_dir : path-like, optional — defaults to ``src.config.DATA_DIR``

    Returns
    -------
    df_moving          : pd.DataFrame — rows where speedMask is True
    nGroups            : int          — number of recording shanks
    nChannelsPerGroup  : list[int]    — channels per shank
    """
    data_dir     = Path(data_dir or DATA_DIR)
    parquet_path = data_dir / PARQUET_NAME
    json_path    = data_dir / JSON_NAME

    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Data not found at {parquet_path}\n"
            "Place the .parquet and .json files in the data/ directory."
        )

    df = pd.read_parquet(parquet_path)
    with open(json_path) as f:
        params = json.load(f)

    nGroups           = params["nGroups"]
    nChannelsPerGroup = [params[f"group{g}"]["nChannels"] for g in range(nGroups)]

    speed_masks = np.array([x[0] for x in df["speedMask"]])
    df_moving   = df[speed_masks].reset_index(drop=True)

    print(f"Loaded {len(df):,} samples → {len(df_moving):,} moving "
          f"({len(df_moving) / len(df):.1%})")
    print(f"nGroups={nGroups}, channels per shank={nChannelsPerGroup}")

    return df_moving, nGroups, nChannelsPerGroup


# ── Sequence reconstruction ───────────────────────────────────────────────────

def reconstruct_sequence(row, nGroups, nChannelsPerGroup, max_seq_len=MAX_SEQ_LEN):
    """Rebuild the chronological spike sequence from a compressed DataFrame row.

    The dataset stores spikes compactly (waveform bank + index list per group).
    This function dereferences indices to recover the ordered multi-shank sequence.

    Parameters
    ----------
    row                : pd.Series — one row of the moving DataFrame
    nGroups            : int
    nChannelsPerGroup  : list[int]
    max_seq_len        : int — maximum sequence length to return

    Returns
    -------
    seq_waveforms : list of (waveform_array, shank_id) tuples
    seq_shank_ids : list of int
    """
    groups = row["groups"]
    length = min(len(groups), max_seq_len)

    waveforms_by_group = {}
    for g in range(nGroups):
        nCh = nChannelsPerGroup[g]
        waveforms_by_group[g] = row[f"group{g}"].reshape(-1, nCh, 32)

    seq_waveforms = []
    seq_shank_ids = []
    for t in range(length):
        g   = int(groups[t])
        idx = int(row[f"indices{g}"][t])
        if 0 < idx <= waveforms_by_group[g].shape[0]:
            seq_waveforms.append((waveforms_by_group[g][idx - 1], g))
            seq_shank_ids.append(g)

    return seq_waveforms, seq_shank_ids


# ── PyTorch Dataset ───────────────────────────────────────────────────────────

class SpikeSequenceDataset(Dataset):
    """PyTorch dataset of spike sequences with (x, y), zone, and curvilinear-d targets.

    Each item is a variable-length sequence of multi-channel spike waveforms.
    The ``collate_fn`` handles padding for batching.
    """

    def __init__(
        self,
        dataframe,
        nGroups,
        nChannelsPerGroup,
        curvilinear_d,
        zone_labels,
        max_seq_len=MAX_SEQ_LEN,
    ):
        self.df                = dataframe
        self.nGroups           = nGroups
        self.nChannelsPerGroup = nChannelsPerGroup
        self.max_seq_len       = max_seq_len
        self.targets           = np.array(
            [[x[0], x[1]] for x in dataframe["pos"]], dtype=np.float32
        )
        self.curvilinear_d = curvilinear_d.astype(np.float32)
        self.zone_labels   = zone_labels.astype(np.int64)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row    = self.df.iloc[idx]
        seq, shank_ids = reconstruct_sequence(
            row, self.nGroups, self.nChannelsPerGroup, self.max_seq_len
        )
        seq_len = max(len(seq), 1)  # at least length 1 to avoid empty tensors

        waveforms     = np.zeros((seq_len, MAX_CHANNELS, 32), dtype=np.float32)
        shank_ids_arr = np.zeros(seq_len, dtype=np.int64)
        for t, (wf, g) in enumerate(seq):
            nCh = wf.shape[0]
            waveforms[t, :nCh, :]  = wf
            shank_ids_arr[t]       = g

        return {
            "waveforms": torch.from_numpy(waveforms),
            "shank_ids": torch.from_numpy(shank_ids_arr),
            "seq_len":   seq_len,
            "target":    torch.from_numpy(self.targets[idx]),
            "d":         torch.tensor(self.curvilinear_d[idx], dtype=torch.float32),
            "zone":      torch.tensor(self.zone_labels[idx],   dtype=torch.long),
        }


# ── Collate function ──────────────────────────────────────────────────────────

def collate_fn(batch):
    """Pad variable-length spike sequences and build a boolean padding mask.

    The padding mask follows the PyTorch convention used by TransformerEncoder:
    ``True`` means "this position is padding, ignore it".
    """
    max_len    = max(item["seq_len"] for item in batch)
    batch_size = len(batch)

    waveforms    = torch.zeros(batch_size, max_len, MAX_CHANNELS, 32)
    shank_ids    = torch.zeros(batch_size, max_len, dtype=torch.long)
    mask         = torch.ones(batch_size, max_len, dtype=torch.bool)   # True = padded
    targets      = torch.stack([item["target"] for item in batch])
    d_targets    = torch.stack([item["d"]      for item in batch])
    zone_targets = torch.stack([item["zone"]   for item in batch])

    for i, item in enumerate(batch):
        sl = item["seq_len"]
        waveforms[i, :sl] = item["waveforms"]
        shank_ids[i, :sl] = item["shank_ids"]
        mask[i, :sl]      = False   # real tokens are not padding

    return {
        "waveforms":    waveforms,
        "shank_ids":    shank_ids,
        "mask":         mask,
        "targets":      targets,
        "d_targets":    d_targets,
        "zone_targets": zone_targets,
    }


import json
import os
import shutil
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

try:
    import torch
    from torch.utils.data import Dataset as _TorchDataset
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    _TorchDataset = object


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


class MmapSpikeDataset(_TorchDataset):
    """Memory-mapped PyTorch dataset for spike data. Fork-safe for num_workers > 0."""

    def __init__(self, mmap_dir: str, n_groups: int):
        with open(os.path.join(mmap_dir, "meta.json"), "r") as f:
            meta = json.load(f)
        self._n_total = meta["n_total"]
        self._n_groups = n_groups

        # Fixed-length fields: mmap directly
        self._pos = np.load(os.path.join(mmap_dir, "pos.npy"), mmap_mode="r")
        self._length = np.load(os.path.join(mmap_dir, "length.npy"), mmap_mode="r")
        self._time = np.load(os.path.join(mmap_dir, "time.npy"), mmap_mode="r")
        self._time_behavior = np.load(os.path.join(mmap_dir, "time_behavior.npy"), mmap_mode="r")
        self._pos_index = np.load(os.path.join(mmap_dir, "pos_index.npy"), mmap_mode="r")

        # Variable-length fields: mmap data + load offsets into RAM (offsets are small)
        self._var_data = {}
        self._var_offsets = {}
        var_fields = (
            ["groups", "indexInDat"]
            + [f"group{g}" for g in range(n_groups)]
            + [f"indices{g}" for g in range(n_groups)]
        )
        for field in var_fields:
            self._var_data[field] = np.load(
                os.path.join(mmap_dir, f"{field}_data.npy"), mmap_mode="r"
            )
            self._var_offsets[field] = np.load(
                os.path.join(mmap_dir, f"{field}_offsets.npy")
            )

    def __len__(self):
        return self._n_total

    def __getitem__(self, idx):
        inputs = {}

        inputs["length"] = torch.as_tensor(self._length[idx].copy())
        inputs["time"] = torch.as_tensor(self._time[idx].copy())
        inputs["time_behavior"] = torch.as_tensor(self._time_behavior[idx].copy())
        inputs["pos_index"] = torch.as_tensor(self._pos_index[idx].copy())

        for field in ["groups", "indexInDat"]:
            lo = int(self._var_offsets[field][idx])
            hi = int(self._var_offsets[field][idx + 1])
            inputs[field] = torch.as_tensor(np.array(self._var_data[field][lo:hi]))

        for g in range(self._n_groups):
            for field in [f"group{g}", f"indices{g}"]:
                lo = int(self._var_offsets[field][idx])
                hi = int(self._var_offsets[field][idx + 1])
                inputs[field] = torch.as_tensor(np.array(self._var_data[field][lo:hi]))

        targets = torch.as_tensor(np.array(self._pos[idx]), dtype=torch.float32)
        return inputs, targets


class SpikeDataset:
    """Unified loader for hippocampal spike data.

    Usage:
        ds = SpikeDataset("dataset", mouse="M1199_PAG", window_size=108)
        train_ds, val_ds = ds.get_tf_dataset(batch_size=256, val_split=0.2)

        # or load raw parquet for custom processing
        df = ds.load_parquet()
    """

    AVAILABLE_WINDOWS = [36, 108, 252]

    def __init__(
        self,
        dataset_dir: str,
        mouse: str = "M1199_PAG",
        window_size: int = 108,
        stride: int = 4,
    ):
        self.dataset_dir = os.path.abspath(dataset_dir)
        self.mouse = mouse
        self.window_size = window_size
        self.stride = stride

        # Load params from JSON
        self.params = self._load_params()
        self.n_groups = self.params.nGroups
        self.n_channels_per_group = self.params.nChannelsPerGroup

        # Build TFRecord feature description
        self.feat_desc = self._build_feat_desc()

    def _load_params(self) -> AttrDict:
        json_path = os.path.join(self.dataset_dir, f"{self.mouse}.json")
        with open(json_path, "r") as f:
            params = AttrDict(json.load(f))

        params.nChannelsPerGroup = []
        for g in range(params.nGroups):
            params.nChannelsPerGroup.append(params[f"group{g}"]["nChannels"])

        return params

    def _build_feat_desc(self) -> Dict:
        feat_desc = {
            "pos_index": tf.io.FixedLenFeature([], tf.int64),
            "pos": tf.io.VarLenFeature(tf.float32),
            "length": tf.io.FixedLenFeature([], tf.int64),
            "groups": tf.io.VarLenFeature(tf.int64),
            "time": tf.io.FixedLenFeature([], tf.float32),
            "time_behavior": tf.io.FixedLenFeature([], tf.float32),
            "indexInDat": tf.io.VarLenFeature(tf.int64),
            "speedMask": tf.io.VarLenFeature(dtype=tf.string),
        }
        for g in range(self.n_groups):
            feat_desc[f"group{g}"] = tf.io.VarLenFeature(tf.float32)
            feat_desc[f"indices{g}"] = tf.io.VarLenFeature(tf.int64)
        return feat_desc

    def _get_filename(self, fmt: str = "tfrec") -> str:
        return f"{self.mouse}_stride{self.stride}_win{self.window_size}_test.{fmt}"

    def _get_filepath(self, fmt: str = "tfrec") -> str:
        return os.path.join(self.dataset_dir, self._get_filename(fmt))

    def _parse_serialized(self, tensors: Dict, batched: bool = False) -> Dict:
        """Reshape spike groups from flat to (n_spikes, n_channels, 32)."""
        if isinstance(tensors["pos"], tf.SparseTensor):
            tensors["pos"] = tf.sparse.to_dense(tensors["pos"])

        tensors["groups"] = tf.reshape(
            tf.sparse.to_dense(tensors["groups"], default_value=-1), [-1]
        )
        tensors["indexInDat"] = tf.reshape(
            tf.sparse.to_dense(tensors["indexInDat"], default_value=-1), [-1]
        )

        for g in range(self.n_groups):
            n_ch = self.n_channels_per_group[g]
            zeros = tf.constant(np.zeros([n_ch, 32]), tf.float32)

            tensors[f"group{g}"] = tf.reshape(
                tf.sparse.to_dense(tensors[f"group{g}"]), [-1]
            )
            tensors[f"indices{g}"] = tf.reshape(
                tf.cast(tf.sparse.to_dense(tensors[f"indices{g}"]), tf.int32), [-1]
            )

            if batched:
                tensors[f"group{g}"] = tf.reshape(
                    tensors[f"group{g}"],
                    [self.params.batchSize, -1, n_ch, 32],
                )

            tensors[f"group{g}"] = tf.reshape(
                tensors[f"group{g}"], [-1, n_ch, 32]
            )

            # Filter out zero-padded spikes
            non_zeros = tf.logical_not(
                tf.equal(
                    tf.reduce_sum(
                        tf.cast(tf.equal(tensors[f"group{g}"], zeros), tf.int32),
                        axis=[1, 2],
                    ),
                    32 * n_ch,
                )
            )
            tensors[f"group{g}"] = tf.gather(
                tensors[f"group{g}"], tf.where(non_zeros)
            )[:, 0, :, :]

        return tensors

    def load_parquet(self) -> pd.DataFrame:
        """Load raw data as a pandas DataFrame."""
        return pd.read_parquet(self._get_filepath("parquet"))

    def load_raw_tf(self, use_speed_mask: bool = False) -> tf.data.Dataset:
        """Load and parse TFRecord into a tf.data.Dataset (unbatched).

        Each element is a dict with keys like group0, groups, pos, etc.
        Spike groups are reshaped to (n_spikes, n_channels, 32).
        """
        file_path = self._get_filepath("tfrec")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"TFRecord not found: {file_path}")

        raw = tf.data.TFRecordDataset(file_path)

        # we don't use the sample at all if the mouse was not moving 
        if use_speed_mask:
            speed_desc = {"speedMask": tf.io.VarLenFeature(dtype=tf.string)}

            @tf.autograph.experimental.do_not_convert
            def _filter_speed(example_proto):
                ex = tf.io.parse_single_example(example_proto, speed_desc)
                return tf.equal(ex["speedMask"].values[0], b"\x01")

            raw = raw.filter(_filter_speed)

        feat_desc = self.feat_desc

        @tf.autograph.experimental.do_not_convert
        def _parse(example_proto):
            return tf.io.parse_single_example(example_proto, feat_desc)

        dataset = raw.map(_parse, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(
            lambda x: self._parse_serialized(x),
            num_parallel_calls=tf.data.AUTOTUNE,
        ).prefetch(tf.data.AUTOTUNE)

        return dataset

    def get_tf_dataset(
        self,
        batch_size: int = 256,
        val_split: float = 0.2,
        use_speed_mask: bool = False,
        shuffle_buffer: int = 4096,
        pos_dims: int = 2,
    ) -> Tuple[tf.data.Dataset, Optional[tf.data.Dataset]]:
        """Ready-to-train (inputs, targets) tf.data.Datasets.

        Returns (train_ds, val_ds). val_ds is None if val_split <= 0.
        """
        dataset = self.load_raw_tf(use_speed_mask=use_speed_mask)

        def reduce_pos(example):
            example["pos"] = example["pos"][:pos_dims]
            example.pop("speedMask", None)
            return example

        dataset = dataset.map(reduce_pos, num_parallel_calls=tf.data.AUTOTUNE)

        # Count total examples for splitting
        n_total = sum(1 for _ in dataset)

        # Split train/val
        if val_split > 0:
            n_val = int(n_total * val_split)
            n_train = n_total - n_val
            train_raw = dataset.take(n_train)
            val_raw = dataset.skip(n_train)
        else:
            train_raw = dataset
            val_raw = None

        # Padded batching config
        padded_shapes = {
            "groups": [None],
            "pos": [pos_dims],
            "indexInDat": [None],
            "time": [],
            "time_behavior": [],
            "length": [],
            "pos_index": [],
        }
        padding_values = {
            "groups": tf.constant(-1, dtype=tf.int64),
            "pos": tf.constant(-1.0, dtype=tf.float32),
            "indexInDat": tf.constant(-1, dtype=tf.int64),
            "time": tf.constant(-1.0, dtype=tf.float32),
            "time_behavior": tf.constant(-1.0, dtype=tf.float32),
            "length": tf.constant(-1, dtype=tf.int64),
            "pos_index": tf.constant(-1, dtype=tf.int64),
        }
        for g in range(self.n_groups):
            padded_shapes[f"group{g}"] = [None, self.n_channels_per_group[g], 32]
            padded_shapes[f"indices{g}"] = [None]
            padding_values[f"group{g}"] = tf.constant(-1.0, dtype=tf.float32)
            padding_values[f"indices{g}"] = tf.constant(0, dtype=tf.int32)

        def split_xy(vals):
            pos = vals.pop("pos")
            return vals, pos

        def _batch_pipeline(ds, shuffle: bool):
            if shuffle:
                ds = ds.shuffle(shuffle_buffer)
            ds = ds.padded_batch(
                batch_size,
                padded_shapes=padded_shapes,
                padding_values=padding_values,
            )
            ds = ds.map(split_xy, num_parallel_calls=tf.data.AUTOTUNE)
            ds = ds.prefetch(tf.data.AUTOTUNE)
            return ds

        train_ds = _batch_pipeline(train_raw, shuffle=True)
        val_ds = _batch_pipeline(val_raw, shuffle=False) if val_raw else None

        return train_ds, val_ds

    def _mmap_dir(self) -> str:
        return os.path.join(
            self.dataset_dir,
            f"{self.mouse}_stride{self.stride}_win{self.window_size}_mmap",
        )

    def preprocess(self, pos_dims: int = 2, force: bool = False):
        """One-time: parse TFRecords and save as memory-mapped .npy files."""
        mmap_dir = self._mmap_dir()
        if os.path.isdir(mmap_dir) and not force:
            print(f"Mmap cache already exists: {mmap_dir}")
            return
        if os.path.isdir(mmap_dir):
            shutil.rmtree(mmap_dir)

        os.makedirs(mmap_dir)
        dataset = self.load_raw_tf(use_speed_mask=False)

        fixed = {"pos": [], "length": [], "time": [], "time_behavior": [], "pos_index": []}
        var_fields = (
            ["groups", "indexInDat"]
            + [f"group{g}" for g in range(self.n_groups)]
            + [f"indices{g}" for g in range(self.n_groups)]
        )
        var_data = {k: [] for k in var_fields}
        var_offsets = {k: [0] for k in var_fields}

        count = 0
        for example in dataset:
            fixed["pos"].append(example["pos"].numpy()[:pos_dims])
            for k in ["length", "time", "time_behavior", "pos_index"]:
                fixed[k].append(example[k].numpy())
            for k in var_fields:
                arr = example[k].numpy()
                var_data[k].append(arr)
                var_offsets[k].append(var_offsets[k][-1] + arr.shape[0])
            count += 1
            if count % 10000 == 0:
                print(f"  processed {count} examples...")

        # Save fixed-length fields
        for k, v in fixed.items():
            np.save(os.path.join(mmap_dir, f"{k}.npy"), np.stack(v))

        # Save variable-length fields (flat data + offset index)
        for k in var_fields:
            np.save(os.path.join(mmap_dir, f"{k}_data.npy"), np.concatenate(var_data[k], axis=0))
            np.save(os.path.join(mmap_dir, f"{k}_offsets.npy"), np.array(var_offsets[k], dtype=np.int64))

        meta = {
            "n_total": count,
            "pos_dims": pos_dims,
            "n_groups": self.n_groups,
            "n_channels_per_group": self.n_channels_per_group,
        }
        with open(os.path.join(mmap_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        print(f"Done: {count} examples saved to {mmap_dir}")

    def get_pytorch_dataset(self, val_split: float = 0.2, pos_dims: int = 2):
        """Returns (train_dataset, val_dataset) backed by memory-mapped files."""
        from torch.utils.data import Subset

        mmap_dir = self._mmap_dir()
        if not os.path.isdir(mmap_dir):
            self.preprocess(pos_dims=pos_dims)

        full_ds = MmapSpikeDataset(mmap_dir, self.n_groups)
        n_total = len(full_ds)

        if val_split > 0:
            n_val = int(n_total * val_split)
            n_train = n_total - n_val
            train_ds = Subset(full_ds, range(n_train))
            val_ds = Subset(full_ds, range(n_train, n_total))
        else:
            train_ds = full_ds
            val_ds = None

        return train_ds, val_ds

    def collate_fn(self, batch):
        """Pads variable-length fields in a batch. Use with DataLoader(collate_fn=ds.collate_fn)."""
        import torch

        inputs_list, targets_list = zip(*batch)
        targets = torch.stack(targets_list)

        keys_scalar = ["length", "time", "time_behavior", "pos_index"]
        keys_var = ["groups", "indexInDat"]

        collated = {}
        for k in keys_scalar:
            collated[k] = torch.stack([inp[k] for inp in inputs_list])

        for k in keys_var:
            tensors = [inp[k] for inp in inputs_list]
            max_len = max(t.shape[0] for t in tensors)
            pad_val = -1
            padded = torch.full((len(tensors), max_len), pad_val, dtype=tensors[0].dtype)
            for i, t in enumerate(tensors):
                padded[i, :t.shape[0]] = t
            collated[k] = padded

        for g in range(self.n_groups):
            # group waveforms: (n_spikes, n_channels, 32) -> pad n_spikes dim
            tensors = [inp[f"group{g}"] for inp in inputs_list]
            max_spikes = max(t.shape[0] for t in tensors)
            n_ch = self.n_channels_per_group[g]
            padded = torch.full((len(tensors), max_spikes, n_ch, 32), -1.0)
            for i, t in enumerate(tensors):
                padded[i, :t.shape[0]] = t
            collated[f"group{g}"] = padded

            # indices: (total_spikes,) -> pad independently
            idx_tensors = [inp[f"indices{g}"] for inp in inputs_list]
            max_idx = max(t.shape[0] for t in idx_tensors)
            padded = torch.zeros((len(idx_tensors), max_idx), dtype=idx_tensors[0].dtype)
            for i, t in enumerate(idx_tensors):
                padded[i, :t.shape[0]] = t
            collated[f"indices{g}"] = padded

        return collated, targets

    def summary(self) -> str:
        """Print a summary of the dataset configuration."""
        lines = [
            f"Mouse:        {self.mouse}",
            f"Window:       {self.window_size} ms",
            f"Stride:       {self.stride}",
            f"Groups:       {self.n_groups}",
            f"Channels:     {self.n_channels_per_group}",
            f"TFRec file:   {self._get_filename('tfrec')}",
            f"Parquet file: {self._get_filename('parquet')}",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"SpikeDataset(mouse={self.mouse!r}, window={self.window_size}ms, groups={self.n_groups})"

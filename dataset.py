import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


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

    def get_pytorch_batches(self, **kwargs):
        """Wraps get_tf_dataset() and yields PyTorch tensors.

        Accepts the same kwargs as get_tf_dataset (batch_size, val_split, etc).
        Returns (train_generator, val_generator). Each yields (inputs_dict, targets).
        """
        import torch

        train_tf, val_tf = self.get_tf_dataset(**kwargs)

        def _to_torch(tf_dataset):
            if tf_dataset is None:
                return None
            def gen():
                for inputs, targets in tf_dataset:
                    pt_inputs = {
                        k: torch.from_numpy(v.numpy()) for k, v in inputs.items()
                    }
                    pt_targets = torch.from_numpy(targets.numpy())
                    yield pt_inputs, pt_targets
            return gen

        return _to_torch(train_tf), _to_torch(val_tf)

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

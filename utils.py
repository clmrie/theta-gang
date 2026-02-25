import numpy as np
import os
from typing import Dict, List
import tensorflow as tf


def load_parsed_dataset(
    file_path: str, params: Dict, featDesc: Dict = None, use_speedMask: bool = False
) -> Dict[str, tf.data.Dataset]:
    """
    Load datasets that were previously saved using _save_datasets_to_tfrec.

    Parameters
    ----------
    file_path : str
        Base path for the TFRecord files.
    params : dict
        Parameters of the network, used for parsing.
    featDesc : dict, optional
        The feature description to use for parsing. If None, uses a default that
        handles variable position dimensions.

    Returns
    -------
    datasets : dict
        Dictionary of loaded tf.data.Dataset objects.
    """
    if featDesc is None:
        # Default featDesc that handles variable length pos and groups
        featDesc = {
            "pos_index": tf.io.FixedLenFeature([], tf.int64),
            "pos": tf.io.VarLenFeature(tf.float32),
            "length": tf.io.FixedLenFeature([], tf.int64),
            "groups": tf.io.VarLenFeature(tf.int64),
            "time": tf.io.FixedLenFeature([], tf.float32),
            "time_behavior": tf.io.FixedLenFeature([], tf.float32),
            "indexInDat": tf.io.VarLenFeature(tf.int64),
            "speedMask": tf.io.VarLenFeature(dtype=tf.string),
        }
        for g in range(params.nGroups):
            featDesc[f"group{g}"] = tf.io.VarLenFeature(tf.float32)
            featDesc[f"indices{g}"] = tf.io.VarLenFeature(tf.int64)
    if not os.path.exists(file_path):
        raise ValueError(f"Warning: File {file_path} does not exist. Skipping.")

    raw_dataset = tf.data.TFRecordDataset(file_path)

    if use_speedMask:
        speed_featDesc = {
            "speedMask": tf.io.VarLenFeature(dtype=tf.string),
        }

        @tf.autograph.experimental.do_not_convert
        def _filter_speed_mask(example_proto):
            example = tf.io.parse_single_example(example_proto, speed_featDesc)
            is_fast = tf.equal(example["speedMask"].values[0], b"\x01")
            return is_fast

        raw_dataset = raw_dataset.filter(_filter_speed_mask)

    @tf.autograph.experimental.do_not_convert
    def _parse_function(example_proto):
        return tf.io.parse_single_example(example_proto, featDesc)

    dataset = raw_dataset.map(_parse_function, num_parallel_calls=tf.data.AUTOTUNE)
    # Re-apply the parsing logic to get the correct shapes (e.g., reshaping groups)
    dataset = dataset.map(
        lambda x: parse_serialized_sequence(params, x, batched=False),
        num_parallel_calls=tf.data.AUTOTUNE,
    ).prefetch(tf.data.AUTOTUNE)

    return dataset


def parse_serialized_sequence(params, tensors, batched=False, count_spikes=False):
    """
    Parse a serialized spike sequence example.
    Args:
        params: parameters of the network
        tensors: parsed tensors from the TFRecord example
        batched: Whether data is batched
        count_spikes: Whether to count spikes

    Returns:
        Parsed tensors with reshaped spike data.
        In particular, each "group" tensor is reshaped to [num_spikes, nChannelsPerGroup[g], 32].
        If batched, the shape should be [batchSize, num_spikes_per_batch, nChannelsPerGroup[g], 32] but is then reshaped to merge batch and spikes, giving:
        [batchSize * num_spikes_per_batch, nChannelsPerGroup[g], 32].
    """
    if isinstance(tensors["pos"], tf.SparseTensor):
        tensors["pos"] = tf.sparse.to_dense(tensors["pos"])

    tensors["groups"] = tf.sparse.to_dense(tensors["groups"], default_value=-1)
    # Pierre 13/02/2021: Why use sparse.to_dense, and not directly a FixedLenFeature?
    # Probably because he wanted a variable length <> inputs sequences
    tensors["groups"] = tf.reshape(tensors["groups"], [-1])

    tensors["indexInDat"] = tf.sparse.to_dense(tensors["indexInDat"], default_value=-1)
    # reshape indexInDat to be a flat array
    tensors["indexInDat"] = tf.reshape(tensors["indexInDat"], [-1])

    for g in range(params.nGroups):
        # here 32 correspond to the number of discretized time bin for a spike
        zeros = tf.constant(np.zeros([params.nChannelsPerGroup[g], 32]), tf.float32)
        tensors["group" + str(g)] = tf.sparse.to_dense(tensors["group" + str(g)])
        tensors["group" + str(g)] = tf.reshape(tensors["group" + str(g)], [-1])
        tensors["indices" + str(g)] = tf.reshape(
            tf.cast(tf.sparse.to_dense(tensors["indices" + str(g)]), dtype=tf.int32),
            [-1],
        )
        if batched:
            tensors["group" + str(g)] = tf.reshape(
                tensors["group" + str(g)],
                [params.batchSize, -1, params.nChannelsPerGroup[g], 32],
            )
            if count_spikes:
                group_batched = tensors["group" + str(g)]
                # nonzero mask per sample
                nonzero_mask = tf.logical_not(
                    tf.equal(
                        tf.reduce_sum(
                            tf.cast(tf.equal(group_batched, zeros), tf.int32),
                            axis=[2, 3],
                        ),
                        32 * params.nChannelsPerGroup[g],
                    )
                )

                # spike counts per sample (shape = [batchSize])
                spike_counts = tf.reduce_sum(tf.cast(nonzero_mask, tf.int32), axis=1)

                # store result in tensors
                tensors[f"group{g}_spikes_count"] = spike_counts
        # Merged spikes: [num_spikes, nChannelsPerGroup[g], 32]
        tensors["group" + str(g)] = tf.reshape(
            tensors["group" + str(g)], [-1, params.nChannelsPerGroup[g], 32]
        )
        # Pierre 12/03/2021: the batchSize and timesteps are gathered together
        nonZeros = tf.logical_not(
            tf.equal(
                tf.reduce_sum(
                    input_tensor=tf.cast(
                        tf.equal(tensors["group" + str(g)], zeros), tf.int32
                    ),
                    axis=[1, 2],
                ),
                32 * params.nChannelsPerGroup[g],
            )
        )
        # nonZeros: control that the voltage measured is not 0, at all channels and time bin inside the detected spike
        tensors["group" + str(g)] = tf.gather(
            tensors["group" + str(g)], tf.where(nonZeros)
        )[:, 0, :, :]
        # I don't understand why it can then call [:,0,:,:] as the output tensor of gather should have the same
        # shape as tensors["group"+str(g)"], [-1,params.nChannels[g],32] ...

    return tensors


def build_dummy_model(params, nFeatures):
    group_inputs = {}
    index_inputs = {}

    # --- 1. Inputs ---
    for g in range(params.nGroups):
        group_inputs[f"group{g}"] = tf.keras.layers.Input(
            shape=(None, params.nChannelsPerGroup[g], 32), name=f"group{g}"
        )
        index_inputs[f"indices{g}"] = tf.keras.layers.Input(
            shape=(None,), dtype=tf.int32, name=f"indices{g}"
        )

    groups_sequence = tf.keras.layers.Input(
        shape=(None,), dtype=tf.int64, name="groups"
    )

    # --- 2. Per-Shank Spike Processing ---
    processed_embeddings = []
    for g in range(params.nGroups):
        # Flatten: (batch, n_spikes, channels * 32)
        n_flat = params.nChannelsPerGroup[g] * 32
        x = tf.keras.layers.Reshape((-1, n_flat))(group_inputs[f"group{g}"])

        # Linear projection of the waveform
        x = tf.keras.layers.Dense(nFeatures, activation="relu", name=f"embed_g{g}")(x)

        # The "Null Spike" Trick:
        # Add a row of zeros at index 0.
        # Now, when index_inputs[g] is 0 (padding), tf.gather picks up these zeros.
        full_emb = tf.keras.layers.Lambda(
            lambda e: tf.concat([tf.zeros_like(e[:, :1, :]), e], axis=1),
            name=f"null_spike_g{g}",
        )(x)

        # Gather: (batch, seqLen, nFeatures)
        # batch_dims=1 is the secret sauce for parallel gathering across the batch
        gathered = tf.keras.layers.Lambda(
            lambda args: tf.gather(args[0], args[1], batch_dims=1),
            output_shape=(None, nFeatures),
            name=f"gather_g{g}",
        )([full_emb, index_inputs[f"indices{g}"]])

        processed_embeddings.append(gathered)

    # --- 3. Sequence Reorganization ---
    # Stack shanks: (batch, seqLen, nGroups, nFeatures)
    stacked = tf.keras.layers.Lambda(
        lambda x: tf.stack(x, axis=2), name="stack_shanks"
    )(processed_embeddings)

    # Pick the active shank for each index in the sequence
    def select_active(args):
        stack, g_seq = args
        # Convert groups_sequence to one-hot, treating -1 as 'no group' (all zeros)
        # tf.one_hot naturally ignores -1 if we cast carefully
        mask = tf.one_hot(tf.cast(g_seq, tf.int32), depth=params.nGroups)
        return tf.reduce_sum(stack * mask[..., tf.newaxis], axis=2)

    sequence_features = tf.keras.layers.Lambda(select_active, name="reorganize")(
        [stacked, groups_sequence]
    )

    # --- 4. Global Average Pooling with Mask ---
    # We must ignore timepoints where groups_sequence == -1
    mask = tf.keras.layers.Lambda(
        lambda x: tf.cast(tf.not_equal(x, -1), tf.float32), name="padding_mask"
    )(groups_sequence)

    def masked_pool(args):
        feats, m = args  # (B, L, F), (B, L)
        m = tf.expand_dims(m, axis=-1)
        total = tf.reduce_sum(feats * m, axis=1)
        count = tf.reduce_sum(m, axis=1) + 1e-7
        return total / count

    context_vector = tf.keras.layers.Lambda(masked_pool, name="masked_avg_pool")(
        [sequence_features, mask]
    )

    # --- 5. Output Head ---
    pos_output = tf.keras.layers.Dense(2, name="pos")(context_vector)

    # Define all model inputs in order
    all_inputs = (
        list(group_inputs.values()) + list(index_inputs.values()) + [groups_sequence]
    )
    model = tf.keras.Model(inputs=all_inputs, outputs=pos_output)

    return model

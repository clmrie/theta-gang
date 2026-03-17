"""Convert .tfrec (TFRecord) to .parquet + extract JSON params."""
import struct
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from google.protobuf.internal.decoder import _DecodeVarint32

# TFRecord Example proto — we only need the generated pb2 from TF
# But since TF may crash, parse manually with raw protobuf.

ROOT = Path(__file__).resolve().parent.parent


def read_tfrecords(path):
    """Yield raw serialized Example bytes from a TFRecord file."""
    with open(path, "rb") as f:
        while True:
            len_bytes = f.read(8)
            if not len_bytes or len(len_bytes) < 8:
                break
            length = struct.unpack("<Q", len_bytes)[0]
            f.read(4)  # length CRC
            data = f.read(length)
            f.read(4)  # data CRC
            yield data


def parse_example(raw_bytes):
    """Parse a tf.train.Example from raw protobuf bytes using TF's pb2."""
    from tensorflow.core.example import example_pb2
    ex = example_pb2.Example()
    ex.ParseFromString(raw_bytes)
    result = {}
    for key, feature in ex.features.feature.items():
        if len(feature.float_list.value) > 0:
            result[key] = np.array(feature.float_list.value, dtype=np.float32)
        elif len(feature.int64_list.value) > 0:
            result[key] = np.array(feature.int64_list.value, dtype=np.int64)
        elif len(feature.bytes_list.value) > 0:
            result[key] = feature.bytes_list.value[0]
        else:
            result[key] = None
    return result


def main():
    tfrec_path = ROOT / "M1199_PAG_stride4_win108_test.tfrec"
    if not tfrec_path.exists():
        print(f"ERROR: {tfrec_path} not found")
        sys.exit(1)

    print(f"Reading {tfrec_path} ...")
    rows = []
    for i, raw in enumerate(read_tfrecords(tfrec_path)):
        row = parse_example(raw)
        rows.append(row)
        if (i + 1) % 5000 == 0:
            print(f"  {i + 1} records parsed...")

    print(f"Total records: {len(rows)}")

    # Inspect first record
    first = rows[0]
    print(f"\nFeature keys: {sorted(first.keys())}")
    for k, v in sorted(first.items()):
        if isinstance(v, np.ndarray):
            print(f"  {k}: {v.dtype} shape={v.shape}")
        else:
            print(f"  {k}: {type(v)}")

    # Build DataFrame
    df = pd.DataFrame(rows)
    out_dir = ROOT / "data"
    out_dir.mkdir(exist_ok=True)
    parquet_path = out_dir / "M1199_PAG_stride4_win108_test.parquet"
    print(f"\nWriting {parquet_path} ...")
    df.to_parquet(parquet_path)
    print(f"Done! {parquet_path.stat().st_size / 1e9:.2f} GB")

    # Also write JSON params (inferred from data)
    import json
    n_groups = 4
    # Infer nChannels per group from waveform sizes
    # group0 has 6 channels, group1 has 4, group2 has 6, group3 has 4
    # Each waveform is nCh * 32 floats per spike
    params = {"nGroups": n_groups}
    channel_counts = [6, 4, 6, 4]  # from notebook output
    for g in range(n_groups):
        params[f"group{g}"] = {"nChannels": channel_counts[g]}

    json_path = out_dir / "M1199_PAG.json"
    with open(json_path, "w") as f:
        json.dump(params, f, indent=2)
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()

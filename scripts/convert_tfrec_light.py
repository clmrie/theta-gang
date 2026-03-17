#!/usr/bin/env python
"""Convert .tfrec → .parquet without importing tensorflow (pure protobuf)."""
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from google.protobuf.descriptor import FieldDescriptor
from google.protobuf import descriptor_pb2

# Inline the tf.train.Example / tf.train.Feature protobuf definitions
# so we don't need to import tensorflow at all.
from google.protobuf import descriptor_pool, symbol_database, reflection, message_factory

EXAMPLE_PROTO = """
syntax = "proto3";
package tensorflow;

message BytesList { repeated bytes value = 1; }
message FloatList { repeated float value = 1 [packed = true]; }
message Int64List { repeated int64 value = 1 [packed = true]; }

message Feature {
  oneof kind {
    BytesList bytes_list = 1;
    FloatList float_list = 2;
    Int64List int64_list = 3;
  }
}

message Features {
  map<string, Feature> feature = 1;
}

message Example {
  Features features = 1;
}
"""

# Instead of compiling proto at runtime, just use the TF pb2 if available,
# otherwise fall back to manual parsing.

ROOT = Path(__file__).resolve().parent.parent


def read_tfrecords(path):
    """Yield raw bytes for each record in a TFRecord file."""
    with open(path, "rb") as f:
        while True:
            header = f.read(8)
            if not header or len(header) < 8:
                break
            length = struct.unpack("<Q", header)[0]
            f.read(4)  # masked CRC of length
            data = f.read(length)
            f.read(4)  # masked CRC of data
            yield data


def parse_feature_manual(data):
    """Parse a single tf.train.Feature from raw bytes.

    Wire format (proto3):
    - bytes_list (field 1): sub-message with repeated bytes (field 1)
    - float_list (field 2): sub-message with packed repeated float (field 1)
    - int64_list (field 3): sub-message with packed repeated int64 (field 1)
    """
    from google.protobuf.internal.decoder import _DecodeVarint
    from google.protobuf.internal.wire_format import WIRETYPE_LENGTH_DELIMITED, WIRETYPE_VARINT

    pos = 0
    result_floats = []
    result_ints = []
    result_bytes = []

    while pos < len(data):
        # Read tag
        tag, pos = _DecodeVarint(data, pos)
        field_number = tag >> 3
        wire_type = tag & 0x7

        if wire_type == 2:  # LENGTH_DELIMITED
            length, pos = _DecodeVarint(data, pos)
            sub_data = data[pos:pos+length]
            pos += length

            if field_number == 2:  # float_list sub-message
                # Parse inner: field 1, packed floats
                inner_pos = 0
                while inner_pos < len(sub_data):
                    inner_tag, inner_pos = _DecodeVarint(sub_data, inner_pos)
                    inner_field = inner_tag >> 3
                    inner_wire = inner_tag & 0x7
                    if inner_wire == 2 and inner_field == 1:
                        inner_len, inner_pos = _DecodeVarint(sub_data, inner_pos)
                        n_floats = inner_len // 4
                        result_floats = list(struct.unpack(f"<{n_floats}f", sub_data[inner_pos:inner_pos+inner_len]))
                        inner_pos += inner_len
                    else:
                        break

            elif field_number == 3:  # int64_list sub-message
                inner_pos = 0
                while inner_pos < len(sub_data):
                    inner_tag, inner_pos = _DecodeVarint(sub_data, inner_pos)
                    inner_field = inner_tag >> 3
                    inner_wire = inner_tag & 0x7
                    if inner_wire == 2 and inner_field == 1:
                        inner_len, inner_pos = _DecodeVarint(sub_data, inner_pos)
                        # Packed varints
                        end = inner_pos + inner_len
                        while inner_pos < end:
                            val, inner_pos = _DecodeVarint(sub_data, inner_pos)
                            # Decode zigzag for signed
                            if val > 0x7FFFFFFFFFFFFFFF:
                                val -= (1 << 64)
                            result_ints.append(val)
                        inner_pos = end
                    else:
                        break

            elif field_number == 1:  # bytes_list sub-message
                inner_pos = 0
                while inner_pos < len(sub_data):
                    inner_tag, inner_pos = _DecodeVarint(sub_data, inner_pos)
                    inner_field = inner_tag >> 3
                    inner_wire = inner_tag & 0x7
                    if inner_wire == 2 and inner_field == 1:
                        inner_len, inner_pos = _DecodeVarint(sub_data, inner_pos)
                        result_bytes.append(sub_data[inner_pos:inner_pos+inner_len])
                        inner_pos += inner_len
                    else:
                        break

    if result_floats:
        return np.array(result_floats, dtype=np.float32)
    elif result_ints:
        return np.array(result_ints, dtype=np.int64)
    elif result_bytes:
        return result_bytes[0] if len(result_bytes) == 1 else result_bytes
    return None


def parse_features_map(data):
    """Parse a Features message (map<string, Feature>)."""
    from google.protobuf.internal.decoder import _DecodeVarint

    result = {}
    pos = 0
    while pos < len(data):
        tag, pos = _DecodeVarint(data, pos)
        field_number = tag >> 3
        wire_type = tag & 0x7

        if wire_type == 2 and field_number == 1:
            # Map entry
            entry_len, pos = _DecodeVarint(data, pos)
            entry_data = data[pos:pos+entry_len]
            pos += entry_len

            # Parse map entry: field 1 = key (string), field 2 = value (Feature)
            epos = 0
            key = None
            value_data = None
            while epos < len(entry_data):
                etag, epos = _DecodeVarint(entry_data, epos)
                efield = etag >> 3
                ewire = etag & 0x7
                if ewire == 2:
                    elen, epos = _DecodeVarint(entry_data, epos)
                    if efield == 1:
                        key = entry_data[epos:epos+elen].decode("utf-8")
                    elif efield == 2:
                        value_data = entry_data[epos:epos+elen]
                    epos += elen
                else:
                    break

            if key is not None and value_data is not None:
                result[key] = parse_feature_manual(value_data)
        else:
            break

    return result


def parse_example(raw_bytes):
    """Parse a tf.train.Example from raw protobuf bytes."""
    from google.protobuf.internal.decoder import _DecodeVarint

    pos = 0
    while pos < len(raw_bytes):
        tag, pos = _DecodeVarint(raw_bytes, pos)
        field_number = tag >> 3
        wire_type = tag & 0x7

        if wire_type == 2 and field_number == 1:
            length, pos = _DecodeVarint(raw_bytes, pos)
            features_data = raw_bytes[pos:pos+length]
            pos += length
            return parse_features_map(features_data)
        else:
            break

    return {}


def main():
    tfrec_path = ROOT / "M1199_PAG_stride4_win108_test.tfrec"
    if not tfrec_path.exists():
        print(f"ERROR: {tfrec_path} not found")
        sys.exit(1)

    print(f"Reading {tfrec_path} ({tfrec_path.stat().st_size / 1e9:.2f} GB)...")
    t0 = time.time()

    rows = []
    for i, raw in enumerate(read_tfrecords(tfrec_path)):
        row = parse_example(raw)
        rows.append(row)
        if (i + 1) % 5000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(f"  {i + 1} records ({elapsed:.0f}s, {rate:.0f} rec/s)")

    elapsed = time.time() - t0
    print(f"Parsed {len(rows)} records in {elapsed:.1f}s")

    # Inspect first record
    first = rows[0]
    print(f"\nFeature keys: {sorted(first.keys())}")
    for k, v in sorted(first.items()):
        if isinstance(v, np.ndarray):
            print(f"  {k}: {v.dtype} shape={v.shape}")
        else:
            print(f"  {k}: {type(v)}")

    # Build DataFrame
    print("\nBuilding DataFrame...")
    df = pd.DataFrame(rows)

    out_dir = ROOT / "data"
    out_dir.mkdir(exist_ok=True)

    parquet_path = out_dir / "M1199_PAG_stride4_win108_test.parquet"
    print(f"Writing {parquet_path} ...")
    df.to_parquet(parquet_path)
    print(f"Done! {parquet_path.stat().st_size / 1e9:.2f} GB")

    # Write JSON params
    params = {"nGroups": 4}
    channel_counts = [6, 4, 6, 4]
    for g in range(4):
        params[f"group{g}"] = {"nChannels": channel_counts[g]}

    json_path = out_dir / "M1199_PAG.json"
    with open(json_path, "w") as f:
        json.dump(params, f, indent=2)
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()

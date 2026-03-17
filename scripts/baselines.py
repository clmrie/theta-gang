"""Baseline models for position decoding — evaluated on the same held-out test set.

Baselines
---------
1. Naive (predict training-set mean position)
2. Ridge regression on flattened spike features
3. k-NN + PCA-80 (the original baseline from the README)

Usage (from repo root)
----------------------
    python scripts/baselines.py

Outputs
-------
Prints a comparison table with: Euclidean error, MSE_x, MSE_y, zone accuracy,
corridor adherence — for each baseline and the Transformer ensemble.
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import json

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

import src.config as cfg
from src.geometry import (
    compute_distance_to_skeleton,
    compute_geometry,
    d_to_zone,
)


# ── Lightweight data loading (no torch dependency) ────────────────────────────

def load_data_light(data_dir=None):
    """Load parquet + JSON without importing torch."""
    data_dir = Path(data_dir or cfg.DATA_DIR)
    df = pd.read_parquet(data_dir / cfg.PARQUET_NAME)
    with open(data_dir / cfg.JSON_NAME) as f:
        params = json.load(f)
    nGroups = params["nGroups"]
    nChannelsPerGroup = [params[f"group{g}"]["nChannels"] for g in range(nGroups)]
    speed_masks = np.array([bool(x[0]) if isinstance(x, (bytes, np.ndarray)) else bool(x)
                            for x in df["speedMask"]])
    df_moving = df[speed_masks].reset_index(drop=True)
    print(f"Loaded {len(df):,} samples → {len(df_moving):,} moving "
          f"({len(df_moving) / len(df):.1%})")
    return df_moving, nGroups, nChannelsPerGroup


def reconstruct_sequence(row, nGroups, nChannelsPerGroup, max_seq_len=cfg.MAX_SEQ_LEN):
    """Rebuild spike sequence from a compressed DataFrame row (no torch)."""
    groups = row["groups"]
    length = min(len(groups), max_seq_len)
    waveforms_by_group = {}
    for g in range(nGroups):
        nCh = nChannelsPerGroup[g]
        raw = row[f"group{g}"]
        if raw is None or (isinstance(raw, float) and np.isnan(raw)):
            waveforms_by_group[g] = np.zeros((0, nCh, 32), dtype=np.float32)
        else:
            waveforms_by_group[g] = np.asarray(raw).reshape(-1, nCh, 32)
    seq = []
    for t in range(length):
        g = int(groups[t])
        idx = int(row[f"indices{g}"][t])
        if 0 < idx <= waveforms_by_group[g].shape[0]:
            seq.append((waveforms_by_group[g][idx - 1], g))
    return seq


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_features(df, nGroups, nChannelsPerGroup):
    """Flatten variable-length spike sequences into fixed-size feature vectors.

    For each sample we compute per-shank summary statistics:
      - spike count
      - mean / std of waveform amplitudes (across channels and time)
      - mean / std of peak-to-peak amplitude per channel

    This gives a reproducible, fixed-length feature vector regardless of
    sequence length.
    """
    features = []
    for idx in range(len(df)):
        row = df.iloc[idx]
        seq = reconstruct_sequence(row, nGroups, nChannelsPerGroup)

        feat = []
        for g in range(nGroups):
            # Collect waveforms for this shank
            wfs = [wf for wf, sid in seq if sid == g]
            n_spikes = len(wfs)
            feat.append(n_spikes)

            if n_spikes > 0:
                wfs_arr = np.stack(wfs)  # (n_spikes, n_channels, 32)
                # Global mean/std of all waveform values
                feat.append(wfs_arr.mean())
                feat.append(wfs_arr.std())
                # Mean peak-to-peak per channel, averaged over spikes
                ptp = wfs_arr.max(axis=2) - wfs_arr.min(axis=2)  # (n_spikes, n_ch)
                feat.append(ptp.mean())
                feat.append(ptp.std())
                # Mean waveform energy
                feat.append((wfs_arr ** 2).mean())
            else:
                feat.extend([0.0] * 5)

        # Total spike count + ratio per shank
        total = sum(1 for _ in seq)
        feat.append(total)
        for g in range(nGroups):
            g_count = sum(1 for _, sid in seq if sid == g)
            feat.append(g_count / max(total, 1))

        features.append(feat)

        if (idx + 1) % 2000 == 0:
            print(f"  Extracted {idx + 1}/{len(df)} samples")

    return np.array(features, dtype=np.float32)


# ── Evaluation helper ──────────────────────────────────────────────────────────

def evaluate(name, y_true, y_pred, zone_true, positions_true, elapsed=None):
    """Compute and print all metrics for a baseline."""
    eucl = np.sqrt(((y_true - y_pred) ** 2).sum(axis=1))
    mse_x = mean_squared_error(y_true[:, 0], y_pred[:, 0])
    mse_y = mean_squared_error(y_true[:, 1], y_pred[:, 1])
    r2_x = r2_score(y_true[:, 0], y_pred[:, 0])
    r2_y = r2_score(y_true[:, 1], y_pred[:, 1])
    mae_x = mean_absolute_error(y_true[:, 0], y_pred[:, 0])
    mae_y = mean_absolute_error(y_true[:, 1], y_pred[:, 1])

    # Zone accuracy from predicted positions
    pred_curv_d, pred_zones = compute_geometry(y_pred)
    zone_acc = accuracy_score(zone_true, pred_zones)

    # Corridor adherence
    dist_to_skel = np.array([
        compute_distance_to_skeleton(y_pred[i, 0], y_pred[i, 1])
        for i in range(len(y_pred))
    ])
    pct_in_corridor = (dist_to_skel <= cfg.CORRIDOR_HALF_WIDTH).mean()

    time_str = f"{elapsed:.1f}s" if elapsed else "—"

    results = {
        "name": name,
        "eucl_mean": eucl.mean(),
        "eucl_median": np.median(eucl),
        "eucl_p90": np.percentile(eucl, 90),
        "mse_x": mse_x,
        "mse_y": mse_y,
        "r2_x": r2_x,
        "r2_y": r2_y,
        "mae_x": mae_x,
        "mae_y": mae_y,
        "zone_acc": zone_acc,
        "corridor": pct_in_corridor,
        "time": time_str,
    }

    print(f"\n{'─'*60}")
    print(f"  {name}")
    print(f"{'─'*60}")
    print(f"  Euclidean error : mean={eucl.mean():.4f}  "
          f"median={np.median(eucl):.4f}  p90={np.percentile(eucl, 90):.4f}")
    print(f"  MSE             : X={mse_x:.5f}  Y={mse_y:.5f}")
    print(f"  MAE             : X={mae_x:.4f}  Y={mae_y:.4f}")
    print(f"  R²              : X={r2_x:.4f}  Y={r2_y:.4f}")
    print(f"  Zone accuracy   : {zone_acc:.1%}")
    print(f"  Corridor adhere : {pct_in_corridor:.1%}")
    print(f"  Time            : {time_str}")

    return results


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    # ── Load data (same as train.py) ──────────────────────────────────────────
    df_moving, nGroups, nChannelsPerGroup = load_data_light()
    positions = np.array(
        [[x[0], x[1]] for x in df_moving["pos"]], dtype=np.float32
    )
    curvilinear_d, zone_labels = compute_geometry(positions)

    # ── Same temporal 90/10 split ─────────────────────────────────────────────
    split_idx = int(len(df_moving) * 0.9)
    df_train = df_moving.iloc[:split_idx].reset_index(drop=True)
    df_test = df_moving.iloc[split_idx:].reset_index(drop=True)

    y_train = positions[:split_idx]
    y_test = positions[split_idx:]
    zone_train = zone_labels[:split_idx]
    zone_test = zone_labels[split_idx:]

    print(f"\nTrain: {len(df_train)}  |  Test: {len(df_test)}")

    # ── Feature extraction ────────────────────────────────────────────────────
    print("\nExtracting features (train)...")
    t0 = time.time()
    X_train = extract_features(df_train, nGroups, nChannelsPerGroup)
    print(f"Extracting features (test)...")
    X_test = extract_features(df_test, nGroups, nChannelsPerGroup)
    feat_time = time.time() - t0
    print(f"Feature extraction done: {feat_time:.1f}s  |  "
          f"Feature dim: {X_train.shape[1]}")

    # Replace NaN/inf
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    all_results = []

    # ══════════════════════════════════════════════════════════════════════════
    # Baseline 1: Naive (predict training mean)
    # ══════════════════════════════════════════════════════════════════════════
    t0 = time.time()
    y_pred_naive = np.tile(y_train.mean(axis=0), (len(y_test), 1))
    elapsed = time.time() - t0
    all_results.append(evaluate("Naive (train mean)", y_test, y_pred_naive,
                                zone_test, positions[split_idx:], elapsed))

    # ══════════════════════════════════════════════════════════════════════════
    # Baseline 2: Ridge regression
    # ══════════════════════════════════════════════════════════════════════════
    print("\nTraining Ridge regression...")
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    t0 = time.time()
    ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
    ridge.fit(X_train_sc, y_train)
    y_pred_ridge = ridge.predict(X_test_sc)
    elapsed = time.time() - t0
    print(f"  Best alpha: {ridge.alpha_}")
    all_results.append(evaluate("Ridge regression", y_test, y_pred_ridge,
                                zone_test, positions[split_idx:], elapsed))

    # ══════════════════════════════════════════════════════════════════════════
    # Baseline 3: k-NN + PCA-80
    # ══════════════════════════════════════════════════════════════════════════
    print("\nTraining k-NN + PCA-80...")
    n_components = min(80, X_train_sc.shape[1])
    pca = PCA(n_components=n_components, random_state=cfg.SEED)
    X_train_pca = pca.fit_transform(X_train_sc)
    X_test_pca = pca.transform(X_test_sc)
    print(f"  PCA: {X_train_sc.shape[1]} → {n_components} components "
          f"(explained variance: {pca.explained_variance_ratio_.sum():.1%})")

    t0 = time.time()
    knn = KNeighborsRegressor(n_neighbors=5, weights="distance", n_jobs=-1)
    knn.fit(X_train_pca, y_train)
    y_pred_knn = knn.predict(X_test_pca)
    elapsed = time.time() - t0
    all_results.append(evaluate("k-NN + PCA-80", y_test, y_pred_knn,
                                zone_test, positions[split_idx:], elapsed))

    # ══════════════════════════════════════════════════════════════════════════
    # Load Transformer results if available
    # ══════════════════════════════════════════════════════════════════════════
    transformer_preds = cfg.OUTPUTS_DIR / "preds_ensemble.npy"
    if transformer_preds.exists():
        y_pred_tx = np.load(transformer_preds)
        all_results.append(evaluate("Spike Transformer (ensemble)",
                                    y_test, y_pred_tx, zone_test,
                                    positions[split_idx:]))

    # ══════════════════════════════════════════════════════════════════════════
    # Summary comparison table
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n\n{'='*80}")
    print(f"  BASELINE COMPARISON TABLE")
    print(f"{'='*80}")
    header = (f"  {'Model':<30s} {'Eucl↓':>8s} {'MSE_x↓':>8s} {'MSE_y↓':>8s} "
              f"{'R²_x↑':>7s} {'R²_y↑':>7s} {'Zone↑':>7s} {'Corr↑':>7s} {'Time':>6s}")
    print(header)
    print(f"  {'─'*len(header)}")
    for r in all_results:
        print(f"  {r['name']:<30s} {r['eucl_mean']:>8.4f} {r['mse_x']:>8.5f} "
              f"{r['mse_y']:>8.5f} {r['r2_x']:>7.4f} {r['r2_y']:>7.4f} "
              f"{r['zone_acc']:>6.1%} {r['corridor']:>6.1%} {r['time']:>6s}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

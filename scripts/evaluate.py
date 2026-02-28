"""Ensemble evaluation and figure generation.

Loads the 5 trained fold models from checkpoints/, runs ensemble predictions
on the held-out test set, computes all metrics, and saves figures.

Usage (from repo root)
----------------------
    python scripts/evaluate.py

Outputs (figures/evaluation/)
------------------------------
01_scatter_pred_vs_true.png      — X, Y, d scatter plots
02_trajectory_uncertainty.png    — 2D trajectory + X/Y uncertainty bands + calibration
03_spatial_heatmaps.png          — error / sigma / corridor-distance heatmaps
04_corridor_adherence.png        — skeleton adherence scatter + histogram
05_zone_dynamics.png             — zone trajectory + P(zone) over time + d curves
06_zone_heatmaps.png             — zone / error / confidence heatmaps
07_confusion_matrix.png          — 3-class confusion matrix (raw + normalised)
08_uncertainty_decomposition.png — aleatoric vs epistemic uncertainty
09_fold_agreement.png            — inter-fold zone prediction agreement
10_sigma_distribution.png        — predicted sigma distributions for X and Y
11_uncertainty_calibration.png   — spatial calibration heatmap (error vs sigma)
12_zone_error_violin.png  [NEW]  — per-zone error distribution violin plots
13_calibration_curve.png  [NEW]  — reliability diagram (uncertainty calibration curve)

Outputs (outputs/)
------------------
preds_ensemble.npy     — ensemble mean position predictions
sigma_ensemble.npy     — ensemble uncertainty
d_pred_ensemble.npy    — ensemble curvilinear distance predictions
zone_pred.npy          — predicted zone labels
probs_ensemble.npy     — zone softmax probabilities
y_test.npy             — true positions
d_test.npy             — true curvilinear distances
zone_test.npy          — true zone labels
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.metrics import confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader

import src.config as cfg
from src.dataset import SpikeSequenceDataset, collate_fn, load_data
from src.geometry import (
    compute_distance_to_skeleton,
    compute_geometry,
    D_LEFT_END,
    D_RIGHT_START,
)
from src.losses import FeasibilityLoss
from src.model import build_model
from src.trainer import eval_epoch

# ── Reproducibility ───────────────────────────────────────────────────────────
np.random.seed(cfg.SEED)
torch.manual_seed(cfg.SEED)

# ── Device ────────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
print(f"Device: {DEVICE}")

FIG_DIR = cfg.FIGURES_DIR / "evaluation"
FIG_DIR.mkdir(parents=True, exist_ok=True)
cfg.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

ZONE_COLORS = ["#4C72B0", "#DD8452", "#55A868"]


def _draw_skeleton(ax, lw=1.5, color="black", alpha=0.4, linestyle="--"):
    for x1, y1, x2, y2 in cfg.SKELETON_SEGMENTS:
        ax.plot([x1, x2], [y1, y2], color=color, lw=lw,
                alpha=alpha, linestyle=linestyle)


def _heatmap(ax, positions, values, nbins=20, cmap="RdYlGn_r", title="", clabel=""):
    x_edges = np.linspace(0, 1, nbins + 1)
    y_edges = np.linspace(0, 1, nbins + 1)
    val_map   = np.full((nbins, nbins), np.nan)
    count_map = np.zeros((nbins, nbins))
    for i in range(len(positions)):
        xi = int(np.clip(np.searchsorted(x_edges, positions[i, 0]) - 1, 0, nbins - 1))
        yi = int(np.clip(np.searchsorted(y_edges, positions[i, 1]) - 1, 0, nbins - 1))
        val_map[yi, xi] = 0.0 if np.isnan(val_map[yi, xi]) else val_map[yi, xi]
        val_map[yi, xi]   += values[i]
        count_map[yi, xi] += 1
    mean_map = np.where(count_map > 0, val_map / count_map, np.nan)
    im = ax.imshow(mean_map, origin="lower", aspect="equal",
                   cmap=cmap, extent=[0, 1, 0, 1])
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_title(title)
    plt.colorbar(im, ax=ax, label=clabel)
    return im


# ════════════════════════════════════════════════════════════════════════════
def main():
    # ── Load data ─────────────────────────────────────────────────────────────
    df_moving, nGroups, nChannelsPerGroup = load_data()
    positions                             = np.array(
        [[x[0], x[1]] for x in df_moving["pos"]], dtype=np.float32
    )
    curvilinear_d, zone_labels = compute_geometry(positions)

    split_idx = int(len(df_moving) * 0.9)
    df_test   = df_moving.iloc[split_idx:].reset_index(drop=True)
    d_test    = curvilinear_d[split_idx:]
    zone_test = zone_labels[split_idx:]

    test_ds     = SpikeSequenceDataset(df_test, nGroups, nChannelsPerGroup, d_test, zone_test)
    test_loader = DataLoader(test_ds, batch_size=cfg.BATCH_SIZE,
                             shuffle=False, collate_fn=collate_fn, num_workers=0)
    print(f"Test set: {len(test_ds)} samples  ({len(test_loader)} batches)")

    # ── Ensemble inference ────────────────────────────────────────────────────
    criterion_ce  = nn.CrossEntropyLoss()
    criterion_nll = nn.GaussianNLLLoss()
    criterion_d   = nn.MSELoss()
    feas_loss_fn  = FeasibilityLoss(cfg.SKELETON_SEGMENTS, cfg.CORRIDOR_HALF_WIDTH).to(DEVICE)

    all_fold_mu, all_fold_sigma, all_fold_probs, all_fold_d = [], [], [], []

    for fold in range(cfg.N_FOLDS):
        ckpt = cfg.CHECKPOINTS_DIR / f"best_model_fold{fold + 1}.pt"
        model = build_model(nGroups, nChannelsPerGroup).to(DEVICE)
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))

        _, fold_acc, (fold_mu, fold_sigma, fold_probs, fold_d,
                      y_test, d_test_arr, zone_test_arr) = eval_epoch(
            model, test_loader,
            criterion_ce, criterion_nll, criterion_d, feas_loss_fn, DEVICE,
        )
        all_fold_mu.append(fold_mu)
        all_fold_sigma.append(fold_sigma)
        all_fold_probs.append(fold_probs)
        all_fold_d.append(fold_d)

        fold_eucl = np.sqrt(((y_test - fold_mu) ** 2).sum(axis=1))
        print(f"Fold {fold+1}: Eucl={fold_eucl.mean():.4f}  cls={fold_acc:.1%}")

    all_fold_mu    = np.stack(all_fold_mu)    # (5, N, 2)
    all_fold_sigma = np.stack(all_fold_sigma) # (5, N, 2)
    all_fold_probs = np.stack(all_fold_probs) # (5, N, 3)
    all_fold_d     = np.stack(all_fold_d)     # (5, N, 1)

    # Ensemble aggregation
    y_pred           = all_fold_mu.mean(axis=0)
    d_pred_ensemble  = all_fold_d.mean(axis=0).squeeze()
    probs_ensemble   = all_fold_probs.mean(axis=0)
    zone_pred        = probs_ensemble.argmax(axis=1)

    # Total uncertainty = aleatoric (mean sigma²) + epistemic (variance of mu)
    mean_var  = (all_fold_sigma ** 2).mean(axis=0)
    var_mu    = all_fold_mu.var(axis=0)
    y_sigma   = np.sqrt(mean_var + var_mu)

    zone_test_targets = zone_test_arr
    d_test_targets    = d_test_arr

    # ── Metrics ───────────────────────────────────────────────────────────────
    eucl_errors     = np.sqrt(((y_test - y_pred) ** 2).sum(axis=1))
    r2_x  = r2_score(y_test[:, 0], y_pred[:, 0])
    r2_y  = r2_score(y_test[:, 1], y_pred[:, 1])
    d_mae = np.abs(d_test_targets - d_pred_ensemble).mean()
    d_r2  = r2_score(d_test_targets, d_pred_ensemble)
    cls_accuracy      = (zone_pred == zone_test_targets).mean()
    zone_confusion    = zone_pred != zone_test_targets
    sigma_mean        = (y_sigma[:, 0] + y_sigma[:, 1]) / 2

    test_dist_to_skel = np.array([
        compute_distance_to_skeleton(y_pred[i, 0], y_pred[i, 1])
        for i in range(len(y_pred))
    ])
    pct_outside = (test_dist_to_skel > cfg.CORRIDOR_HALF_WIDTH).mean()

    print(f"\n{'='*60}\nENSEMBLE RESULTS ({cfg.N_FOLDS} folds)\n{'='*60}")
    print(f"  MSE  : X={mean_squared_error(y_test[:,0], y_pred[:,0]):.5f}  "
          f"Y={mean_squared_error(y_test[:,1], y_pred[:,1]):.5f}")
    print(f"  MAE  : X={mean_absolute_error(y_test[:,0], y_pred[:,0]):.4f}  "
          f"Y={mean_absolute_error(y_test[:,1], y_pred[:,1]):.4f}")
    print(f"  R²   : X={r2_x:.4f}  Y={r2_y:.4f}")
    print(f"  Eucl : mean={eucl_errors.mean():.4f}  "
          f"median={np.median(eucl_errors):.4f}  "
          f"p90={np.percentile(eucl_errors, 90):.4f}")
    print(f"  d    : MAE={d_mae:.4f}  R²={d_r2:.4f}")
    print(f"  Zone : accuracy={cls_accuracy:.1%}  outside={pct_outside:.1%}")

    in_1s = (eucl_errors < sigma_mean).mean()
    in_2s = (eucl_errors < 2 * sigma_mean).mean()
    in_3s = (eucl_errors < 3 * sigma_mean).mean()
    print(f"  Calibration : <1σ={in_1s:.1%}  <2σ={in_2s:.1%}  <3σ={in_3s:.1%}")

    # ── Per-zone metrics ──────────────────────────────────────────────────────
    print("\n  Per-zone breakdown:")
    for z, name in enumerate(cfg.ZONE_NAMES):
        zmask = zone_test_targets == z
        if zmask.any():
            z_eucl = eucl_errors[zmask]
            z_acc  = (zone_pred[zmask] == z).mean()
            print(f"    {name:6s} : Eucl={z_eucl.mean():.4f}±{z_eucl.std():.4f}  "
                  f"cls={z_acc:.1%}  (n={zmask.sum()})")

    # ══════════════════════════════════════════════════════════════════════════
    # FIGURES
    # ══════════════════════════════════════════════════════════════════════════

    # ── 01: Scatter pred vs true ──────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (true, pred, label, r2) in zip(axes, [
        (y_test[:, 0], y_pred[:, 0], "X position",         r2_x),
        (y_test[:, 1], y_pred[:, 1], "Y position",         r2_y),
        (d_test_targets, d_pred_ensemble, "Curvilinear d", d_r2),
    ]):
        ax.scatter(true, pred, s=1, alpha=0.3, color="steelblue")
        ax.plot([0, 1], [0, 1], "r--", linewidth=2)
        ax.set_xlabel(f"True {label}"); ax.set_ylabel(f"Predicted {label}")
        ax.set_title(f"{label}  (R²={r2:.3f})"); ax.set_aspect("equal")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "01_scatter_pred_vs_true.png", dpi=150)
    plt.close(fig)

    # ── 02: Trajectory + uncertainty bands ───────────────────────────────────
    seg     = slice(0, 500)
    seg_idx = np.arange(500)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    colors_pts = np.arange(500)
    axes[0, 0].scatter(y_test[seg, 0], y_test[seg, 1],
                       c=colors_pts, cmap="winter", s=8, alpha=0.6, label="True")
    sc = axes[0, 0].scatter(y_pred[seg, 0], y_pred[seg, 1],
                             c=colors_pts, cmap="autumn", s=8, alpha=0.6,
                             marker="x", label="Predicted")
    _draw_skeleton(axes[0, 0])
    axes[0, 0].set_xlabel("X"); axes[0, 0].set_ylabel("Y")
    axes[0, 0].set_title("2D positions – first 500 test points")
    axes[0, 0].legend(); axes[0, 0].set_aspect("equal")
    plt.colorbar(sc, ax=axes[0, 0], label="Temporal index")

    for ax, dim, label in [
        (axes[0, 1], 0, "X"),
        (axes[1, 0], 1, "Y"),
    ]:
        ax.plot(seg_idx, y_test[seg, dim], "b-", label=f"True {label}", linewidth=1.5)
        ax.plot(seg_idx, y_pred[seg, dim], "r-", alpha=0.7,
                label="Predicted (μ)", linewidth=1)
        ax.fill_between(
            seg_idx,
            y_pred[seg, dim] - 2 * y_sigma[seg, dim],
            y_pred[seg, dim] + 2 * y_sigma[seg, dim],
            alpha=0.2, color="red", label="2σ uncertainty",
        )
        ax.set_xlabel("Index"); ax.set_ylabel(f"Position {label}")
        ax.set_title(f"Position {label} with uncertainty"); ax.legend()

    axes[1, 1].scatter(sigma_mean, eucl_errors, s=1, alpha=0.3, color="steelblue")
    srange = np.linspace(0, sigma_mean.max(), 100)
    axes[1, 1].plot(srange, 2 * srange, "r--", label="y = 2σ", linewidth=1.5)
    axes[1, 1].set_xlabel("Predicted mean σ")
    axes[1, 1].set_ylabel("Actual Euclidean error")
    axes[1, 1].set_title("Calibration: uncertainty vs actual error")
    axes[1, 1].legend()

    plt.tight_layout()
    fig.savefig(FIG_DIR / "02_trajectory_uncertainty.png", dpi=150)
    plt.close(fig)

    # ── 03: Spatial heatmaps ──────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    _heatmap(axes[0], y_test,  eucl_errors,                 cmap="RdYlGn_r",
             title="Mean Euclidean error",         clabel="Error")
    _heatmap(axes[1], y_test,  sigma_mean,                  cmap="RdYlGn_r",
             title="Mean predicted σ",             clabel="σ")
    _heatmap(axes[2], y_pred,  test_dist_to_skel,           cmap="Reds",
             title="Distance to skeleton (predictions)", clabel="Distance")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "03_spatial_heatmaps.png", dpi=150)
    plt.close(fig)

    # ── 04: Corridor adherence ────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    sc = axes[0].scatter(y_pred[:, 0], y_pred[:, 1],
                         c=test_dist_to_skel, cmap="Reds", s=2, alpha=0.5,
                         vmin=0, vmax=0.3)
    _draw_skeleton(axes[0], lw=2, color="blue", alpha=0.8, linestyle="-")
    axes[0].set_xlabel("X"); axes[0].set_ylabel("Y")
    axes[0].set_title("Predictions coloured by distance to skeleton")
    axes[0].set_aspect("equal")
    plt.colorbar(sc, ax=axes[0], label="Distance to skeleton")

    axes[1].hist(test_dist_to_skel, bins=50, alpha=0.7, edgecolor="white",
                 color="steelblue")
    axes[1].axvline(cfg.CORRIDOR_HALF_WIDTH, color="red", linestyle="--",
                    linewidth=2, label=f"Corridor threshold ({cfg.CORRIDOR_HALF_WIDTH})")
    axes[1].set_xlabel("Distance to skeleton"); axes[1].set_ylabel("Count")
    axes[1].set_title(f"Corridor adherence  (outside: {pct_outside:.1%})")
    axes[1].legend()
    plt.tight_layout()
    fig.savefig(FIG_DIR / "04_corridor_adherence.png", dpi=150)
    plt.close(fig)

    # ── 05: Zone dynamics ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    for z, (name, color) in enumerate(zip(cfg.ZONE_NAMES, ZONE_COLORS)):
        zmask = zone_pred == z
        axes[0, 0].scatter(y_pred[zmask, 0], y_pred[zmask, 1],
                           c=color, s=2, alpha=0.3, label=name)
    _draw_skeleton(axes[0, 0])
    axes[0, 0].set_xlabel("X"); axes[0, 0].set_ylabel("Y")
    axes[0, 0].set_title("Predictions coloured by predicted zone")
    axes[0, 0].legend(markerscale=10); axes[0, 0].set_aspect("equal")

    for z, (name, color) in enumerate(zip(cfg.ZONE_NAMES, ZONE_COLORS)):
        axes[0, 1].plot(seg_idx, probs_ensemble[seg, z], label=name,
                        color=color, linewidth=1)
    axes[0, 1].plot(seg_idx, zone_test_targets[seg] / 2, "k--",
                    alpha=0.3, label="True zone (scaled)")
    axes[0, 1].set_xlabel("Index"); axes[0, 1].set_ylabel("P(zone)")
    axes[0, 1].set_title("Zone probabilities over time (500 pts)")
    axes[0, 1].legend(); axes[0, 1].set_ylim(-0.05, 1.05)

    axes[1, 0].plot(seg_idx, d_test_targets[seg], "b-",
                    label="True d", linewidth=1.5)
    axes[1, 0].plot(seg_idx, d_pred_ensemble[seg], "r-",
                    alpha=0.7, label="Predicted d", linewidth=1)
    axes[1, 0].axhline(D_LEFT_END,    color=ZONE_COLORS[0], linestyle=":",
                       alpha=0.7, label=f"Left|Top  ({D_LEFT_END:.3f})")
    axes[1, 0].axhline(D_RIGHT_START, color=ZONE_COLORS[2], linestyle=":",
                       alpha=0.7, label=f"Top|Right ({D_RIGHT_START:.3f})")
    axes[1, 0].set_xlabel("Index"); axes[1, 0].set_ylabel("Curvilinear d")
    axes[1, 0].set_title("Curvilinear distance d (500 pts)"); axes[1, 0].legend()

    correct = ~zone_confusion
    axes[1, 1].scatter(y_test[correct, 0], y_test[correct, 1],
                       c="green", s=1, alpha=0.2,
                       label=f"Correct ({correct.mean():.1%})")
    if zone_confusion.any():
        axes[1, 1].scatter(y_test[zone_confusion, 0], y_test[zone_confusion, 1],
                           c="red", s=5, alpha=0.8,
                           label=f"Error ({zone_confusion.mean():.1%})")
    _draw_skeleton(axes[1, 1])
    axes[1, 1].set_xlabel("X"); axes[1, 1].set_ylabel("Y")
    axes[1, 1].set_title("Zone classification errors (red = wrong)")
    axes[1, 1].legend(markerscale=5); axes[1, 1].set_aspect("equal")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "05_zone_dynamics.png", dpi=150)
    plt.close(fig)

    # ── 06: Zone heatmaps ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    _heatmap(axes[0], y_test, zone_pred.astype(float),      cmap="RdYlBu",
             title="Predicted zone (argmax)", clabel="Zone index")
    _heatmap(axes[1], y_test, eucl_errors,                   cmap="RdYlGn_r",
             title="Euclidean error", clabel="Error")
    _heatmap(axes[2], y_test, probs_ensemble.max(axis=1),    cmap="RdYlGn",
             title="Zone confidence (max prob)", clabel="Confidence")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "06_zone_heatmaps.png", dpi=150)
    plt.close(fig)

    # ── 07: Confusion matrix ─────────────────────────────────────────────────
    cm      = confusion_matrix(zone_test_targets, zone_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, data, fmt, title in [
        (axes[0], cm,      "{:d}",   "Confusion matrix (counts)"),
        (axes[1], cm_norm, "{:.1%}", "Confusion matrix (normalised)"),
    ]:
        im = ax.imshow(data, cmap="Blues", vmin=0,
                       vmax=(None if fmt == "{:d}" else 1))
        for i in range(cfg.N_ZONES):
            for j in range(cfg.N_ZONES):
                val   = data[i, j]
                color = "white" if val > (cm.max() / 2 if fmt == "{:d}" else 0.5) else "black"
                ax.text(j, i, fmt.format(val), ha="center", va="center",
                        fontsize=13, color=color)
        ax.set_xticks(range(cfg.N_ZONES)); ax.set_xticklabels(cfg.ZONE_NAMES)
        ax.set_yticks(range(cfg.N_ZONES)); ax.set_yticklabels(cfg.ZONE_NAMES)
        ax.set_xlabel("Predicted zone"); ax.set_ylabel("True zone")
        ax.set_title(title); plt.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "07_confusion_matrix.png", dpi=150)
    plt.close(fig)

    # ── 08: Uncertainty decomposition ────────────────────────────────────────
    aleatoric = np.sqrt((all_fold_sigma ** 2).mean(axis=0)).mean(axis=1)
    epistemic = np.sqrt(var_mu).mean(axis=1)
    total_unc = np.sqrt(aleatoric ** 2 + epistemic ** 2)
    fold_std_eucl = np.sqrt(all_fold_mu.std(axis=0) ** 2).mean(axis=1)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    sc = axes[0, 0].scatter(y_test[:, 0], y_test[:, 1],
                             c=fold_std_eucl, cmap="Reds", s=2, alpha=0.5)
    axes[0, 0].set_title("Epistemic uncertainty (inter-fold variance)")
    axes[0, 0].set_xlabel("X"); axes[0, 0].set_ylabel("Y")
    axes[0, 0].set_aspect("equal")
    plt.colorbar(sc, ax=axes[0, 0], label="Std of fold μ")

    max_val = max(aleatoric.max(), epistemic.max())
    axes[0, 1].scatter(aleatoric, epistemic, s=1, alpha=0.3, color="steelblue")
    axes[0, 1].plot([0, max_val], [0, max_val], "r--", label="y=x")
    axes[0, 1].set_xlabel("Aleatoric uncertainty (mean σ)")
    axes[0, 1].set_ylabel("Epistemic uncertainty (std of μ)")
    axes[0, 1].set_title("Aleatoric vs Epistemic"); axes[0, 1].legend()

    axes[1, 0].hist(aleatoric, bins=50, alpha=0.7, label="Aleatoric", color="steelblue")
    axes[1, 0].hist(epistemic, bins=50, alpha=0.7, label="Epistemic", color="coral")
    axes[1, 0].set_xlabel("Uncertainty"); axes[1, 0].set_ylabel("Count")
    axes[1, 0].set_title("Uncertainty distribution"); axes[1, 0].legend()

    u_range = np.linspace(0, total_unc.max(), 100)
    axes[1, 1].scatter(total_unc, eucl_errors, s=1, alpha=0.3, color="steelblue")
    axes[1, 1].plot(u_range, 2 * u_range, "r--", label="y = 2σ_total")
    axes[1, 1].set_xlabel("Total uncertainty (√(alea² + epis²))")
    axes[1, 1].set_ylabel("Euclidean error")
    axes[1, 1].set_title("Total uncertainty calibration"); axes[1, 1].legend()

    plt.tight_layout()
    fig.savefig(FIG_DIR / "08_uncertainty_decomposition.png", dpi=150)
    plt.close(fig)

    corr_alea, _ = spearmanr(aleatoric, eucl_errors)
    corr_epis, _ = spearmanr(epistemic, eucl_errors)
    corr_tot,  _ = spearmanr(total_unc, eucl_errors)
    print(f"\n  Spearman(uncertainty, error): "
          f"aleatoric={corr_alea:.3f}  epistemic={corr_epis:.3f}  total={corr_tot:.3f}")

    # ── 09: Inter-fold agreement ──────────────────────────────────────────────
    fold_zone_preds = np.stack([fp.argmax(axis=1) for fp in all_fold_probs])
    zone_agreement  = np.array([
        np.bincount(fold_zone_preds[:, i], minlength=3).max() / cfg.N_FOLDS
        for i in range(len(y_test))
    ])

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    sc = axes[0].scatter(y_test[:, 0], y_test[:, 1],
                         c=zone_agreement, cmap="RdYlGn",
                         s=2, alpha=0.5, vmin=0.4, vmax=1.0)
    _draw_skeleton(axes[0])
    axes[0].set_xlabel("X"); axes[0].set_ylabel("Y")
    axes[0].set_title(f"Inter-fold zone agreement ({cfg.N_FOLDS} folds)")
    axes[0].set_aspect("equal")
    plt.colorbar(sc, ax=axes[0], label="Agreement fraction")

    unique_vals = [i / cfg.N_FOLDS for i in range(1, cfg.N_FOLDS + 1)]
    labels  = [f"{i}/{cfg.N_FOLDS}" for i in range(1, cfg.N_FOLDS + 1)]
    counts  = [(np.abs(zone_agreement - v) < 0.01).sum() for v in unique_vals]
    bar_colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, cfg.N_FOLDS))
    bars = axes[1].bar(labels, counts, color=bar_colors)
    for bar, c in zip(bars, counts):
        axes[1].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 5, str(c), ha="center")
    axes[1].set_xlabel("Agreement"); axes[1].set_ylabel("Number of points")
    axes[1].set_title("Zone agreement distribution across folds")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "09_fold_agreement.png", dpi=150)
    plt.close(fig)
    print(f"  Inter-fold perfect agreement: {(zone_agreement == 1.0).mean():.1%}")

    # ── 10: Sigma distribution ────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, dim, color, label in [
        (axes[0], 0, "steelblue", "X"),
        (axes[1], 1, "coral",     "Y"),
    ]:
        ax.hist(y_sigma[:, dim], bins=50, alpha=0.7, color=color, edgecolor="white")
        ax.axvline(y_sigma[:, dim].mean(), color="red", linestyle="--",
                   label=f"Mean = {y_sigma[:, dim].mean():.4f}")
        ax.set_xlabel(f"σ {label}"); ax.set_ylabel("Count")
        ax.set_title(f"Predicted uncertainty on {label}"); ax.legend()
    plt.tight_layout()
    fig.savefig(FIG_DIR / "10_sigma_distribution.png", dpi=150)
    plt.close(fig)

    # ── 11: Uncertainty calibration heatmap ───────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    _heatmap(axes[0], y_test, eucl_errors, cmap="RdYlGn_r",
             title="Actual error (binned by true position)", clabel="Error")
    _heatmap(axes[1], y_test, sigma_mean,  cmap="RdYlGn_r",
             title="Predicted σ (binned by true position)", clabel="σ")
    corr, pval = spearmanr(sigma_mean, eucl_errors)
    plt.suptitle(f"Spatial calibration — Spearman(σ, error) = {corr:.3f}  (p={pval:.1e})",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "11_uncertainty_calibration.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ════════════════════════════════════════════════════════════════════════
    # NEW FIGURES
    # ════════════════════════════════════════════════════════════════════════

    # ── 12 [NEW]: Per-zone error violin plot ─────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    zone_eucl_by_class = [eucl_errors[zone_test_targets == z] for z in range(cfg.N_ZONES)]

    vp = axes[0].violinplot(zone_eucl_by_class, positions=range(cfg.N_ZONES),
                            showmedians=True, showextrema=True)
    for i, (body, color) in enumerate(zip(vp["bodies"], ZONE_COLORS)):
        body.set_facecolor(color)
        body.set_alpha(0.6)
    vp["cmedians"].set_color("black")
    axes[0].set_xticks(range(cfg.N_ZONES))
    axes[0].set_xticklabels(cfg.ZONE_NAMES)
    axes[0].set_ylabel("Euclidean error")
    axes[0].set_title("Error distribution by zone")
    axes[0].grid(True, axis="y", alpha=0.3)

    # Overlay per-zone mean ± std
    for z, (data, color) in enumerate(zip(zone_eucl_by_class, ZONE_COLORS)):
        axes[0].scatter(z, np.median(data), color="black", zorder=5, s=30)

    # Right: error quantiles per zone (p25, p50, p75, p90)
    quantiles = [25, 50, 75, 90]
    x = np.arange(len(quantiles))
    width = 0.22
    for z, (name, color) in enumerate(zip(cfg.ZONE_NAMES, ZONE_COLORS)):
        data = zone_eucl_by_class[z]
        vals = [np.percentile(data, q) for q in quantiles]
        axes[1].bar(x + z * width, vals, width, label=name, color=color, alpha=0.8)
    axes[1].set_xticks(x + width)
    axes[1].set_xticklabels([f"p{q}" for q in quantiles])
    axes[1].set_ylabel("Euclidean error")
    axes[1].set_title("Error percentiles by zone")
    axes[1].legend(); axes[1].grid(True, axis="y", alpha=0.3)

    plt.suptitle("Per-zone position decoding error", fontsize=13, y=1.01)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "12_zone_error_violin.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {FIG_DIR / '12_zone_error_violin.png'}")

    # ── 13 [NEW]: Calibration reliability diagram ─────────────────────────────
    # Bin predicted sigma and compute fraction of points with error < sigma
    n_bins  = 15
    sig_min = np.percentile(sigma_mean, 1)
    sig_max = np.percentile(sigma_mean, 99)
    bin_edges   = np.linspace(sig_min, sig_max, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    frac_1s, frac_2s, frac_3s, counts_per_bin = [], [], [], []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (sigma_mean >= lo) & (sigma_mean < hi)
        if mask.sum() == 0:
            frac_1s.append(np.nan); frac_2s.append(np.nan); frac_3s.append(np.nan)
            counts_per_bin.append(0)
            continue
        e = eucl_errors[mask]
        s = sigma_mean[mask]
        frac_1s.append((e < s).mean())
        frac_2s.append((e < 2 * s).mean())
        frac_3s.append((e < 3 * s).mean())
        counts_per_bin.append(mask.sum())

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: fraction within 1/2/3 sigma vs sigma bin
    axes[0].plot(bin_centers, frac_1s, "o-", color="steelblue",
                 label="error < 1σ  (ideal ≈39%)")
    axes[0].plot(bin_centers, frac_2s, "s-", color="orange",
                 label="error < 2σ  (ideal ≈86%)")
    axes[0].plot(bin_centers, frac_3s, "^-", color="green",
                 label="error < 3σ  (ideal ≈99%)")
    axes[0].axhline(0.3935, color="steelblue", linestyle="--", alpha=0.4)
    axes[0].axhline(0.8647, color="orange",    linestyle="--", alpha=0.4)
    axes[0].axhline(0.9889, color="green",     linestyle="--", alpha=0.4)
    axes[0].set_xlabel("Predicted σ bin"); axes[0].set_ylabel("Fraction of points covered")
    axes[0].set_title("Reliability diagram — calibration by σ bin")
    axes[0].legend(); axes[0].set_ylim(0, 1.05); axes[0].grid(True, alpha=0.3)

    # Right: cumulative calibration curve
    sorted_idx   = np.argsort(sigma_mean)
    sorted_sigma = sigma_mean[sorted_idx]
    sorted_error = eucl_errors[sorted_idx]
    cum_frac     = np.array([
        (sorted_error[:k] < sorted_sigma[:k]).mean()
        for k in range(1, len(sorted_sigma) + 1)
    ])
    axes[1].plot(sorted_sigma, cum_frac, color="steelblue", linewidth=1, alpha=0.8)
    axes[1].axhline(0.3935, color="gray", linestyle="--", alpha=0.6, label="Ideal 1σ (39%)")
    axes[1].set_xlabel("Predicted σ (sorted)"); axes[1].set_ylabel("Cumulative fraction (error < σ)")
    axes[1].set_title("Cumulative calibration curve")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.suptitle(f"Uncertainty calibration — overall: "
                 f"<1σ={in_1s:.1%}  <2σ={in_2s:.1%}  <3σ={in_3s:.1%}", fontsize=12)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "13_calibration_curve.png", dpi=150)
    plt.close(fig)
    print(f"Saved {FIG_DIR / '13_calibration_curve.png'}")

    # ── Save predictions ──────────────────────────────────────────────────────
    np.save(cfg.OUTPUTS_DIR / "preds_ensemble.npy",    y_pred)
    np.save(cfg.OUTPUTS_DIR / "sigma_ensemble.npy",    y_sigma)
    np.save(cfg.OUTPUTS_DIR / "d_pred_ensemble.npy",   d_pred_ensemble)
    np.save(cfg.OUTPUTS_DIR / "zone_pred.npy",         zone_pred)
    np.save(cfg.OUTPUTS_DIR / "probs_ensemble.npy",    probs_ensemble)
    np.save(cfg.OUTPUTS_DIR / "y_test.npy",            y_test)
    np.save(cfg.OUTPUTS_DIR / "d_test.npy",            d_test_targets)
    np.save(cfg.OUTPUTS_DIR / "zone_test.npy",         zone_test_targets)
    print(f"\nPredictions saved to {cfg.OUTPUTS_DIR}/")
    print(f"Figures saved to     {FIG_DIR}/")


if __name__ == "__main__":
    main()


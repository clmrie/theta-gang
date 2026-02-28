"""5-fold cross-validation training script.

Usage (from repo root)
----------------------
    python scripts/train.py

Outputs
-------
checkpoints/best_model_fold{1..5}.pt   — best model weights per fold
figures/training/01_total_loss.png     — train / val loss curves per fold
figures/training/02_loss_breakdown.png — per-component loss curves
figures/training/03_lr_schedule.png    — OneCycleLR schedule visualisation
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
import torch.optim as optim
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

import src.config as cfg
from src.dataset import SpikeSequenceDataset, collate_fn, load_data
from src.geometry import compute_geometry, print_geometry_stats
from src.losses import FeasibilityLoss
from src.model import build_model
from src.trainer import eval_epoch, train_epoch


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

# ── Create output directories ─────────────────────────────────────────────────
cfg.CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
(cfg.FIGURES_DIR / "training").mkdir(parents=True, exist_ok=True)


def visualise_lr_schedule(n_steps: int, max_lr: float, save_path: Path):
    """Plot the OneCycleLR schedule for one fold."""
    dummy_model = torch.nn.Linear(1, 1)
    dummy_opt   = optim.AdamW(dummy_model.parameters(), lr=max_lr)
    dummy_sch   = optim.lr_scheduler.OneCycleLR(
        dummy_opt, max_lr=max_lr, total_steps=n_steps
    )
    lrs = []
    for _ in range(n_steps):
        lrs.append(dummy_opt.param_groups[0]["lr"])
        dummy_sch.step()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(lrs, color="steelblue", linewidth=1.5)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Learning rate")
    ax.set_title(f"OneCycleLR schedule  (max_lr={max_lr}, {n_steps} steps)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved {save_path}")


def main():
    # ── Load data ─────────────────────────────────────────────────────────────
    df_moving, nGroups, nChannelsPerGroup = load_data()
    positions     = np.array([[x[0], x[1]] for x in df_moving["pos"]], dtype=np.float32)
    curvilinear_d, zone_labels = compute_geometry(positions)
    print_geometry_stats(positions, curvilinear_d, zone_labels)

    # ── Temporal 90/10 train / test split ─────────────────────────────────────
    split_idx     = int(len(df_moving) * 0.9)
    df_train_full = df_moving.iloc[:split_idx].reset_index(drop=True)
    d_train_full  = curvilinear_d[:split_idx]
    zone_train_full = zone_labels[:split_idx]
    print(f"\nTrain: {len(df_train_full)}  |  Test (held out): {len(df_moving) - split_idx}")

    # ── K-Fold setup ──────────────────────────────────────────────────────────
    kf            = KFold(n_splits=cfg.N_FOLDS, shuffle=True, random_state=41)
    colors        = plt.cm.tab10(np.linspace(0, 1, cfg.N_FOLDS))
    fold_results  = []
    all_train_losses, all_val_losses    = {}, {}
    all_train_detail, all_val_detail    = {}, {}

    for fold, (train_idx, val_idx) in enumerate(kf.split(df_train_full)):
        print(f"\n{'='*60}\nFOLD {fold + 1}/{cfg.N_FOLDS}\n{'='*60}")

        df_ft = df_train_full.iloc[train_idx].reset_index(drop=True)
        df_fv = df_train_full.iloc[val_idx].reset_index(drop=True)

        ds_t = SpikeSequenceDataset(df_ft, nGroups, nChannelsPerGroup,
                                    d_train_full[train_idx], zone_train_full[train_idx])
        ds_v = SpikeSequenceDataset(df_fv, nGroups, nChannelsPerGroup,
                                    d_train_full[val_idx],   zone_train_full[val_idx])
        dl_t = DataLoader(ds_t, batch_size=cfg.BATCH_SIZE,
                          shuffle=True,  collate_fn=collate_fn, num_workers=0)
        dl_v = DataLoader(ds_v, batch_size=cfg.BATCH_SIZE,
                          shuffle=False, collate_fn=collate_fn, num_workers=0)
        print(f"  Train: {len(ds_t)}  |  Val: {len(ds_v)}")

        model    = build_model(nGroups, nChannelsPerGroup).to(DEVICE)
        optimizer = optim.AdamW(model.parameters(),
                                lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=cfg.LR,
            epochs=cfg.EPOCHS, steps_per_epoch=len(dl_t),
        )
        criterion_ce  = nn.CrossEntropyLoss()
        criterion_nll = nn.GaussianNLLLoss()
        criterion_d   = nn.MSELoss()
        feas_loss_fn  = FeasibilityLoss(cfg.SKELETON_SEGMENTS, cfg.CORRIDOR_HALF_WIDTH).to(DEVICE)

        # Plot LR schedule (once, for fold 1)
        if fold == 0:
            n_total_steps = cfg.EPOCHS * len(dl_t)
            visualise_lr_schedule(
                n_total_steps, cfg.LR,
                cfg.FIGURES_DIR / "training" / "03_lr_schedule.png",
            )

        best_val_loss     = float("inf")
        patience_counter  = 0
        train_losses, val_losses     = [], []
        train_detail, val_detail     = [], []
        model_path = cfg.CHECKPOINTS_DIR / f"best_model_fold{fold + 1}.pt"

        for epoch in range(cfg.EPOCHS):
            t_losses, t_acc   = train_epoch(
                model, dl_t, optimizer, scheduler,
                criterion_ce, criterion_nll, criterion_d, feas_loss_fn, DEVICE,
            )
            v_losses, v_acc, _ = eval_epoch(
                model, dl_v,
                criterion_ce, criterion_nll, criterion_d, feas_loss_fn, DEVICE,
            )
            train_losses.append(t_losses["loss"])
            val_losses.append(v_losses["loss"])
            train_detail.append(t_losses)
            val_detail.append(v_losses)

            if epoch % 5 == 0 or epoch == cfg.EPOCHS - 1:
                print(
                    f"  Ep {epoch+1:02d}/{cfg.EPOCHS} | "
                    f"Train {t_losses['loss']:.4f} "
                    f"(cls={t_losses['cls']:.4f} pos={t_losses['pos']:.4f} "
                    f"d={t_losses['d']:.5f} feas={t_losses['feas']:.6f} acc={t_acc:.1%}) | "
                    f"Val {v_losses['loss']:.4f} (acc={v_acc:.1%})"
                )

            if v_losses["loss"] < best_val_loss:
                best_val_loss    = v_losses["loss"]
                patience_counter = 0
                torch.save(model.state_dict(), model_path)
            else:
                patience_counter += 1
                if patience_counter >= cfg.PATIENCE:
                    print(f"  Early stopping at epoch {epoch + 1}")
                    break

        all_train_losses[fold] = train_losses
        all_val_losses[fold]   = val_losses
        all_train_detail[fold] = train_detail
        all_val_detail[fold]   = val_detail

        # Evaluate best checkpoint on validation set
        model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
        from sklearn.metrics import r2_score
        _, val_acc, (val_mu, val_sigma, val_probs, val_d_pred,
                     val_targets, val_d_targets, val_zone_targets) = eval_epoch(
            model, dl_v,
            criterion_ce, criterion_nll, criterion_d, feas_loss_fn, DEVICE,
        )
        val_eucl  = np.sqrt(((val_targets - val_mu) ** 2).sum(axis=1))
        val_d_mae = np.abs(val_d_targets - val_d_pred.squeeze()).mean()

        from src.geometry import compute_distance_to_skeleton
        pct_outside = np.mean([
            compute_distance_to_skeleton(val_mu[i, 0], val_mu[i, 1]) > cfg.CORRIDOR_HALF_WIDTH
            for i in range(len(val_mu))
        ])

        fold_results.append({
            "fold":           fold + 1,
            "best_val_loss":  best_val_loss,
            "val_eucl_mean":  val_eucl.mean(),
            "val_r2_x":       r2_score(val_targets[:, 0], val_mu[:, 0]),
            "val_r2_y":       r2_score(val_targets[:, 1], val_mu[:, 1]),
            "val_d_mae":      val_d_mae,
            "val_cls_acc":    val_acc,
            "val_pct_outside": pct_outside,
            "epochs":         len(train_losses),
        })
        r = fold_results[-1]
        print(
            f"  => Eucl={r['val_eucl_mean']:.4f} | "
            f"R2: X={r['val_r2_x']:.4f} Y={r['val_r2_y']:.4f} | "
            f"d_MAE={r['val_d_mae']:.4f} | cls={r['val_cls_acc']:.1%} | "
            f"outside={r['val_pct_outside']:.1%}"
        )

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}\nCROSS-VALIDATION SUMMARY ({cfg.N_FOLDS} folds)\n{'='*60}")
    for r in fold_results:
        print(
            f"  Fold {r['fold']}: Eucl={r['val_eucl_mean']:.4f} | "
            f"R2_X={r['val_r2_x']:.4f} R2_Y={r['val_r2_y']:.4f} | "
            f"d_MAE={r['val_d_mae']:.4f} | cls={r['val_cls_acc']:.1%} | "
            f"epochs={r['epochs']}"
        )
    eucl_vals = [r["val_eucl_mean"] for r in fold_results]
    print(f"\n  Mean Eucl = {np.mean(eucl_vals):.4f} ± {np.std(eucl_vals):.4f}")

    # ── Plot: total loss curves ───────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    for fold in range(cfg.N_FOLDS):
        axes[0].plot(all_train_losses[fold], color=colors[fold],
                     linewidth=1.5, label=f"Fold {fold+1}")
        axes[1].plot(all_val_losses[fold],   color=colors[fold],
                     linewidth=1.5, label=f"Fold {fold+1}")
    for ax, title in zip(axes, ["Train Loss", "Validation Loss"]):
        ax.set_xlabel("Epoch"); ax.set_ylabel("Total loss")
        ax.set_title(title); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = cfg.FIGURES_DIR / "training" / "01_total_loss.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nSaved {out}")

    # ── Plot: per-component loss breakdown ────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    keys   = ["cls", "pos", "d", "feas"]
    titles = [
        "CrossEntropy (zone classification)",
        "GaussianNLL (position regression)",
        "MSE (curvilinear distance d)",
        "Feasibility (corridor violation)",
    ]
    for ax_idx, (key, title) in enumerate(zip(keys, titles)):
        ax = axes[ax_idx // 2, ax_idx % 2]
        for fold in range(cfg.N_FOLDS):
            t_vals = [d[key] for d in all_train_detail[fold]]
            v_vals = [d[key] for d in all_val_detail[fold]]
            ax.plot(t_vals, color=colors[fold], linewidth=1,   alpha=0.5)
            ax.plot(v_vals, color=colors[fold], linewidth=1.5, linestyle="--")
        ax.plot([], [], "k-",  linewidth=1,   label="Train")
        ax.plot([], [], "k--", linewidth=1.5, label="Val")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.set_title(title); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = cfg.FIGURES_DIR / "training" / "02_loss_breakdown.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()


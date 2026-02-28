"""Data exploration and maze geometry visualisation.

Run this script before training to inspect the dataset, maze geometry,
spike statistics, and zone distribution.

Usage (from repo root)
----------------------
    python scripts/visualize_data.py

Outputs (figures/data/)
-----------------------
01_maze_geometry.png    — U-maze skeleton overlay + zone classification
02_curvilinear_dist.png — distribution of d along the U + zone thresholds
03_spike_statistics.png — spike counts per shank + sequence-length distribution
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import src.config as cfg
from src.dataset import load_data
from src.geometry import (
    D_LEFT_END,
    D_RIGHT_START,
    compute_curvilinear_distance,
    compute_geometry,
    print_geometry_stats,
)

FIG_DIR = cfg.FIGURES_DIR / "data"
FIG_DIR.mkdir(parents=True, exist_ok=True)

ZONE_COLORS = ["#4C72B0", "#DD8452", "#55A868"]


def main():
    df_moving, nGroups, nChannelsPerGroup = load_data()
    positions = np.array([[x[0], x[1]] for x in df_moving["pos"]], dtype=np.float32)
    curvilinear_d, zone_labels = compute_geometry(positions)
    print_geometry_stats(positions, curvilinear_d, zone_labels)

    # ── Figure 01: Maze geometry ──────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))

    # 1a. Trajectory + skeleton
    axes[0].scatter(positions[:, 0], positions[:, 1],
                    c="lightgray", s=1, alpha=0.3, label="Mouse positions")
    for x1, y1, x2, y2 in cfg.SKELETON_SEGMENTS:
        axes[0].plot([x1, x2], [y1, y2], "r-", linewidth=3)
    axes[0].set_xlabel("X"); axes[0].set_ylabel("Y")
    axes[0].set_title("U-maze skeleton overlaid on positions")
    axes[0].set_aspect("equal")
    axes[0].legend(markerscale=8)

    # 1b. Curvilinear distance d
    sc = axes[1].scatter(positions[:, 0], positions[:, 1],
                         c=curvilinear_d, s=1, alpha=0.5, cmap="viridis")
    plt.colorbar(sc, ax=axes[1], label="d (normalised curvilinear distance)")
    axes[1].set_xlabel("X"); axes[1].set_ylabel("Y")
    axes[1].set_title("Curvilinear distance d along the U")
    axes[1].set_aspect("equal")

    # 1c. 3-zone classification
    for z, (name, color) in enumerate(zip(cfg.ZONE_NAMES, ZONE_COLORS)):
        mask = zone_labels == z
        axes[2].scatter(positions[mask, 0], positions[mask, 1],
                        c=color, s=1, alpha=0.3, label=f"{name} ({mask.mean():.1%})")
    axes[2].set_xlabel("X"); axes[2].set_ylabel("Y")
    axes[2].set_title("Zone classification (3 zones)")
    axes[2].legend(markerscale=10); axes[2].set_aspect("equal")

    plt.tight_layout()
    out = FIG_DIR / "01_maze_geometry.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")

    # ── Figure 02: Curvilinear distance distribution ─────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    axes[0].hist(curvilinear_d, bins=100, alpha=0.7,
                 edgecolor="black", linewidth=0.3, color="steelblue")
    axes[0].axvline(D_LEFT_END, color=ZONE_COLORS[0], linestyle="--",
                    linewidth=2, label=f"Left | Top  (d={D_LEFT_END:.3f})")
    axes[0].axvline(D_RIGHT_START, color=ZONE_COLORS[2], linestyle="--",
                    linewidth=2, label=f"Top | Right (d={D_RIGHT_START:.3f})")

    # Shade zone backgrounds
    axes[0].axvspan(0,             D_LEFT_END,    alpha=0.08, color=ZONE_COLORS[0])
    axes[0].axvspan(D_LEFT_END,    D_RIGHT_START, alpha=0.08, color=ZONE_COLORS[1])
    axes[0].axvspan(D_RIGHT_START, 1,             alpha=0.08, color=ZONE_COLORS[2])

    axes[0].set_xlabel("d (curvilinear distance)"); axes[0].set_ylabel("Count")
    axes[0].set_title("Distribution of d along the U-maze")
    axes[0].legend()

    # Pie chart of zone distribution
    zone_counts = [(zone_labels == z).sum() for z in range(cfg.N_ZONES)]
    wedges, _, autotexts = axes[1].pie(
        zone_counts, labels=cfg.ZONE_NAMES, colors=ZONE_COLORS,
        autopct="%1.1f%%", startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    for t in autotexts:
        t.set_fontsize(12)
    axes[1].set_title(f"Zone distribution  (N={len(zone_labels):,})")

    plt.tight_layout()
    out = FIG_DIR / "02_curvilinear_dist.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")

    # ── Figure 03: Spike statistics ───────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))

    # 3a. Mean spikes per window per shank
    shank_spike_counts = []
    for g in range(nGroups):
        col = f"group{g}"
        if col in df_moving.columns:
            counts = [len(row["groups"]) for _, row in df_moving.head(2000).iterrows()]
            shank_spike_counts.append(counts)
        else:
            shank_spike_counts.append([0])

    # Spike counts from 'groups' array (total spikes per window)
    total_per_window = np.array([len(row["groups"]) for _, row in df_moving.iterrows()])

    # Per-shank spike contribution
    shank_totals = []
    for g in range(nGroups):
        n = np.array([
            np.sum(np.array(row["groups"]) == g)
            for _, row in df_moving.iterrows()
        ])
        shank_totals.append(n.mean())

    shank_labels = [f"Shank {g}\n({nChannelsPerGroup[g]} ch)" for g in range(nGroups)]
    bars = axes[0].bar(shank_labels, shank_totals,
                       color=plt.cm.tab10(np.linspace(0, 0.4, nGroups)), alpha=0.8)
    for bar, val in zip(bars, shank_totals):
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.1, f"{val:.1f}",
                     ha="center", va="bottom", fontsize=11)
    axes[0].set_ylabel("Mean spikes per window")
    axes[0].set_title("Average spike contribution per shank")
    axes[0].grid(True, axis="y", alpha=0.3)

    # 3b. Sequence length distribution
    axes[1].hist(total_per_window, bins=50, alpha=0.7,
                 color="steelblue", edgecolor="white")
    axes[1].axvline(total_per_window.mean(), color="red", linestyle="--",
                    linewidth=2, label=f"Mean = {total_per_window.mean():.1f}")
    axes[1].axvline(cfg.MAX_SEQ_LEN, color="orange", linestyle="--",
                    linewidth=2, label=f"Max seq len = {cfg.MAX_SEQ_LEN}")
    axes[1].set_xlabel("Total spikes per window")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Sequence length distribution")
    axes[1].legend()

    # 3c. Spike count per shank stacked histogram
    shank_data = []
    for g in range(nGroups):
        n = np.array([
            int(np.sum(np.array(row["groups"]) == g))
            for _, row in df_moving.iterrows()
        ])
        shank_data.append(n)

    colors = plt.cm.tab10(np.linspace(0, 0.4, nGroups))
    bottom = np.zeros(len(df_moving))
    bins = np.arange(0, cfg.MAX_SEQ_LEN + 10, 5)
    for g, (data, color) in enumerate(zip(shank_data, colors)):
        axes[2].hist(data, bins=bins, alpha=0.6, color=color,
                     label=f"Shank {g}", histtype="step", linewidth=2)
    axes[2].set_xlabel("Spikes per window")
    axes[2].set_ylabel("Count")
    axes[2].set_title("Per-shank spike count distribution")
    axes[2].legend()

    plt.tight_layout()
    out = FIG_DIR / "03_spike_statistics.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")

    print(f"\nAll data figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()

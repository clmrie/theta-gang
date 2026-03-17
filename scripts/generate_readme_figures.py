"""Generate comparison figures for the README from the results table."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figures" / "readme"
OUT.mkdir(parents=True, exist_ok=True)

# ── Data from the results table ──────────────────────────────────────────────
methods = ["Naive\n(mean)", "Ridge\nRegression", "k-NN\n+ PCA", "Spike\nTransformer"]
colors = ["#9E9E9E", "#9E9E9E", "#9E9E9E", "#E53935"]
edge_colors = ["#757575", "#757575", "#757575", "#B71C1C"]

mse   = [0.094, 0.085, 0.100, 0.030]
euc   = [0.422, 0.391, 0.403, 0.22]
r2_x  = [-0.00, 0.05, -0.16, 0.65]
r2_y  = [-0.02, 0.11, -0.02, 0.72]
zone  = [39.7, 49.3, 43.9, 82.0]
corr  = [0.0, 17.7, 49.2, 95.0]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 13,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

def bar_label(ax, bars, fmt="{:.3f}"):
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h,
                fmt.format(h), ha="center", va="bottom",
                fontsize=11, fontweight="bold")


# ── Figure 1: MSE & Euclidean Error (lower is better) ───────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

for ax, vals, title, ylabel in [
    (axes[0], mse, "Mean Squared Error", "MSE"),
    (axes[1], euc, "Euclidean Error", "Error"),
]:
    bars = ax.bar(methods, vals, color=colors, edgecolor=edge_colors, linewidth=1.2, width=0.6)
    bar_label(ax, bars)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, max(vals) * 1.25)
    # Add "lower is better" annotation
    ax.annotate("↓ lower is better", xy=(0.98, 0.95), xycoords="axes fraction",
                ha="right", va="top", fontsize=9, color="#666", fontstyle="italic")

fig.suptitle("Spike Transformer vs. Baselines — Error Metrics", fontsize=15, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(OUT / "comparison_error.png", dpi=180, bbox_inches="tight", facecolor="white")
print(f"Saved {OUT / 'comparison_error.png'}")
plt.close(fig)


# ── Figure 2: R² (higher is better) ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

for ax, vals, title in [
    (axes[0], r2_x, "R² — X coordinate"),
    (axes[1], r2_y, "R² — Y coordinate"),
]:
    bars = ax.bar(methods, vals, color=colors, edgecolor=edge_colors, linewidth=1.2, width=0.6)
    bar_label(ax, bars, fmt="{:.2f}")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
    ax.set_ylabel("R²")
    ax.set_ylim(min(min(vals) * 1.3, -0.25), max(vals) * 1.3)
    ax.axhline(0, color="#ccc", linewidth=0.8, zorder=0)
    ax.annotate("↑ higher is better", xy=(0.98, 0.95), xycoords="axes fraction",
                ha="right", va="top", fontsize=9, color="#666", fontstyle="italic")

fig.suptitle("Spike Transformer vs. Baselines — R² Score", fontsize=15, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(OUT / "comparison_r2.png", dpi=180, bbox_inches="tight", facecolor="white")
print(f"Saved {OUT / 'comparison_r2.png'}")
plt.close(fig)


# ── Figure 3: Zone Accuracy & Corridor Adherence (higher is better) ─────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

for ax, vals, title in [
    (axes[0], zone, "Zone Classification Accuracy"),
    (axes[1], corr, "Corridor Adherence"),
]:
    bars = ax.bar(methods, vals, color=colors, edgecolor=edge_colors, linewidth=1.2, width=0.6)
    bar_label(ax, bars, fmt="{:.1f}%")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
    ax.set_ylabel("Percentage (%)")
    ax.set_ylim(0, 110)
    ax.annotate("↑ higher is better", xy=(0.98, 0.95), xycoords="axes fraction",
                ha="right", va="top", fontsize=9, color="#666", fontstyle="italic")

fig.suptitle("Spike Transformer vs. Baselines — Classification & Geometry", fontsize=15, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(OUT / "comparison_classification.png", dpi=180, bbox_inches="tight", facecolor="white")
print(f"Saved {OUT / 'comparison_classification.png'}")
plt.close(fig)

print("Done!")

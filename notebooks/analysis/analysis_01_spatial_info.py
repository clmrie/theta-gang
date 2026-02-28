"""Analysis 01 — Spatial Information & Rate Maps par shank.

Génère 5 figures dans artifacts/analysis_01/ :
  01_overview.png          — occupancy, trajectoire, stats spikes
  02_rate_maps.png         — rate maps (brut + log) par shank
  03_spatial_info.png      — SI scores (bits/spike + bits/sample)
  04_arm_analysis.png      — spike counts par bras (Left/Top/Right)
  05_correlations.png      — corrélations features × position

Usage (depuis la racine du repo) :
    python notebooks/analysis_01_spatial_info.py
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from matplotlib.colors import LogNorm
from scipy.stats import spearmanr

# ── Config ────────────────────────────────────────────────────────────────────

MOUSE       = "M1199_PAG"
DATA_DIR    = "data"
OUT_DIR     = "artifacts/analysis_01"
GRID        = 40          # grille pour rate maps
TRAIN_FRAC  = 0.9
MIN_OCC     = 3           # visites min pour rate map valide

N_GROUPS    = 4
N_CH        = [6, 4, 6, 4]
SHANK_NAMES = [f"Shank {g}  ({N_CH[g]}ch)" for g in range(N_GROUPS)]

SKEL = np.array([
    [0.15, 0.0,  0.15, 0.85],
    [0.15, 0.85, 0.85, 0.85],
    [0.85, 0.85, 0.85, 0.0 ],
])
ARM_NAMES  = ["Left", "Top", "Right"]
ARM_COLORS = ["#4C72B0", "#DD8452", "#55A868"]
SHANK_PAL  = plt.cm.tab10(np.linspace(0, 0.4, N_GROUPS))

os.makedirs(OUT_DIR, exist_ok=True)

# ── Helpers ───────────────────────────────────────────────────────────────────

def arm_of(xy):
    """Vectorised arm assignment.  Returns (N,) int 0/1/2."""
    best_d = np.full(len(xy), np.inf)
    best_a = np.zeros(len(xy), dtype=int)
    for a, (x1, y1, x2, y2) in enumerate(SKEL):
        dx, dy = x2 - x1, y2 - y1
        sq = dx*dx + dy*dy
        t  = np.clip(((xy[:, 0]-x1)*dx + (xy[:, 1]-y1)*dy) / sq, 0, 1)
        d  = np.hypot(xy[:, 0]-x1-t*dx, xy[:, 1]-y1-t*dy)
        better = d < best_d
        best_d[better] = d[better]
        best_a[better] = a
    return best_a

def draw_skel(ax, lw=1.5, color="red", alpha=0.75):
    for x1, y1, x2, y2 in SKEL:
        ax.plot([x1, x2], [y1, y2], color=color, lw=lw, alpha=alpha)

def cell_xy(i, j, g=GRID):
    return (i + 0.5) / g, (j + 0.5) / g

def compute_rate_map(positions, spike_counts_g, occ_min=MIN_OCC, g=GRID):
    """Occupancy-normalised rate map.  Returns (g, g) float, NaN where undef."""
    occ = np.zeros((g, g), dtype=np.float32)
    ssum= np.zeros((g, g), dtype=np.float32)
    ci  = np.clip((positions[:, 0] * g).astype(int), 0, g-1)
    cj  = np.clip((positions[:, 1] * g).astype(int), 0, g-1)
    np.add.at(occ,  (ci, cj), 1)
    np.add.at(ssum, (ci, cj), spike_counts_g)
    rate = np.full((g, g), np.nan)
    valid = occ >= occ_min
    rate[valid] = ssum[valid] / occ[valid]
    return rate, occ

def spatial_info(rate, occ):
    """Skaggs et al. SI in bits/sample and bits/spike."""
    valid  = np.isfinite(rate) & (occ > 0)
    p      = occ[valid] / occ[valid].sum()        # occupancy fraction
    lam    = rate[valid]
    lam_bar= (p * lam).sum()
    if lam_bar < 1e-12:
        return 0.0, 0.0
    ratio  = lam / lam_bar
    pos    = ratio > 0
    si_sample = (p[pos] * ratio[pos] * np.log2(ratio[pos])).sum()
    si_spike  = si_sample / lam_bar
    return float(si_sample), float(si_spike)


# ── Load data ─────────────────────────────────────────────────────────────────

print("Loading parquet …")
parquet_path = f"{DATA_DIR}/{MOUSE}_stride4_win108_test.parquet"
pf    = pq.ParquetFile(parquet_path)
table = pf.read(columns=["pos", "speedMask", "groups", "time"])

# Positions
pos_flat = table.column("pos").combine_chunks().flatten()
all_xy   = np.array(pos_flat, dtype=np.float32).reshape(-1, 4)[:, :2]

# Speed mask
sm_flat = table.column("speedMask").combine_chunks().flatten()
moving  = np.array(sm_flat, dtype=bool)

# Time (for speed estimation)
time_flat = table.column("time").combine_chunks().flatten()
all_time  = np.array(time_flat, dtype=np.float64)

# Spike counts per shank — vectorised via bincount on flattened offset array
groups_arr = table.column("groups").combine_chunks()
offsets    = np.array(groups_arr.offsets, dtype=np.int64)
values     = np.array(groups_arr.values,  dtype=np.int64)
N          = len(all_xy)

combined      = np.repeat(np.arange(N, dtype=np.int64), np.diff(offsets)) * N_GROUPS + values
counts_flat   = np.bincount(combined, minlength=N * N_GROUPS)
spike_counts  = counts_flat.reshape(N, N_GROUPS).astype(np.int32)  # (N, 4)

# Train split (moving only, first 90%)
moving_idx = np.where(moving)[0]
split      = int(len(moving_idx) * TRAIN_FRAC)
train_idx  = moving_idx[:split]
train_xy   = all_xy[train_idx]
train_sc   = spike_counts[train_idx]       # (M, 4)
train_t    = all_time[train_idx]

# Arm labels
train_arm  = arm_of(train_xy)

print(f"  Total: {N}  |  Moving: {moving.sum()}  |  Train: {len(train_idx)}")
print(f"  Mean spike counts per shank: {train_sc.mean(axis=0).round(2)}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Overview
# ══════════════════════════════════════════════════════════════════════════════
print("\nFig 1 — Overview …")

# Speed estimate
dt    = np.diff(train_t)
dpos  = np.linalg.norm(np.diff(train_xy, axis=0), axis=1)
speed = dpos / np.where(dt > 0, dt, 1e-6)
speed = np.clip(speed, 0, np.percentile(speed, 99))

fig = plt.figure(figsize=(18, 10))
fig.suptitle(f"{MOUSE} — Overview", fontsize=15, fontweight="bold")
gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.38)

# 1a — Occupancy
ax = fig.add_subplot(gs[0, 0])
occ_map = np.zeros((GRID, GRID))
ci = np.clip((train_xy[:, 0]*GRID).astype(int), 0, GRID-1)
cj = np.clip((train_xy[:, 1]*GRID).astype(int), 0, GRID-1)
np.add.at(occ_map, (ci, cj), 1)
im = ax.imshow(occ_map.T, origin="lower", extent=[0,1,0,1],
               cmap="YlOrRd", aspect="equal")
plt.colorbar(im, ax=ax, fraction=0.046, label="visits")
draw_skel(ax)
ax.set_title("Occupancy map (train)"); ax.set_xlabel("x"); ax.set_ylabel("y")

# 1b — Trajectory (first 1000 samples)
ax = fig.add_subplot(gs[0, 1])
n_show = min(1000, len(train_xy))
sc_plot = ax.scatter(train_xy[:n_show, 0], train_xy[:n_show, 1],
                     c=np.arange(n_show), cmap="plasma", s=4, alpha=0.7)
plt.colorbar(sc_plot, ax=ax, fraction=0.046, label="time step")
draw_skel(ax)
ax.set_title(f"Trajectory (first {n_show} steps)")
ax.set_xlabel("x"); ax.set_ylabel("y")

# 1c — Speed distribution
ax = fig.add_subplot(gs[0, 2])
ax.hist(speed, bins=60, color="#4C72B0", edgecolor="white", linewidth=0.3)
ax.axvline(np.median(speed), color="red", lw=1.5, linestyle="--",
           label=f"median={np.median(speed):.3f}")
ax.set_xlabel("Speed (a.u. / s)"); ax.set_ylabel("Count")
ax.set_title("Speed distribution"); ax.legend(fontsize=8)

# 1d — Arm proportion
ax = fig.add_subplot(gs[0, 3])
arm_counts = [(train_arm == a).sum() for a in range(3)]
bars = ax.bar(ARM_NAMES, arm_counts, color=ARM_COLORS, edgecolor="white")
for bar, cnt in zip(bars, arm_counts):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+20,
            f"{cnt/len(train_arm):.1%}", ha="center", va="bottom", fontsize=9)
ax.set_ylabel("Samples"); ax.set_title("Arm distribution (train)")

# 1e — Total spike count per sample histogram (all shanks)
ax = fig.add_subplot(gs[1, 0])
total_sc = train_sc.sum(axis=1)
ax.hist(total_sc, bins=60, color="#55A868", edgecolor="white", linewidth=0.3)
ax.axvline(np.median(total_sc), color="red", lw=1.5, linestyle="--",
           label=f"median={int(np.median(total_sc))}")
ax.set_xlabel("Total spikes / window"); ax.set_ylabel("Count")
ax.set_title("Total spike count distribution"); ax.legend(fontsize=8)

# 1f — Per-shank mean spike count
ax = fig.add_subplot(gs[1, 1])
means = train_sc.mean(axis=0)
stds  = train_sc.std(axis=0)
bars  = ax.bar(range(N_GROUPS), means, yerr=stds, capsize=4,
               color=SHANK_PAL, edgecolor="white")
ax.set_xticks(range(N_GROUPS)); ax.set_xticklabels([f"S{g}" for g in range(N_GROUPS)])
ax.set_ylabel("Mean spikes / window"); ax.set_title("Per-shank spike counts (mean ± std)")
for b, m in zip(bars, means):
    ax.text(b.get_x()+b.get_width()/2, m+0.05, f"{m:.1f}",
            ha="center", va="bottom", fontsize=8)

# 1g — Spike count CDF per shank
ax = fig.add_subplot(gs[1, 2])
for g in range(N_GROUPS):
    sorted_sc = np.sort(train_sc[:, g])
    cdf = np.arange(1, len(sorted_sc)+1) / len(sorted_sc)
    ax.plot(sorted_sc, cdf, label=f"S{g}", color=SHANK_PAL[g], lw=1.5)
ax.set_xlabel("Spike count / window"); ax.set_ylabel("CDF")
ax.set_title("Spike count CDF per shank"); ax.legend(fontsize=8)
ax.set_xlim(left=0)

# 1h — Spike count per arm, stacked bar
ax = fig.add_subplot(gs[1, 3])
width = 0.6
bottom = np.zeros(N_GROUPS)
for a in range(3):
    m = train_sc[train_arm == a].mean(axis=0)
    ax.bar(range(N_GROUPS), m, width, bottom=bottom,
           label=ARM_NAMES[a], color=ARM_COLORS[a], alpha=0.85)
    bottom += m
ax.set_xticks(range(N_GROUPS)); ax.set_xticklabels([f"S{g}" for g in range(N_GROUPS)])
ax.set_ylabel("Mean spikes / window"); ax.set_title("Per-arm spike contribution per shank")
ax.legend(fontsize=8, loc="upper right")

plt.savefig(f"{OUT_DIR}/01_overview.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {OUT_DIR}/01_overview.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Rate maps
# ══════════════════════════════════════════════════════════════════════════════
print("Fig 2 — Rate maps …")

rate_maps = []
occ_maps  = []
for g in range(N_GROUPS):
    rm, occ = compute_rate_map(train_xy, train_sc[:, g])
    rate_maps.append(rm)
    occ_maps.append(occ)

fig, axes = plt.subplots(N_GROUPS, 3, figsize=(14, 4*N_GROUPS))
fig.suptitle(f"{MOUSE} — Rate maps par shank  (grille {GRID}×{GRID})", fontsize=14, fontweight="bold")

for g in range(N_GROUPS):
    rm  = rate_maps[g]
    occ = occ_maps[g]

    # Raw rate map
    ax = axes[g, 0]
    vmax = np.nanpercentile(rm, 98)
    im = ax.imshow(rm.T, origin="lower", extent=[0,1,0,1],
                   cmap="hot", aspect="equal", vmin=0, vmax=vmax)
    plt.colorbar(im, ax=ax, fraction=0.046, label="spikes/window")
    draw_skel(ax)
    ax.set_title(f"Shank {g} ({N_CH[g]} ch) — Rate map")
    ax.set_xlabel("x"); ax.set_ylabel("y")

    # Log rate map
    ax = axes[g, 1]
    log_rm = np.where(np.isfinite(rm) & (rm > 0), np.log1p(rm), np.nan)
    im = ax.imshow(log_rm.T, origin="lower", extent=[0,1,0,1],
                   cmap="viridis", aspect="equal")
    plt.colorbar(im, ax=ax, fraction=0.046, label="log(1+rate)")
    draw_skel(ax)

    # Peak: where is max rate?
    flat_idx = np.nanargmax(rm)
    pi, pj   = np.unravel_index(flat_idx, rm.shape)
    px, py   = cell_xy(pi, pj)
    ax.scatter([px], [py], c="red", s=80, marker="*", zorder=5, label=f"peak ({rm[pi,pj]:.1f})")
    ax.legend(fontsize=7, loc="lower right")
    ax.set_title(f"Shank {g} — Log rate map")
    ax.set_xlabel("x"); ax.set_ylabel("y")

    # Rate histogram
    ax = axes[g, 2]
    valid_rates = rm[np.isfinite(rm) & (rm > 0)]
    ax.hist(valid_rates, bins=40, color=SHANK_PAL[g], edgecolor="white", linewidth=0.3)
    ax.axvline(np.nanmean(rm[np.isfinite(rm)]), color="red", lw=1.5,
               linestyle="--", label=f"mean={np.nanmean(rm[np.isfinite(rm)]):.2f}")
    ax.axvline(np.nanmax(rm), color="orange", lw=1.5,
               linestyle=":", label=f"max={np.nanmax(rm):.2f}")
    ax.set_xlabel("Firing rate (spikes/window)"); ax.set_ylabel("# cells")
    ax.set_title(f"Shank {g} — Rate distribution")
    ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/02_rate_maps.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {OUT_DIR}/02_rate_maps.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Spatial Information
# ══════════════════════════════════════════════════════════════════════════════
print("Fig 3 — Spatial Information …")

_, occ_ref = compute_rate_map(train_xy, train_sc[:, 0])

si_sample_list, si_spike_list = [], []
for g in range(N_GROUPS):
    sism, sispk = spatial_info(rate_maps[g], occ_maps[g])
    si_sample_list.append(sism)
    si_spike_list.append(sispk)
    print(f"  Shank {g}: SI = {sism:.4f} bits/sample  |  {sispk:.4f} bits/spike")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(f"{MOUSE} — Spatial Information (Skaggs et al.)", fontsize=14, fontweight="bold")

# 3a — SI bits/spike bar
ax = axes[0, 0]
bars = ax.bar(range(N_GROUPS), si_spike_list, color=SHANK_PAL, edgecolor="white", width=0.6)
ax.set_xticks(range(N_GROUPS))
ax.set_xticklabels([f"S{g}" for g in range(N_GROUPS)])
ax.set_ylabel("SI (bits/spike)"); ax.set_title("Spatial Information — bits/spike")
best_g = int(np.argmax(si_spike_list))
bars[best_g].set_edgecolor("red"); bars[best_g].set_linewidth(2.5)
for b, v in zip(bars, si_spike_list):
    ax.text(b.get_x()+b.get_width()/2, v+0.0002, f"{v:.4f}",
            ha="center", va="bottom", fontsize=9)
ax.set_title("Spatial Information — bits/spike\n(red = best shank)")

# 3b — SI bits/sample bar
ax = axes[0, 1]
bars = ax.bar(range(N_GROUPS), si_sample_list, color=SHANK_PAL, edgecolor="white", width=0.6)
ax.set_xticks(range(N_GROUPS))
ax.set_xticklabels([f"S{g}" for g in range(N_GROUPS)])
ax.set_ylabel("SI (bits/sample)")
best_g2 = int(np.argmax(si_sample_list))
bars[best_g2].set_edgecolor("red"); bars[best_g2].set_linewidth(2.5)
for b, v in zip(bars, si_sample_list):
    ax.text(b.get_x()+b.get_width()/2, v+0.0002, f"{v:.4f}",
            ha="center", va="bottom", fontsize=9)
ax.set_title("Spatial Information — bits/sample\n(red = best shank)")

# 3c — Sorted per-cell SI contribution (cumulative)
ax = axes[1, 0]
valid_mask = np.isfinite(occ_ref) & (occ_ref >= MIN_OCC)
p_all      = occ_ref[valid_mask] / occ_ref[valid_mask].sum()
for g in range(N_GROUPS):
    rm   = rate_maps[g]
    lam  = rm[valid_mask]
    lam_bar = (p_all * lam).sum()
    if lam_bar < 1e-12:
        continue
    ratio = lam / lam_bar
    cell_si = np.where(ratio > 0, p_all * ratio * np.log2(np.where(ratio > 0, ratio, 1)), 0)
    sorted_si = np.sort(cell_si)[::-1]
    cumsi = np.cumsum(sorted_si)
    ax.plot(np.arange(1, len(cumsi)+1) / len(cumsi), cumsi / cumsi[-1],
            label=f"S{g} (total={cumsi[-1]:.3f})", color=SHANK_PAL[g], lw=2)
ax.set_xlabel("Fraction of cells (sorted by SI contribution, best first)")
ax.set_ylabel("Cumulative SI (normalised)")
ax.set_title("Cumulative SI — which cells carry the most info?")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# 3d — Rate map of best shank with SI-weighted color
ax = axes[1, 1]
rm   = rate_maps[best_g]
lam  = rm.copy()
valid= np.isfinite(lam)
lam_bar = np.nanmean(lam)
ratio_map = np.where(valid & (lam > 0), lam / lam_bar, np.nan)
# SI contribution per cell
si_map = np.full_like(lam, np.nan)
valid2 = valid & (ratio_map > 0)
p_map  = occ_maps[best_g] / occ_maps[best_g][occ_maps[best_g] >= MIN_OCC].sum()
si_map[valid2] = p_map[valid2] * ratio_map[valid2] * np.log2(ratio_map[valid2])

im = ax.imshow(si_map.T, origin="lower", extent=[0,1,0,1],
               cmap="magma", aspect="equal")
plt.colorbar(im, ax=ax, fraction=0.046, label="SI contribution (bits/sample)")
draw_skel(ax)
ax.set_title(f"Shank {best_g} — Spatial Info map\n(best shank, bits/sample per cell)")
ax.set_xlabel("x"); ax.set_ylabel("y")

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/03_spatial_info.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {OUT_DIR}/03_spatial_info.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4 — Arm analysis
# ══════════════════════════════════════════════════════════════════════════════
print("Fig 4 — Arm analysis …")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle(f"{MOUSE} — Analyse par bras du U-maze", fontsize=14, fontweight="bold")

# 4a — Violin plots per shank per arm
for g in range(N_GROUPS):
    ax = axes[0, g] if g < 3 else axes[1, 0]
    data = [train_sc[train_arm == a, g] for a in range(3)]
    parts = ax.violinplot(data, positions=range(3), showmedians=True,
                          showextrema=False)
    for pc, col in zip(parts['bodies'], ARM_COLORS):
        pc.set_facecolor(col); pc.set_alpha(0.75)
    parts['cmedians'].set_color('black')
    ax.set_xticks(range(3)); ax.set_xticklabels(ARM_NAMES)
    ax.set_ylabel("Spikes / window"); ax.set_title(f"Shank {g} ({N_CH[g]}ch)")
    means = [d.mean() for d in data]
    for i, m in enumerate(means):
        ax.text(i, ax.get_ylim()[1]*0.95, f"μ={m:.1f}",
                ha="center", fontsize=8, color="black")

# 4b (axes[0,3] doesn't exist, use axes[1,1]) — ratio Left/Right per shank
ax = axes[1, 1]
sc_left  = train_sc[train_arm == 0].mean(axis=0)
sc_top   = train_sc[train_arm == 1].mean(axis=0)
sc_right = train_sc[train_arm == 2].mean(axis=0)
x = np.arange(N_GROUPS)
w = 0.25
ax.bar(x - w, sc_left,  w, label="Left",  color=ARM_COLORS[0], edgecolor="white")
ax.bar(x,     sc_top,   w, label="Top",   color=ARM_COLORS[1], edgecolor="white")
ax.bar(x + w, sc_right, w, label="Right", color=ARM_COLORS[2], edgecolor="white")
ax.set_xticks(x); ax.set_xticklabels([f"S{g}" for g in range(N_GROUPS)])
ax.set_ylabel("Mean spikes / window"); ax.set_title("Mean spikes per arm per shank")
ax.legend(fontsize=8)

# 4c — Spike ratio map on maze (shank 0)
ax = axes[1, 2]
sc_map, occ_map2 = compute_rate_map(train_xy, train_sc[:, best_g])
valid_g = np.isfinite(sc_map) & (occ_map2 >= MIN_OCC)
# Classify each cell's arm and color by relative rate
arm_img = np.full(sc_map.shape, np.nan)
for i in range(GRID):
    for j in range(GRID):
        if valid_g[i, j]:
            cx, cy = cell_xy(i, j)
            arm_img[i, j] = sc_map[i, j]
im = ax.imshow(arm_img.T, origin="lower", extent=[0,1,0,1],
               cmap="hot", aspect="equal")
plt.colorbar(im, ax=ax, fraction=0.046, label="spikes/window")
draw_skel(ax)
ax.set_title(f"Shank {best_g} — Rate map avec bras")
patches = [mpatches.Patch(color=c, label=n) for c, n in zip(ARM_COLORS, ARM_NAMES)]
ax.legend(handles=patches, fontsize=7, loc="lower right")
ax.set_xlabel("x"); ax.set_ylabel("y")

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/04_arm_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {OUT_DIR}/04_arm_analysis.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 5 — Corrélations features × position
# ══════════════════════════════════════════════════════════════════════════════
print("Fig 5 — Correlations …")

fig, axes = plt.subplots(3, N_GROUPS, figsize=(5*N_GROUPS, 12))
fig.suptitle(f"{MOUSE} — Corrélations spike count × position", fontsize=14, fontweight="bold")

corr_table = np.zeros((N_GROUPS, 3))  # [rho_x, rho_y, rho_arm]

for g in range(N_GROUPS):
    sc_g = train_sc[:, g].astype(float)
    x    = train_xy[:, 0]
    y    = train_xy[:, 1]
    arm  = train_arm.astype(float)

    rho_x,   _ = spearmanr(sc_g, x)
    rho_y,   _ = spearmanr(sc_g, y)
    rho_arm, _ = spearmanr(sc_g, arm)
    corr_table[g] = [rho_x, rho_y, rho_arm]

    # Scatter: sc vs x, colored by arm
    ax = axes[0, g]
    for a in range(3):
        mask = train_arm == a
        ax.scatter(x[mask], sc_g[mask], c=ARM_COLORS[a], s=1, alpha=0.3,
                   label=ARM_NAMES[a])
    ax.set_xlabel("x position"); ax.set_ylabel("Spikes / window")
    ax.set_title(f"S{g} vs x  (ρ={rho_x:.3f})")
    if g == 0: ax.legend(fontsize=7, markerscale=5)

    # Scatter: sc vs y
    ax = axes[1, g]
    for a in range(3):
        mask = train_arm == a
        ax.scatter(y[mask], sc_g[mask], c=ARM_COLORS[a], s=1, alpha=0.3)
    ax.set_xlabel("y position"); ax.set_ylabel("Spikes / window")
    ax.set_title(f"S{g} vs y  (ρ={rho_y:.3f})")

    # 2D hexbin
    ax = axes[2, g]
    hb = ax.hexbin(x, y, C=sc_g, gridsize=25, cmap="hot",
                   reduce_C_function=np.mean, extent=[0,1,0,1])
    plt.colorbar(hb, ax=ax, fraction=0.046, label="mean spikes")
    draw_skel(ax)
    ax.set_title(f"S{g} — 2D spike density map")
    ax.set_xlabel("x"); ax.set_ylabel("y")

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/05_correlations.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {OUT_DIR}/05_correlations.png")


# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("RÉSUMÉ")
print("="*55)
print(f"{'Shank':<10} {'SI (bits/spk)':<18} {'SI (bits/spl)':<18} {'ρ(x)':<10} {'ρ(y)':<10}")
print("-"*55)
for g in range(N_GROUPS):
    print(f"S{g} ({N_CH[g]}ch)  {si_spike_list[g]:<18.4f} {si_sample_list[g]:<18.4f} "
          f"{corr_table[g,0]:<10.3f} {corr_table[g,1]:<10.3f}")
print(f"\n→ Shank le plus informatif (bits/spike) : Shank {best_g}")
print(f"→ Figures sauvegardées dans : {OUT_DIR}/")


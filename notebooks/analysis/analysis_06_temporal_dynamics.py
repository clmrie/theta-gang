"""Analysis 06 — Dynamique temporelle & modulation par la vitesse.

Questions :
  1. La vitesse module-t-elle le taux de décharge ?  (speed ↔ firing rate)
  2. Quelle est l'autocorrélation temporelle des features neurales ?
     → calibre le bruit de processus du filtre de Kalman
  3. L'erreur de décodage k-NN est-elle corrélée à la vitesse ?
     → si oui, speed peut être une feature auxiliaire de confiance
  4. Y a-t-il une sélectivité directionnelle ?  (gauche vs droite vs haut/bas)
  5. Comment évolue la position au fil du temps ?  (trajectoire + persistance)

Figures → artifacts/analysis_06/
  01_speed_vs_firing.png       : Speed distribution + corrélation speed / firing rate
  02_temporal_autocorr.png     : Autocorrélation temporelle des features PCA
  03_error_vs_speed.png        : Erreur k-NN vs vitesse (et direction)
  04_direction_tuning.png      : Sélectivité directionnelle par shank
  05_trajectory_heatmap.png    : Trajectoires + vitesse sur le maze + persistance

Usage : python notebooks/analysis_06_temporal_dynamics.py
"""
import os, sys
from pathlib import Path
ROOT = Path(__file__).parent.parent; sys.path.insert(0, str(ROOT)); os.chdir(ROOT)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pyarrow.parquet as pq
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_predict, KFold
from scipy.stats import spearmanr, pearsonr
from scipy.ndimage import gaussian_filter

OUT  = "artifacts/analysis_06"
N_CH = [6, 4, 6, 4]
GRID = 50
SEED = 42
K_NN = 20
np.random.seed(SEED)
os.makedirs(OUT, exist_ok=True)

SKEL = np.array([[.15,.0,.15,.85],[.15,.85,.85,.85],[.85,.85,.85,.0]])
def skel_overlay(ax, c="white", lw=1.8):
    for x1,y1,x2,y2 in SKEL: ax.plot([x1,x2],[y1,y2],c=c,lw=lw,alpha=0.85)

def get_arm(xy):
    x, y = xy[:, 0], xy[:, 1]
    arm = np.full(len(x), 1, dtype=np.int8)
    arm[(x < 0.35) & (y > 0.45)] = 0
    arm[(x > 0.65) & (y > 0.45)] = 2
    return arm

# ─── Step 1: Load all needed columns ─────────────────────────────────────────
print("Loading data …")
pf = pq.ParquetFile("data/M1199_PAG_stride4_win108_test.parquet")
base = pf.read(columns=["pos", "speedMask", "groups", "time_behavior"])

xy_all    = np.array(base.column("pos").combine_chunks().flatten(),
                     dtype=np.float32).reshape(-1, 4)[:, :2]
N_ALL     = len(xy_all)

# Speed mask (per row boolean)
sm_col    = base.column("speedMask")
speed_mask = np.array([bool(sm_col[i][0]) for i in range(N_ALL)], dtype=bool)

# Time (behavior clock)
tb_col = base.column("time_behavior")
time_all = np.array([float(tb_col[i][0]) for i in range(N_ALL)], dtype=np.float64)

# Spike count per shank
grp_col  = base.column("groups").combine_chunks()
grp_off  = np.array(grp_col.offsets, dtype=np.int64)
grp_vals = np.array(grp_col.values,  dtype=np.int64)
sc_all   = np.bincount(
    np.repeat(np.arange(N_ALL, dtype=np.int64), np.diff(grp_off)) * 4 + grp_vals,
    minlength=N_ALL * 4
).reshape(N_ALL, 4).astype(np.int32)
del base, grp_col, grp_off, grp_vals

# Compute instantaneous speed from position differences
# Speed = euclidean distance between consecutive positions (all samples, not just moving)
dxy       = np.diff(xy_all, axis=0)
speed_raw = np.concatenate([[0.0], np.sqrt((dxy**2).sum(axis=1))])  # pixels / stride
speed_all = speed_raw.astype(np.float32)

# Moving samples
moving_idx = np.where(speed_mask)[0]
xy_mov     = xy_all[moving_idx]
sc_mov     = sc_all[moving_idx]        # (N_mov, 4)
sp_mov     = speed_all[moving_idx]     # speed per window
t_mov      = time_all[moving_idx]      # time per window
N_MOV      = len(moving_idx)

# Movement direction angle
dxy_mov = np.diff(xy_mov, axis=0, prepend=xy_mov[:1])
angle_mov = np.arctan2(dxy_mov[:, 1], dxy_mov[:, 0])  # radians, -π to π

print(f"  Moving: {N_MOV}  speed range: {sp_mov.min():.4f}–{sp_mov.max():.4f}  "
      f"mean: {sp_mov.mean():.4f}")

# ─── Step 2: Load waveforms + PCA features ───────────────────────────────────
print("Loading waveforms …")
mean_wf_all = {}
for g, n_ch in enumerate(N_CH):
    col = pf.read(columns=[f"group{g}"]).column(f"group{g}")
    dim = n_ch * 32
    mw  = np.zeros((N_MOV, dim), dtype=np.float32)
    for ii, idx in enumerate(moving_idx):
        flat = np.array(col[idx], dtype=np.float32)
        n_sp = len(flat) // dim
        if n_sp > 0:
            mw[ii] = flat.reshape(n_sp, dim).mean(axis=0)
    del col
    mean_wf_all[g] = mw
    print(f"  S{g}: done")

valid = np.ones(N_MOV, dtype=bool)
for g in range(4): valid &= (mean_wf_all[g].sum(axis=1) != 0)
N_V = valid.sum()
xy_v  = xy_mov[valid];  sc_v  = sc_mov[valid]
sp_v  = sp_mov[valid];  t_v   = t_mov[valid];  ang_v = angle_mov[valid]
print(f"  Valid: {N_V}")

print("PCA per shank …")
pca_feats = []
for g in range(4):
    X    = mean_wf_all[g][valid]
    X_sc = StandardScaler().fit_transform(X)
    Z    = PCA(n_components=20, random_state=SEED).fit_transform(X_sc)
    pca_feats.append(Z)
F_all = np.hstack(pca_feats)   # (N_V, 80)
del mean_wf_all

# ─── Figure 1: Speed vs firing rate ──────────────────────────────────────────
print("\nFig 1: Speed vs firing rate …")
fig, axes = plt.subplots(1, 5, figsize=(22, 4))
fig.suptitle("Vitesse vs taux de décharge — M1199_PAG (windows mouvants)",
             fontsize=13, fontweight="bold")

# Speed distribution
ax = axes[0]
ax.hist(sp_v, bins=60, color="#4292c6", edgecolor="white", density=True)
ax.axvline(np.median(sp_v), c="red", lw=2, ls="--", label=f"Médiane={np.median(sp_v):.4f}")
ax.set_xlabel("Vitesse (unités/stride)"); ax.set_ylabel("Densité")
ax.set_title("Distribution de vitesse"); ax.legend(fontsize=8)

# Per-shank scatter + spearman
colors_sh = ["#e41a1c", "#ff7f00", "#4daf4a", "#377eb8"]
for g in range(4):
    ax = axes[g + 1]
    # Bin speed into 20 bins, plot mean firing rate per bin
    bins   = np.percentile(sp_v, np.linspace(0, 100, 21))
    bins   = np.unique(bins)
    idx_b  = np.digitize(sp_v, bins) - 1
    idx_b  = np.clip(idx_b, 0, len(bins) - 2)
    bin_sp = []; bin_fr = []
    for b in range(len(bins) - 1):
        m = idx_b == b
        if m.sum() >= 5:
            bin_sp.append(sp_v[m].mean())
            bin_fr.append(sc_v[m, g].mean())
    bin_sp, bin_fr = np.array(bin_sp), np.array(bin_fr)

    rho, pval = spearmanr(sp_v, sc_v[:, g])
    ax.scatter(sp_v, sc_v[:, g], s=2, alpha=0.15, color=colors_sh[g], rasterized=True)
    ax.plot(bin_sp, bin_fr, "ko-", lw=2, ms=5, label=f"ρ={rho:.3f} (p={pval:.1e})")
    ax.set_xlabel("Vitesse"); ax.set_ylabel("Spike count")
    ax.set_title(f"S{g} ({N_CH[g]}ch)"); ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

plt.tight_layout()
plt.savefig(f"{OUT}/01_speed_vs_firing.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved {OUT}/01_speed_vs_firing.png")

# ─── Figure 2: Temporal autocorrelation of PCA features ──────────────────────
print("Fig 2: Temporal autocorrelation …")
# Sort by time to get temporal order
t_order = np.argsort(t_v)
F_sorted = F_all[t_order]      # (N_V, 80) in temporal order
xy_sorted = xy_v[t_order]

# Compute autocorrelation of PC1 per shank and of position
MAX_LAG = 50  # windows

def autocorr(x, max_lag):
    x = x - x.mean()
    norm = np.dot(x, x)
    return np.array([np.dot(x[lag:], x[:len(x)-lag]) / norm
                     for lag in range(max_lag + 1)])

# Autocorr of PC1 per shank
pc1_by_shank = [F_all[t_order, g * 20] for g in range(4)]  # PC1 of each shank
ac_x   = autocorr(xy_sorted[:, 0], MAX_LAG)
ac_y   = autocorr(xy_sorted[:, 1], MAX_LAG)
ac_pc  = [autocorr(pc1_by_shank[g], MAX_LAG) for g in range(4)]

# Stride in time (approx stride = 4 samples, 1 sample ≈ 1/1250 s = 0.8ms, stride=3.2ms)
STRIDE_MS = 4 * 0.8  # ~3.2ms per window

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Autocorrélation temporelle — features neurales et position",
             fontsize=13, fontweight="bold")

lags_ms = np.arange(MAX_LAG + 1) * STRIDE_MS

ax = axes[0]
ax.plot(lags_ms, ac_x, "b-o", ms=4, lw=1.5, label="x")
ax.plot(lags_ms, ac_y, "r-o", ms=4, lw=1.5, label="y")
ax.axhline(0, color="gray", lw=0.8)
ax.set_xlabel("Lag (ms)"); ax.set_ylabel("Autocorrélation")
ax.set_title("Position (x, y)"); ax.legend(fontsize=9)
ax.grid(True, alpha=0.25)
# Find lag where autocorr drops to 0.5
for arr, name, col in [(ac_x, "x", "b"), (ac_y, "y", "r")]:
    half = np.argmax(arr < 0.5)
    if half > 0:
        ax.axvline(lags_ms[half], color=col, lw=1, ls="--", alpha=0.6,
                   label=f"τ½({name})={lags_ms[half]:.0f}ms")
ax.legend(fontsize=7)

ax = axes[1]
for g in range(4):
    ax.plot(lags_ms, ac_pc[g], color=colors_sh[g], lw=1.5, label=f"S{g} PC1")
ax.axhline(0, color="gray", lw=0.8)
ax.set_xlabel("Lag (ms)"); ax.set_ylabel("Autocorrélation")
ax.set_title("PC1 des waveforms par shank"); ax.legend(fontsize=9)
ax.grid(True, alpha=0.25)

# Mean autocorr over all 80 PCA dims
ac_mean_all = np.mean([autocorr(F_sorted[:, d], MAX_LAG) for d in range(80)], axis=0)
ax = axes[2]
ax.plot(lags_ms, ac_mean_all, "k-o", ms=4, lw=2, label="Mean all PCs")
ax.fill_between(lags_ms,
                np.percentile([autocorr(F_sorted[:, d], MAX_LAG) for d in range(80)], 10, axis=0),
                np.percentile([autocorr(F_sorted[:, d], MAX_LAG) for d in range(80)], 90, axis=0),
                alpha=0.3, color="gray", label="10–90th percentile")
ax.axhline(0, color="gray", lw=0.8)
half_all = np.argmax(ac_mean_all < 0.5)
if half_all > 0:
    ax.axvline(lags_ms[half_all], color="red", lw=1.5, ls="--",
               label=f"τ½={lags_ms[half_all]:.0f}ms")
ax.set_xlabel("Lag (ms)"); ax.set_ylabel("Autocorrélation")
ax.set_title("Autocorr. moyenne — 80 PCA dims"); ax.legend(fontsize=9)
ax.grid(True, alpha=0.25)

plt.tight_layout()
plt.savefig(f"{OUT}/02_temporal_autocorr.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved {OUT}/02_temporal_autocorr.png")

# ─── K-NN predictions (needed for fig 3) ─────────────────────────────────────
print("K-NN predictions …")
F_sc = StandardScaler().fit_transform(F_all)
cv   = KFold(n_splits=5, shuffle=True, random_state=SEED)
knn  = KNeighborsRegressor(n_neighbors=K_NN, algorithm="ball_tree", n_jobs=-1)
xy_pred  = cross_val_predict(knn, F_sc, xy_v, cv=cv)
eucl_err = np.sqrt(((xy_v - xy_pred) ** 2).sum(axis=1))

# ─── Figure 3: Error vs speed ─────────────────────────────────────────────────
print("Fig 3: Error vs speed …")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Erreur k-NN vs vitesse et direction de mouvement",
             fontsize=13, fontweight="bold")

ax = axes[0]
# Bin speed, plot mean error per bin
n_bins = 25
pct    = np.percentile(sp_v, np.linspace(5, 95, n_bins + 1))
pct    = np.unique(pct)
idx_b  = np.digitize(sp_v, pct) - 1
idx_b  = np.clip(idx_b, 0, len(pct) - 2)
bs_sp, bs_err, bs_std = [], [], []
for b in range(len(pct) - 1):
    m = idx_b == b
    if m.sum() >= 5:
        bs_sp.append(sp_v[m].mean())
        bs_err.append(eucl_err[m].mean())
        bs_std.append(eucl_err[m].std())
bs_sp, bs_err, bs_std = map(np.array, [bs_sp, bs_err, bs_std])
rho_s, _ = spearmanr(sp_v, eucl_err)
ax.scatter(sp_v, eucl_err, s=2, alpha=0.1, color="steelblue", rasterized=True)
ax.plot(bs_sp, bs_err, "ko-", lw=2, ms=5)
ax.fill_between(bs_sp, bs_err - bs_std, bs_err + bs_std, alpha=0.3, color="gray")
ax.set_xlabel("Vitesse"); ax.set_ylabel("Erreur k-NN")
ax.set_title(f"Erreur vs vitesse (ρ={rho_s:.3f})")
ax.grid(True, alpha=0.25)

ax = axes[1]
# Polar plot of error per direction
n_dir = 12
ang_bins = np.linspace(-np.pi, np.pi, n_dir + 1)
dir_idx  = np.digitize(ang_v, ang_bins) - 1
dir_idx  = np.clip(dir_idx, 0, n_dir - 1)
dir_err  = np.array([eucl_err[dir_idx == d].mean() if (dir_idx == d).sum() > 5 else np.nan
                     for d in range(n_dir)])
dir_cnt  = np.array([(dir_idx == d).sum() for d in range(n_dir)])
ang_centers = (ang_bins[:-1] + ang_bins[1:]) / 2

ax2 = fig.add_subplot(1, 3, 2, projection="polar")
theta_plot = np.append(ang_centers, ang_centers[0])
err_plot   = np.append(dir_err, dir_err[0])
cnt_plot   = np.append(dir_cnt, dir_cnt[0])
ax2.plot(theta_plot, err_plot, "b-o", lw=2, ms=5, label="Erreur k-NN")
ax2.fill(theta_plot, err_plot, alpha=0.3, color="blue")
ax2.set_title("Erreur par direction\n(0=→ droite, π/2=↑ haut)", pad=15)
ax2.legend(fontsize=7, loc="upper right")

axes[1].set_visible(False)  # hide the original axes[1], replaced by polar

ax = axes[2]
# 2D scatter: x=speed, y=error, colored by arm
arm_v = get_arm(xy_v)
arm_names = ["Left", "Top", "Right"]
arm_colors = ["#e41a1c", "#4daf4a", "#377eb8"]
for a, (aname, acol) in enumerate(zip(arm_names, arm_colors)):
    m = arm_v == a
    rho_a, _ = spearmanr(sp_v[m], eucl_err[m])
    ax.scatter(sp_v[m], eucl_err[m], s=2, alpha=0.2, color=acol,
               label=f"{aname} ρ={rho_a:.3f}", rasterized=True)
ax.set_xlabel("Vitesse"); ax.set_ylabel("Erreur k-NN")
ax.set_title("Erreur vs vitesse par bras")
ax.legend(fontsize=8, markerscale=4); ax.grid(True, alpha=0.25)

plt.tight_layout()
plt.savefig(f"{OUT}/03_error_vs_speed.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved {OUT}/03_error_vs_speed.png")

# ─── Figure 4: Direction tuning ───────────────────────────────────────────────
print("Fig 4: Direction tuning …")
# For each shank: mean spike count per direction (polar)
n_dir = 16
ang_bins16 = np.linspace(-np.pi, np.pi, n_dir + 1)
ang_centers16 = (ang_bins16[:-1] + ang_bins16[1:]) / 2
dir16 = np.clip(np.digitize(ang_v, ang_bins16) - 1, 0, n_dir - 1)

fig = plt.figure(figsize=(18, 8))
fig.suptitle("Sélectivité directionnelle par shank",
             fontsize=13, fontweight="bold")

for g in range(4):
    ax_p = fig.add_subplot(2, 4, g + 1, projection="polar")
    mean_fr = np.array([sc_v[dir16 == d, g].mean() if (dir16 == d).sum() > 5 else 0.0
                        for d in range(n_dir)])
    theta_p = np.append(ang_centers16, ang_centers16[0])
    fr_p    = np.append(mean_fr, mean_fr[0])
    ax_p.plot(theta_p, fr_p, "o-", lw=2, color=colors_sh[g])
    ax_p.fill(theta_p, fr_p, alpha=0.3, color=colors_sh[g])
    ax_p.set_title(f"S{g} — firing rate / direction\n(mean={mean_fr.mean():.1f})", pad=10)
    # Rayleigh-like: direction bias = std / mean of direction counts
    bias = mean_fr.std() / (mean_fr.mean() + 1e-6)
    ax_p.set_title(f"S{g} ({N_CH[g]}ch) — biais={bias:.3f}", pad=10)

    # Bottom: speed-conditioned direction tuning (fast vs slow)
    sp_med = np.median(sp_v)
    for row, (label, mask_sp) in enumerate(
        [("Lent", sp_v < sp_med), ("Rapide", sp_v >= sp_med)]
    ):
        ax_p2 = fig.add_subplot(2, 4, 4 + g + 1) if row == 0 else None
    ax_b = fig.add_subplot(2, 4, 4 + g + 1)
    mean_fr_slow = np.array([sc_v[dir16 == d, g][sp_v[dir16 == d] < sp_med].mean()
                              if ((dir16 == d) & (sp_v < sp_med)).sum() > 3 else 0.0
                              for d in range(n_dir)])
    mean_fr_fast = np.array([sc_v[dir16 == d, g][sp_v[dir16 == d] >= sp_med].mean()
                              if ((dir16 == d) & (sp_v >= sp_med)).sum() > 3 else 0.0
                              for d in range(n_dir)])
    x16 = np.arange(n_dir)
    ax_b.bar(x16 - 0.2, mean_fr_slow, 0.4, label="Lent", color="steelblue", alpha=0.8)
    ax_b.bar(x16 + 0.2, mean_fr_fast, 0.4, label="Rapide", color="tomato", alpha=0.8)
    ax_b.set_xticks(x16[::4]); ax_b.set_xticklabels([f"{a:.0f}°" for a in
                                                        np.degrees(ang_centers16[::4]).astype(int)])
    ax_b.set_title(f"S{g} — lent vs rapide"); ax_b.set_xlabel("Direction")
    ax_b.set_ylabel("spk/win"); ax_b.legend(fontsize=7); ax_b.grid(True, alpha=0.2, axis="y")

plt.tight_layout()
plt.savefig(f"{OUT}/04_direction_tuning.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved {OUT}/04_direction_tuning.png")

# ─── Figure 5: Speed heatmap + trajectory persistence ─────────────────────────
print("Fig 5: Trajectory heatmap …")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Vitesse et persistance spatiale — M1199_PAG",
             fontsize=13, fontweight="bold")

# Mean speed per cell
sp_sum = np.zeros((GRID, GRID)); sp_cnt = np.zeros((GRID, GRID))
ci = np.clip((xy_v[:, 0] * GRID).astype(int), 0, GRID - 1)
cj = np.clip((xy_v[:, 1] * GRID).astype(int), 0, GRID - 1)
np.add.at(sp_sum, (ci, cj), sp_v)
np.add.at(sp_cnt, (ci, cj), 1)
sp_map = np.where(sp_cnt >= 3, sp_sum / sp_cnt, np.nan)
sp_sm  = gaussian_filter(np.where(np.isfinite(sp_map), sp_map, 0), sigma=1.5)
sp_sm  = np.where(np.isfinite(sp_map), sp_sm, np.nan)

ax = axes[0]
im = ax.imshow(sp_sm.T, origin="lower", extent=[0,1,0,1],
               cmap="plasma", aspect="equal")
plt.colorbar(im, ax=ax, fraction=0.046, label="Vitesse moy.")
skel_overlay(ax, c="cyan")
ax.set_title("Vitesse moyenne par cellule"); ax.set_xlabel("x"); ax.set_ylabel("y")

# Dwell time (total frames per cell — proportional to occupancy)
occ = sp_cnt.copy()
occ_sm = gaussian_filter(np.where(occ > 0, occ, 0), sigma=1.0)
occ_sm = np.where(occ > 0, occ_sm, np.nan)
ax = axes[1]
im = ax.imshow(occ_sm.T, origin="lower", extent=[0,1,0,1],
               cmap="YlOrRd", aspect="equal")
plt.colorbar(im, ax=ax, fraction=0.046, label="# visites")
skel_overlay(ax, c="black")
ax.set_title("Persistance (# visites)"); ax.set_xlabel("x"); ax.set_ylabel("y")

# Speed vs error heatmap (average kNN error per speed quartile × position arm)
ax = axes[2]
sp_q   = np.percentile(sp_v, [25, 50, 75])
arm_v2 = get_arm(xy_v)
n_arms, n_q = 3, 4
err_mat = np.zeros((n_arms, n_q))
for a in range(n_arms):
    for q, (lo, hi) in enumerate(zip([0] + list(sp_q), list(sp_q) + [np.inf])):
        m = (arm_v2 == a) & (sp_v >= lo) & (sp_v < hi)
        err_mat[a, q] = eucl_err[m].mean() if m.sum() > 5 else np.nan

im = ax.imshow(err_mat, cmap="RdYlGn_r", aspect="auto", vmin=0.2, vmax=0.45)
plt.colorbar(im, ax=ax, fraction=0.046, label="kNN error")
ax.set_xticks(range(4)); ax.set_xticklabels(["Q1\n(slow)", "Q2", "Q3", "Q4\n(fast)"])
ax.set_yticks(range(3)); ax.set_yticklabels(arm_names)
ax.set_title("Erreur k-NN : bras × quartile de vitesse")
for a in range(n_arms):
    for q in range(n_q):
        if np.isfinite(err_mat[a, q]):
            ax.text(q, a, f"{err_mat[a, q]:.3f}", ha="center", va="center",
                    fontsize=9, color="black")

plt.tight_layout()
plt.savefig(f"{OUT}/05_trajectory_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved {OUT}/05_trajectory_heatmap.png")

# ─── Print summary ────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY — Analysis 06")
print("=" * 60)
print("Speed–firing rate correlations (Spearman ρ):")
for g in range(4):
    rho_g, pval_g = spearmanr(sp_v, sc_v[:, g])
    print(f"  S{g}: ρ={rho_g:.4f}  p={pval_g:.2e}")

half_x = np.argmax(ac_x < 0.5)
half_y = np.argmax(ac_y < 0.5)
half_n = np.argmax(ac_mean_all < 0.5)
print(f"\nAutocorrelation τ½ (lag where r drops below 0.5):")
print(f"  x-position  : lag={half_x}  ({half_x * STRIDE_MS:.0f} ms)")
print(f"  y-position  : lag={half_y}  ({half_y * STRIDE_MS:.0f} ms)")
print(f"  neural (PCA): lag={half_n}  ({half_n * STRIDE_MS:.0f} ms)")

print(f"\nError vs speed (global ρ={spearmanr(sp_v, eucl_err)[0]:.4f}):")
for a, aname in enumerate(arm_names):
    m = arm_v == a
    rho_a, _ = spearmanr(sp_v[m], eucl_err[m])
    print(f"  {aname}: ρ={rho_a:.4f}")

print(f"\nkNN error by speed quartile × arm:")
for a in range(n_arms):
    print(f"  {arm_names[a]:6s}: {' '.join(f'{err_mat[a,q]:.3f}' for q in range(n_q))}")
print("\nDone.")

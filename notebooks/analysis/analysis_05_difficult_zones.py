"""Analysis 05 — Zones difficiles à décoder (variance positionnelle).

Question centrale : quelles zones du labyrinthe sont les plus dures à décoder ?
Métriques par cellule de grille :
  1. Variance neurale  — même position, état neural variable → incertitude encodeur
  2. Erreur k-NN       — voisins neuraux proches mais positions éloignées → ambiguïté
  3. CV spike count    — variabilité du taux de décharge par position
  4. Sous-couverture   — peu de visites → moins de données d'entraînement
  5. Score composite   — combinaison normalisée des métriques ci-dessus

Figures → artifacts/analysis_05/
  01_occupancy.png         : Nombre de visites par cellule (couverture)
  02_neural_variance.png   : Variance des features neurales par position
  03_knn_error.png         : Erreur positionnelle k-NN par position
  04_spike_cv.png          : CV spike count par position (4 shanks)
  05_difficulty_score.png  : Score composite + worst zones overlay

Usage : python notebooks/analysis_05_difficult_zones.py
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
from scipy.ndimage import gaussian_filter

OUT   = "artifacts/analysis_05"
N_CH  = [6, 4, 6, 4]
GRID  = 50
SEED  = 42
MIN_OCC = 5        # minimum de visites pour inclure une cellule
K_NN    = 20       # voisins pour k-NN decoding
np.random.seed(SEED)
os.makedirs(OUT, exist_ok=True)

SKEL = np.array([[.15,.0,.15,.85],[.15,.85,.85,.85],[.85,.85,.85,.0]])
def skel_overlay(ax, c="white", lw=1.8):
    for x1,y1,x2,y2 in SKEL: ax.plot([x1,x2],[y1,y2], c=c, lw=lw, alpha=0.85)

def get_arm(xy):
    x, y = xy[:, 0], xy[:, 1]
    arm = np.full(len(x), 1, dtype=np.int8)
    arm[(x < 0.35) & (y > 0.45)] = 0
    arm[(x > 0.65) & (y > 0.45)] = 2
    return arm

def cell_ij(xy, grid=GRID):
    ci = np.clip((xy[:, 0] * grid).astype(int), 0, grid - 1)
    cj = np.clip((xy[:, 1] * grid).astype(int), 0, grid - 1)
    return ci, cj

# ─── Step 1: Load pos + speedMask + spike counts ─────────────────────────────
print("Loading pos + speedMask …")
pf = pq.ParquetFile("data/M1199_PAG_stride4_win108_test.parquet")
base = pf.read(columns=["pos", "speedMask", "groups"])
xy_all = np.array(base.column("pos").combine_chunks().flatten(),
                  dtype=np.float32).reshape(-1, 4)[:, :2]
N_ALL  = len(xy_all)
speed_mask = np.array([bool(base.column("speedMask")[i][0]) for i in range(N_ALL)], dtype=bool)

# Spike count per shank from 'groups' column
grp_col  = base.column("groups").combine_chunks()
grp_off  = np.array(grp_col.offsets, dtype=np.int64)
grp_vals = np.array(grp_col.values,  dtype=np.int64)
sc_all = np.bincount(
    np.repeat(np.arange(N_ALL, dtype=np.int64), np.diff(grp_off)) * 4 + grp_vals,
    minlength=N_ALL * 4
).reshape(N_ALL, 4).astype(np.int32)
del base, grp_col, grp_off, grp_vals

moving_idx = np.where(speed_mask)[0]
xy_mov     = xy_all[moving_idx]
sc_mov     = sc_all[moving_idx]           # (N_mov, 4)
N_MOV      = len(moving_idx)
print(f"  Moving: {N_MOV}")

# ─── Step 2: Load waveforms → mean per window → all-shank PCA ────────────────
print("Loading waveforms + computing mean waveforms …")
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

# Valid mask (all shanks active)
valid = np.ones(N_MOV, dtype=bool)
for g in range(4):
    valid &= (mean_wf_all[g].sum(axis=1) != 0)
print(f"  Valid: {valid.sum()} / {N_MOV}")

xy_v  = xy_mov[valid]
sc_v  = sc_mov[valid]       # (N_valid, 4) spike counts
ci, cj = cell_ij(xy_v)

# PCA 20 per shank → concatenate → all-shank features
print("PCA per shank …")
pca_feats = []
for g in range(4):
    X    = mean_wf_all[g][valid]
    X_sc = StandardScaler().fit_transform(X)
    Z    = PCA(n_components=20, random_state=SEED).fit_transform(X_sc)
    pca_feats.append(Z)
F_all = np.hstack(pca_feats)   # (N_valid, 80)
del mean_wf_all

N_V = valid.sum()

# ─── Figure 1: Occupancy ──────────────────────────────────────────────────────
print("\nFig 1: Occupancy …")
occ = np.zeros((GRID, GRID))
np.add.at(occ, (ci, cj), 1)
occ_valid = np.where(occ >= MIN_OCC, occ, np.nan)

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(occ_valid.T, origin="lower", extent=[0,1,0,1],
               cmap="YlOrRd", aspect="equal")
plt.colorbar(im, ax=ax, fraction=0.046, label="# visites")
skel_overlay(ax, c="black")
ax.set_title("Couverture spatiale (# visites par cellule 50×50)\nM1199_PAG — windows mouvants", fontweight="bold")
ax.set_xlabel("x"); ax.set_ylabel("y")
# Mark low-coverage zones
low_occ = (occ < 10) & (occ >= 1)
ci_low, cj_low = np.where(low_occ)
ax.scatter((ci_low + 0.5) / GRID, (cj_low + 0.5) / GRID,
           s=4, c="blue", alpha=0.4, label="<10 visites")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(f"{OUT}/01_occupancy.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved {OUT}/01_occupancy.png")

# ─── Compute neural variance per grid cell ────────────────────────────────────
print("Computing neural variance per cell …")
# For each cell: mean of within-cell feature variance (trace of covariance)
var_map = np.full((GRID, GRID), np.nan)
cnt_map = np.zeros((GRID, GRID), dtype=int)
np.add.at(cnt_map, (ci, cj), 1)

# Per-cell variance of all-shank PCA features
cell_idx = ci * GRID + cj                  # flat cell index per sample
n_cells  = GRID * GRID
for flat_c in range(n_cells):
    mask = cell_idx == flat_c
    if mask.sum() >= MIN_OCC:
        F_c = F_all[mask]
        # Trace of sample covariance (= sum of per-feature variances)
        var_map[flat_c // GRID, flat_c % GRID] = F_c.var(axis=0).mean()

# ─── Figure 2: Neural variance ────────────────────────────────────────────────
print("Fig 2: Neural variance map …")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Variance des features neurales par position",
             fontsize=13, fontweight="bold")

ax = axes[0]
vm = var_map.copy()
im = ax.imshow(vm.T, origin="lower", extent=[0,1,0,1],
               cmap="hot", aspect="equal")
plt.colorbar(im, ax=ax, fraction=0.046, label="Var moy. features PCA")
skel_overlay(ax, c="cyan")
ax.set_title("Variance neurale brute"); ax.set_xlabel("x"); ax.set_ylabel("y")

ax = axes[1]
# Smooth + highlight top 20% most variable cells
vm_sm = gaussian_filter(np.where(np.isfinite(vm), vm, 0), sigma=1.5)
vm_sm = np.where(np.isfinite(vm), vm_sm, np.nan)
thr   = np.nanpercentile(vm_sm, 80)
im    = ax.imshow(vm_sm.T, origin="lower", extent=[0,1,0,1],
                  cmap="hot", aspect="equal")
plt.colorbar(im, ax=ax, fraction=0.046, label="Var (lissée)")
skel_overlay(ax, c="cyan")
# Highlight worst zones (top 20%)
hard_ci, hard_cj = np.where(np.isfinite(vm_sm) & (vm_sm >= thr))
ax.scatter((hard_ci + 0.5) / GRID, (hard_cj + 0.5) / GRID,
           s=6, c="lime", alpha=0.5, label="Top 20% variance")
ax.legend(fontsize=8)
ax.set_title("Variance lissée — zones instables (vert)"); ax.set_xlabel("x"); ax.set_ylabel("y")

plt.tight_layout()
plt.savefig(f"{OUT}/02_neural_variance.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved {OUT}/02_neural_variance.png")

# ─── K-NN decoding error ─────────────────────────────────────────────────────
print("K-NN decoding (5-fold CV) …")
knn = KNeighborsRegressor(n_neighbors=K_NN, algorithm="ball_tree", n_jobs=-1)
cv  = KFold(n_splits=5, shuffle=True, random_state=SEED)

# Scale features before kNN
F_sc = StandardScaler().fit_transform(F_all)
xy_pred = cross_val_predict(knn, F_sc, xy_v, cv=cv)    # (N_valid, 2)

eucl_err = np.sqrt(((xy_v - xy_pred) ** 2).sum(axis=1))
print(f"  Global k-NN error: mean={eucl_err.mean():.4f}  median={np.median(eucl_err):.4f}"
      f"  p90={np.percentile(eucl_err, 90):.4f}")

# Map to grid
err_map = np.full((GRID, GRID), np.nan)
err_sum = np.zeros((GRID, GRID))
err_cnt = np.zeros((GRID, GRID))
np.add.at(err_sum, (ci, cj), eucl_err)
np.add.at(err_cnt, (ci, cj), 1)
v = err_cnt >= MIN_OCC
err_map[v] = err_sum[v] / err_cnt[v]

# ─── Figure 3: KNN error map ─────────────────────────────────────────────────
print("Fig 3: K-NN error map …")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f"Erreur k-NN (k={K_NN}) par position — all-shank features",
             fontsize=13, fontweight="bold")

ax = axes[0]
em = err_map.copy()
em_sm = gaussian_filter(np.where(np.isfinite(em), em, 0), sigma=1.5)
em_sm = np.where(np.isfinite(em), em_sm, np.nan)
im = ax.imshow(em_sm.T, origin="lower", extent=[0,1,0,1],
               cmap="RdYlGn_r", aspect="equal",
               vmin=np.nanpercentile(em_sm, 2),
               vmax=np.nanpercentile(em_sm, 98))
plt.colorbar(im, ax=ax, fraction=0.046, label="Erreur eucl. moy.")
skel_overlay(ax, c="black")
ax.set_title("Erreur k-NN par cellule"); ax.set_xlabel("x"); ax.set_ylabel("y")

ax = axes[1]
# Scatter plot: true vs predicted colored by error
sc = ax.scatter(xy_v[:, 0], xy_v[:, 1], c=eucl_err, cmap="RdYlGn_r",
                s=3, alpha=0.5, rasterized=True,
                vmin=np.percentile(eucl_err, 2), vmax=np.percentile(eucl_err, 98))
plt.colorbar(sc, ax=ax, fraction=0.046, label="Erreur eucl.")
skel_overlay(ax, c="black")
ax.set_title(f"Erreur par sample (médiane={np.median(eucl_err):.3f})")
ax.set_xlabel("x"); ax.set_ylabel("y")

plt.tight_layout()
plt.savefig(f"{OUT}/03_knn_error.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved {OUT}/03_knn_error.png")

# ─── Spike count CV per position ─────────────────────────────────────────────
print("Fig 4: Spike count CV map …")
fig, axes = plt.subplots(1, 4, figsize=(20, 4))
fig.suptitle("Coefficient de variation du spike count par position (std/mean)",
             fontsize=13, fontweight="bold")

cv_maps = []
for g in range(4):
    sc_sum  = np.zeros((GRID, GRID))
    sc_sum2 = np.zeros((GRID, GRID))
    sc_cnt  = np.zeros((GRID, GRID))
    np.add.at(sc_sum,  (ci, cj), sc_v[:, g])
    np.add.at(sc_sum2, (ci, cj), sc_v[:, g] ** 2)
    np.add.at(sc_cnt,  (ci, cj), 1)
    v2 = sc_cnt >= MIN_OCC
    mean_c = np.where(v2, sc_sum / sc_cnt, np.nan)
    var_c  = np.where(v2, sc_sum2 / sc_cnt - mean_c ** 2, np.nan)
    cv_c   = np.where(v2 & (mean_c > 0), np.sqrt(np.maximum(var_c, 0)) / mean_c, np.nan)
    cv_maps.append(cv_c)

    ax = axes[g]
    im = ax.imshow(cv_c.T, origin="lower", extent=[0,1,0,1], cmap="plasma",
                   aspect="equal", vmin=0, vmax=np.nanpercentile(cv_c, 95))
    plt.colorbar(im, ax=ax, fraction=0.046, label="CV")
    skel_overlay(ax)
    ax.set_title(f"S{g} ({N_CH[g]}ch)")
    ax.set_xlabel("x"); ax.set_ylabel("y")

plt.tight_layout()
plt.savefig(f"{OUT}/04_spike_cv.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved {OUT}/04_spike_cv.png")

# ─── Figure 5: Composite difficulty score ────────────────────────────────────
print("Fig 5: Composite difficulty score …")

def norm01(m):
    lo, hi = np.nanpercentile(m, 2), np.nanpercentile(m, 98)
    return np.clip((m - lo) / (hi - lo + 1e-9), 0, 1)

# Neural variance (normalized)
s_var = norm01(vm_sm)
# KNN error (normalized)
s_knn = norm01(em_sm)
# Spike count CV (mean over shanks, normalized)
cv_mean = np.nanmean(np.stack([cv_maps[g] for g in range(4)], axis=0), axis=0)
cv_sm   = gaussian_filter(np.where(np.isfinite(cv_mean), cv_mean, 0), sigma=1.5)
cv_sm   = np.where(np.isfinite(cv_mean), cv_sm, np.nan)
s_cv    = norm01(cv_sm)
# Low coverage penalty: inverse of log(count+1)
log_occ = np.log1p(occ)
log_occ_n = np.where(occ >= MIN_OCC, norm01(log_occ), np.nan)
s_cov   = 1 - np.where(np.isfinite(log_occ_n), log_occ_n, np.nan)

# Composite: weighted mean (error has highest weight)
weights  = [0.35, 0.35, 0.15, 0.15]   # knn, var, cv, cov
maps_n   = [s_knn, s_var, s_cv, s_cov]
valid_px = np.isfinite(s_knn) & np.isfinite(s_var)
composite = np.full((GRID, GRID), np.nan)
composite[valid_px] = sum(w * m[valid_px] for w, m in zip(weights, maps_n)
                          if np.isfinite(m[valid_px]).all())
# Fallback for pixels where some maps are NaN
for i in range(GRID):
    for j in range(GRID):
        vals = [m[i,j] for m in maps_n if np.isfinite(m[i,j])]
        ws   = [w for w, m in zip(weights, maps_n) if np.isfinite(m[i,j])]
        if vals and occ[i,j] >= MIN_OCC:
            composite[i,j] = sum(v * w for v, w in zip(vals, ws)) / sum(ws)

fig, axes = plt.subplots(1, 3, figsize=(19, 5))
fig.suptitle("Score composite de difficulté de décodage",
             fontsize=13, fontweight="bold")

# Full heatmap
ax = axes[0]
im = ax.imshow(composite.T, origin="lower", extent=[0,1,0,1],
               cmap="RdYlGn_r", aspect="equal", vmin=0, vmax=1)
plt.colorbar(im, ax=ax, fraction=0.046, label="Difficulté [0–1]")
skel_overlay(ax, c="black")
ax.set_title("Score composite\n(kNN×0.35 + var×0.35 + CV×0.15 + cov×0.15)")
ax.set_xlabel("x"); ax.set_ylabel("y")

# Top 25% hardest zones overlaid on occupancy
ax = axes[1]
ax.imshow(occ_valid.T, origin="lower", extent=[0,1,0,1], cmap="Greys", aspect="equal", alpha=0.5)
hard_thr = np.nanpercentile(composite, 75)
hi2, hj2 = np.where(np.isfinite(composite) & (composite >= hard_thr))
ax.scatter((hi2 + 0.5) / GRID, (hj2 + 0.5) / GRID, s=8, c="red",
           alpha=0.6, label="Top 25% difficile", zorder=3)
easy_thr = np.nanpercentile(composite, 25)
ei2, ej2 = np.where(np.isfinite(composite) & (composite <= easy_thr))
ax.scatter((ei2 + 0.5) / GRID, (ej2 + 0.5) / GRID, s=8, c="lime",
           alpha=0.6, label="Top 25% facile", zorder=3)
skel_overlay(ax, c="black")
ax.legend(fontsize=8); ax.set_title("Zones difficiles (rouge) vs faciles (vert)")
ax.set_xlabel("x"); ax.set_ylabel("y")

# Per-arm distribution of difficulty
ax = axes[2]
arm_v    = get_arm(xy_v)
# For each sample, get composite score of its cell
comp_per_sample = composite[ci, cj]
mask_valid_smp  = np.isfinite(comp_per_sample)
arm_labels = ["Left", "Top", "Right"]
arm_colors = ["#e41a1c", "#4daf4a", "#377eb8"]
parts = ax.violinplot(
    [comp_per_sample[mask_valid_smp & (arm_v == a)] for a in range(3)],
    positions=range(3), showmedians=True
)
for pc, col in zip(parts["bodies"], arm_colors):
    pc.set_facecolor(col); pc.set_alpha(0.7)
parts["cmedians"].set_color("black")
ax.set_xticks(range(3)); ax.set_xticklabels(arm_labels, fontsize=11)
ax.set_ylabel("Score de difficulté")
ax.set_title("Distribution de difficulté par bras")
ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(f"{OUT}/05_difficulty_score.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved {OUT}/05_difficulty_score.png")

# ─── Per-arm statistics ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY — Analysis 05")
print("=" * 60)
print(f"  K-NN global  : mean={eucl_err.mean():.4f}  median={np.median(eucl_err):.4f}"
      f"  p90={np.percentile(eucl_err, 90):.4f}")
for a, aname in enumerate(["Left", "Top", "Right"]):
    m = arm_v == a
    e = eucl_err[m]
    d = comp_per_sample[m & mask_valid_smp]
    print(f"  {aname:6s}: kNN_err={e.mean():.4f}±{e.std():.4f}  "
          f"difficulty={d.mean():.3f}±{d.std():.3f}  n={m.sum()}")
print(f"\n  Hardest zones (top 25%):")
hi2, hj2 = np.where(np.isfinite(composite) & (composite >= hard_thr))
hard_arm = get_arm(np.stack([(hi2 + 0.5) / GRID, (hj2 + 0.5) / GRID], axis=1))
for a, aname in enumerate(["Left", "Top", "Right"]):
    print(f"    {aname}: {(hard_arm == a).sum()} cellules dans le top-25% difficile")
print("\nDone.")


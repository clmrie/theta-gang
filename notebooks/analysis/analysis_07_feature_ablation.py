"""Analysis 07 — Feature Ablation & Per-channel Ranking.

Questions :
  1. Quelle représentation du signal donne le meilleur décodage ?
     (spike count → peak amp → handcrafted → PCA waveform → waveform full)
  2. Combien de composantes PCA sont suffisantes ? (elbow curve)
  3. Quels canaux sont les plus informatifs ? (ρ position, F-stat bras)
  4. Est-ce que combiner handcrafted + PCA apporte quelque chose ?

Protocole : split temporel 90/10 (train/test) · k-NN k=20 · métrique = médian euclidean

Figures → artifacts/analysis_07/
  01_feature_ablation.png  : Erreur k-NN par type de features
  02_pca_elbow.png         : Elbow curve PCA n_components vs erreur
  03_channel_rho.png       : ρ(y) par canal (peak amplitude moyen)
  04_channel_arm_fstat.png : F-stat ANOVA bras par canal
  05_feature_corr.png      : Heatmap corrélation features handcrafted (73 dims)

Usage : python notebooks/analysis_07_feature_ablation.py
"""
import os, sys
from pathlib import Path
ROOT = Path(__file__).parent.parent; sys.path.insert(0, str(ROOT)); os.chdir(ROOT)

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pyarrow.parquet as pq
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from scipy.stats import spearmanr, f_oneway

OUT  = "artifacts/analysis_07"
N_CH = [6, 4, 6, 4]
SEED = 42
K_NN = 20
np.random.seed(SEED)
os.makedirs(OUT, exist_ok=True)

# Physical channel IDs from JSON (for labelling)
with open("data/M1199_PAG.json") as f:
    jcfg = json.load(f)
CH_IDS = []
for g in range(4):
    CH_IDS.append([jcfg[f"group{g}"][f"channel{c}"] for c in range(N_CH[g])])

ARM_NAMES  = ["Left", "Top", "Right"]
SH_COLORS  = ["#e41a1c", "#ff7f00", "#4daf4a", "#377eb8"]

def get_arm(xy):
    x, y = xy[:, 0], xy[:, 1]
    arm = np.full(len(x), 1, dtype=np.int8)
    arm[(x < 0.35) & (y > 0.45)] = 0
    arm[(x > 0.65) & (y > 0.45)] = 2
    return arm

def knn_error(F_train, y_train, F_test, y_test, k=K_NN):
    """k-NN regressor, returns (mean, median, p90) euclidean error."""
    sc = StandardScaler().fit(F_train)
    knn = KNeighborsRegressor(n_neighbors=k, algorithm="ball_tree", n_jobs=-1)
    knn.fit(sc.transform(F_train), y_train)
    pred = knn.predict(sc.transform(F_test))
    err  = np.sqrt(((y_test - pred) ** 2).sum(axis=1))
    return err.mean(), np.median(err), np.percentile(err, 90)

# ─── Step 1: Load pos + speedMask ────────────────────────────────────────────
print("Loading base columns …")
pf = pq.ParquetFile("data/M1199_PAG_stride4_win108_test.parquet")
base = pf.read(columns=["pos", "speedMask", "groups"])
xy_all = np.array(base.column("pos").combine_chunks().flatten(),
                  dtype=np.float32).reshape(-1, 4)[:, :2]
N_ALL  = len(xy_all)
speed_mask = np.array([bool(base.column("speedMask")[i][0]) for i in range(N_ALL)], dtype=bool)
grp_col  = base.column("groups").combine_chunks()
grp_off  = np.array(grp_col.offsets, dtype=np.int64)
grp_vals = np.array(grp_col.values,  dtype=np.int64)
sc_all   = np.bincount(
    np.repeat(np.arange(N_ALL, dtype=np.int64), np.diff(grp_off)) * 4 + grp_vals,
    minlength=N_ALL * 4
).reshape(N_ALL, 4).astype(np.float32)
del base, grp_col, grp_off, grp_vals

moving_idx = np.where(speed_mask)[0]
xy_mov     = xy_all[moving_idx]
sc_mov     = sc_all[moving_idx]
N_MOV      = len(moving_idx)

# ─── Step 2: Load waveforms → compute all feature types in one pass ───────────
print("Loading waveforms …")
# Per-window, per-shank storage:
mean_wf      = {}   # g → (N_mov, n_ch*32) mean waveform
peak_amp     = {}   # g → (N_mov, n_ch)    mean peak amplitude
trough_arr   = {}   # g → (N_mov, n_ch)    mean trough
amp_std_arr  = {}   # g → (N_mov, n_ch)    std of peak amp

for g, n_ch in enumerate(N_CH):
    col = pf.read(columns=[f"group{g}"]).column(f"group{g}")
    dim = n_ch * 32
    mw  = np.zeros((N_MOV, dim),  dtype=np.float32)
    pa  = np.zeros((N_MOV, n_ch), dtype=np.float32)
    tr  = np.zeros((N_MOV, n_ch), dtype=np.float32)
    as_ = np.zeros((N_MOV, n_ch), dtype=np.float32)

    for ii, idx in enumerate(moving_idx):
        flat = np.array(col[idx], dtype=np.float32)
        n_sp = len(flat) // dim
        if n_sp > 0:
            wf = flat.reshape(n_sp, n_ch, 32)   # (n_sp, n_ch, 32)
            mw[ii]  = wf.mean(axis=0).ravel()
            pa[ii]  = wf.max(axis=2).mean(axis=0)
            tr[ii]  = wf.min(axis=2).mean(axis=0)
            as_[ii] = wf.max(axis=2).std(axis=0) if n_sp > 1 else 0.0
    del col
    mean_wf[g]     = mw
    peak_amp[g]    = pa
    trough_arr[g]  = tr
    amp_std_arr[g] = as_
    print(f"  S{g}: done")

# Valid mask
valid = np.ones(N_MOV, dtype=bool)
for g in range(4): valid &= (mean_wf[g].sum(axis=1) != 0)
print(f"  Valid: {valid.sum()} / {N_MOV}")

xy_v   = xy_mov[valid]
sc_v   = sc_mov[valid]
arms_v = get_arm(xy_v)
N_V    = valid.sum()

# Temporal train/test split (90/10)
split  = int(N_V * 0.9)
tr_sl  = slice(None, split)
te_sl  = slice(split, None)
y_tr   = xy_v[tr_sl];  y_te = xy_v[te_sl]

# ─── Step 3: Build feature sets ───────────────────────────────────────────────
print("Building feature sets …")

# A: Spike count only (4 features)
F_sc = sc_v

# B: Peak amplitude per channel (20 features)
F_peak = np.hstack([peak_amp[g][valid] for g in range(4)])

# C: Handcrafted (73 features) — same as features.py
hc_parts = []
for g, n_ch in enumerate(N_CH):
    v_pa  = peak_amp[g][valid]
    v_tr  = trough_arr[g][valid]
    v_std = amp_std_arr[g][valid]
    v_sc  = sc_v[:, g:g+1]
    p2t   = (v_pa - v_tr).mean(axis=1, keepdims=True)
    hc_parts.append(np.hstack([v_sc, v_pa, v_tr, v_std, p2t]))
total_sc = sc_v.sum(axis=1, keepdims=True)
ratios   = sc_v / np.maximum(total_sc, 1)
F_hc     = np.hstack(hc_parts + [total_sc, ratios])   # 73 features
print(f"  Handcrafted dim: {F_hc.shape[1]}")

# D: Mean waveform full (640 features)
F_full = np.hstack([mean_wf[g][valid] for g in range(4)])

# E–J: PCA variants (fit on train only!)
print("Fitting PCA on train set …")
pca_models = {}
F_pcas     = {}

# Per-shank PCA, then concatenate
for n_comp in [1, 2, 3, 5, 8, 10, 15, 20, 30, 50]:
    parts = []
    for g in range(4):
        X   = mean_wf[g][valid]
        sc2 = StandardScaler().fit(X[tr_sl])
        X_sc = sc2.transform(X)
        nc   = min(n_comp, X_sc.shape[1], X_sc.shape[0] - 1)
        pca  = PCA(n_components=nc, random_state=SEED).fit(X_sc[tr_sl])
        parts.append(pca.transform(X_sc))
    F_pcas[n_comp] = np.hstack(parts)

F_pca5  = F_pcas[5]
F_pca20 = F_pcas[20]

# F: Combined handcrafted + PCA-20
F_comb = np.hstack([F_hc, F_pca20])

feature_sets = {
    "Spike count\n(4)":         F_sc,
    "Peak amp/ch\n(20)":        F_peak,
    "Handcrafted\n(73)":        F_hc,
    "PCA-5\n(20)":              F_pca5,
    "PCA-20\n(80)":             F_pca20,
    "HC + PCA-20\n(153)":       F_comb,
    "Full waveform\n(640)":     F_full,
}

# ─── Figure 1: Feature ablation ───────────────────────────────────────────────
print("\nFig 1: Feature ablation …")
results = {}
for name, F in feature_sets.items():
    me, med, p90 = knn_error(F[tr_sl], y_tr, F[te_sl], y_te)
    results[name] = (me, med, p90)
    print(f"  {name.replace(chr(10),' '):30s}  mean={me:.4f}  median={med:.4f}  p90={p90:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Ablation des features — erreur k-NN (k=20, split 90/10)",
             fontsize=13, fontweight="bold")

labels   = list(results.keys())
medians  = [results[k][1] for k in labels]
p90s     = [results[k][2] for k in labels]
colors   = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(labels)))

ax = axes[0]
bars = ax.bar(range(len(labels)), medians, color=colors, edgecolor="white", width=0.65)
ax.axhline(0.031, c="green", lw=2, ls="--", label="Cible doc (0.031)")
ax.axhline(medians[0], c="gray", lw=1, ls=":", alpha=0.7, label=f"Spike count baseline")
ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("Médiane erreur euclidean"); ax.set_title("Médiane erreur par type de features")
ax.legend(fontsize=9); ax.set_ylim(0, max(medians) * 1.2)
ax.grid(True, alpha=0.25, axis="y")
for i, (b, v) in enumerate(zip(bars, medians)):
    ax.text(b.get_x() + b.get_width()/2, v + 0.002, f"{v:.3f}",
            ha="center", va="bottom", fontsize=8)

ax = axes[1]
bars = ax.bar(range(len(labels)), p90s, color=colors, edgecolor="white", width=0.65)
ax.axhline(0.089, c="green", lw=2, ls="--", label="Cible doc p90 (0.089)")
ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("p90 erreur euclidean"); ax.set_title("p90 erreur par type de features")
ax.legend(fontsize=9); ax.set_ylim(0, max(p90s) * 1.2)
ax.grid(True, alpha=0.25, axis="y")
for i, (b, v) in enumerate(zip(bars, p90s)):
    ax.text(b.get_x() + b.get_width()/2, v + 0.003, f"{v:.3f}",
            ha="center", va="bottom", fontsize=8)

plt.tight_layout()
plt.savefig(f"{OUT}/01_feature_ablation.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved {OUT}/01_feature_ablation.png")

# ─── Figure 2: PCA elbow curve ────────────────────────────────────────────────
print("Fig 2: PCA elbow …")
pca_ns    = [1, 2, 3, 5, 8, 10, 15, 20, 30, 50]
pca_errs  = []
for n in pca_ns:
    F = F_pcas[n]
    _, med, _ = knn_error(F[tr_sl], y_tr, F[te_sl], y_te)
    pca_errs.append(med)
    print(f"  PCA-{n:2d} ({n*4:3d} dims) → médian={med:.4f}")

fig, ax = plt.subplots(figsize=(10, 5))
fig.suptitle("Elbow curve — PCA components vs erreur k-NN (all-shank)",
             fontsize=13, fontweight="bold")
ax.plot(pca_ns, pca_errs, "ko-", lw=2, ms=7)
for n, e in zip(pca_ns, pca_errs):
    ax.annotate(f"{e:.3f}", (n, e), textcoords="offset points",
                xytext=(0, 8), ha="center", fontsize=8)
ax.axhline(0.031, c="green", lw=2, ls="--", label="Cible doc (0.031)")
ax.axhline(results["Spike count\n(4)"][1], c="red", lw=1.5, ls=":",
           label=f"Spike count baseline ({results['Spike count(4)' if 'Spike count(4)' in results else list(results.keys())[0]][1]:.3f})")
ax.set_xlabel("Nb composantes PCA par shank (total = n×4)"); ax.set_ylabel("Médiane erreur euclidean")
ax.set_title("Elbow curve — combien de PCs pour le décodage ?")
ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/02_pca_elbow.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved {OUT}/02_pca_elbow.png")

# ─── Figure 3: Per-channel ρ(y) — peak amplitude ─────────────────────────────
print("Fig 3: Per-channel ρ(y) …")
ch_rho_x = []; ch_rho_y = []; ch_labels = []; ch_shank = []
for g, n_ch in enumerate(N_CH):
    for c in range(n_ch):
        pa_c  = peak_amp[g][valid, c]
        rx, _ = spearmanr(pa_c, xy_v[:, 0])
        ry, _ = spearmanr(pa_c, xy_v[:, 1])
        ch_rho_x.append(abs(rx))
        ch_rho_y.append(abs(ry))
        ch_labels.append(f"S{g}.ch{c}\n(e{CH_IDS[g][c]})")
        ch_shank.append(g)

ch_rho_x   = np.array(ch_rho_x)
ch_rho_y   = np.array(ch_rho_y)
n_total_ch = len(ch_labels)
order_y    = np.argsort(ch_rho_y)[::-1]   # sorted by ρ(y)

fig, axes = plt.subplots(2, 1, figsize=(16, 10))
fig.suptitle("Informativité par canal — peak amplitude ↔ position",
             fontsize=13, fontweight="bold")

x_pos = np.arange(n_total_ch)
for ax, rho, label_r in [(axes[0], ch_rho_y, "ρ(y)"), (axes[1], ch_rho_x, "ρ(x)")]:
    col_c = [SH_COLORS[ch_shank[i]] for i in range(n_total_ch)]
    ax.bar(x_pos, rho, color=col_c, edgecolor="white")
    ax.set_xticks(x_pos); ax.set_xticklabels(ch_labels, fontsize=6, rotation=45, ha="right")
    ax.set_ylabel(f"|Spearman {label_r}|")
    ax.set_title(f"Informativité spatiale par canal ({label_r})")
    ax.grid(True, alpha=0.25, axis="y")
    # Annotate top 5
    top5 = np.argsort(rho)[-5:][::-1]
    for t in top5:
        ax.annotate(f"{rho[t]:.3f}", (t, rho[t]), textcoords="offset points",
                    xytext=(0, 3), ha="center", fontsize=7, color="red")
    # Legend
    from matplotlib.patches import Patch
    patches = [Patch(fc=SH_COLORS[g], label=f"S{g}") for g in range(4)]
    ax.legend(handles=patches, fontsize=9, loc="upper right")

plt.tight_layout()
plt.savefig(f"{OUT}/03_channel_rho.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved {OUT}/03_channel_rho.png")

# ─── Figure 4: Per-channel arm discriminability (ANOVA F-stat) ───────────────
print("Fig 4: Channel arm F-stat …")
ch_fstat = []
for g, n_ch in enumerate(N_CH):
    for c in range(n_ch):
        pa_c    = peak_amp[g][valid, c]
        groups  = [pa_c[arms_v == a] for a in range(3)]
        groups  = [g2 for g2 in groups if len(g2) >= 5]
        if len(groups) == 3:
            f, _ = f_oneway(*groups)
            ch_fstat.append(f)
        else:
            ch_fstat.append(0.0)

ch_fstat = np.array(ch_fstat)
order_f  = np.argsort(ch_fstat)[::-1]

fig, axes = plt.subplots(1, 2, figsize=(18, 5))
fig.suptitle("Discriminabilité des bras par canal — ANOVA F-stat (peak amplitude)",
             fontsize=13, fontweight="bold")

ax = axes[0]
col_c = [SH_COLORS[ch_shank[i]] for i in range(n_total_ch)]
ax.bar(x_pos, ch_fstat, color=col_c, edgecolor="white")
ax.set_xticks(x_pos); ax.set_xticklabels(ch_labels, fontsize=6, rotation=45, ha="right")
ax.set_ylabel("F-statistic (ANOVA)"); ax.set_title("Discriminabilité bras par canal")
ax.grid(True, alpha=0.25, axis="y")
top5f = np.argsort(ch_fstat)[-5:][::-1]
for t in top5f:
    ax.annotate(f"{ch_fstat[t]:.0f}", (t, ch_fstat[t]), textcoords="offset points",
                xytext=(0, 3), ha="center", fontsize=7, color="red")
patches = [Patch(fc=SH_COLORS[g], label=f"S{g}") for g in range(4)]
ax.legend(handles=patches, fontsize=9, loc="upper right")

# Top 10 channels: boxplot of peak amp per arm
ax = axes[1]
top10_ch = np.argsort(ch_fstat)[-10:][::-1]
positions_box = []
colors_box    = []
all_data      = []
tick_labels   = []
for rank, ci in enumerate(top10_ch):
    # identify shank and channel within shank
    cumch = np.cumsum([0] + N_CH)
    g = np.searchsorted(cumch, ci + 1) - 1
    c = ci - cumch[g]
    pa_c = peak_amp[g][valid, c]
    for a in range(3):
        all_data.append(pa_c[arms_v == a])
    positions_box.extend([rank * 4, rank * 4 + 1, rank * 4 + 2])
    colors_box.extend(["#e41a1c", "#4daf4a", "#377eb8"])
    tick_labels.append(ch_labels[ci].replace("\n", " "))

bp = ax.boxplot(all_data, positions=positions_box, widths=0.7, patch_artist=True,
                flierprops=dict(ms=2, alpha=0.3), medianprops=dict(color="black", lw=2))
for patch, col in zip(bp["boxes"], colors_box):
    patch.set_facecolor(col); patch.set_alpha(0.7)
ax.set_xticks([r * 4 + 1 for r in range(10)])
ax.set_xticklabels(tick_labels, fontsize=7, rotation=45, ha="right")
ax.set_ylabel("Peak amplitude")
ax.set_title("Top-10 canaux discriminants — peak amp par bras\n(rouge=Left, vert=Top, bleu=Right)")
ax.grid(True, alpha=0.25, axis="y")

plt.tight_layout()
plt.savefig(f"{OUT}/04_channel_arm_fstat.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved {OUT}/04_channel_arm_fstat.png")

# ─── Figure 5: Handcrafted feature correlation heatmap ───────────────────────
print("Fig 5: Feature correlation heatmap …")
F_hc_sc = StandardScaler().fit_transform(F_hc)
corr     = np.corrcoef(F_hc_sc.T)

# Boundaries between shanks
boundaries = []
offset = 0
for g, n_ch in enumerate(N_CH):
    dim_g = 1 + n_ch * 3 + 1  # per-shank dim
    offset += dim_g
    boundaries.append(offset)

fig, ax = plt.subplots(figsize=(12, 10))
fig.suptitle("Corrélation Pearson — features handcrafted (73 dims)",
             fontsize=13, fontweight="bold")
im = ax.imshow(np.abs(corr), cmap="hot_r", vmin=0, vmax=1, aspect="equal")
plt.colorbar(im, ax=ax, fraction=0.03, label="|ρ|")
for b in boundaries[:-1]:
    ax.axhline(b - 0.5, c="cyan", lw=1.5)
    ax.axvline(b - 0.5, c="cyan", lw=1.5)
# Shank labels at center of each block
prev = 0
for g, b in enumerate(boundaries):
    mid = (prev + b) / 2
    ax.text(mid, -2, f"S{g}", ha="center", va="bottom", fontsize=9,
            color=SH_COLORS[g], fontweight="bold")
    ax.text(-2, mid, f"S{g}", ha="right", va="center", fontsize=9,
            color=SH_COLORS[g], fontweight="bold")
    prev = b
ax.set_title("Redondance inter-features (blocs = shanks, cyan = frontières)")
ax.set_xlabel("Feature index"); ax.set_ylabel("Feature index")
plt.tight_layout()
plt.savefig(f"{OUT}/05_feature_corr.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved {OUT}/05_feature_corr.png")

# ─── Summary ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SUMMARY — Analysis 07")
print("=" * 65)
print("\nFeature ablation (k-NN k=20, 90/10 split):")
for name, (me, med, p90) in results.items():
    print(f"  {name.replace(chr(10),' '):30s}  median={med:.4f}  p90={p90:.4f}")
print(f"\nPCA elbow:")
for n, e in zip(pca_ns, pca_errs):
    print(f"  PCA-{n:2d}: {e:.4f}")
print(f"\nTop-5 canaux ρ(y):")
for t in np.argsort(ch_rho_y)[-5:][::-1]:
    print(f"  {ch_labels[t].replace(chr(10),' ')}: ρ(y)={ch_rho_y[t]:.4f}  F={ch_fstat[t]:.1f}")


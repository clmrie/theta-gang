"""Analysis 03 — Waveform PCA / t-SNE.

Figures → artifacts/analysis_03/
  01_mean_waveforms.png    : Formes d'onde moyennes ± std par shank
  02_pca_variance.png      : Variance expliquée (scree + cumulatif)
  03_pca_scatter.png       : PC1 vs PC2 coloré par x / y / bras
  04_pc_position_corr.png  : Corrélations Spearman PC1-5 vs x, y
  05_tsne.png              : t-SNE coloré par bras et y_position

Usage : python notebooks/analysis_03_waveform_pca.py
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
from sklearn.manifold import TSNE
from scipy.stats import spearmanr

OUT   = "artifacts/analysis_03"
N_CH  = [6, 4, 6, 4]
N_PCA = 20          # components à calculer
N_TSNE_SAMPLES = 2000  # echantillon pour t-SNE (lourd)
SEED  = 42
np.random.seed(SEED)
os.makedirs(OUT, exist_ok=True)

ARM_NAMES   = ["Left", "Top", "Right"]
ARM_COLORS  = ["#e41a1c", "#4daf4a", "#377eb8"]

# ─── Arm detection (même logique que cell_map) ───────────────────────────────
def get_arm(xy):
    """Returns 0=Left, 1=Top, 2=Right for each position."""
    x, y = xy[:, 0], xy[:, 1]
    arm = np.full(len(x), -1, dtype=np.int8)
    arm[(x < 0.35) & (y > 0.45)] = 0        # Left arm
    arm[(x > 0.35) & (x < 0.65) & (y > 0.45)] = 1  # Top arm
    arm[(x > 0.65) & (y > 0.45)] = 2        # Right arm
    arm[arm == -1] = 1                       # fallback → Top
    return arm

# ─── Step 1: Load pos + speedMask (small) ────────────────────────────────────
print("Loading pos + speedMask …")
table_meta = pq.ParquetFile("data/M1199_PAG_stride4_win108_test.parquet").read(
    columns=["pos", "speedMask"])
xy_all   = np.array(table_meta.column("pos").combine_chunks().flatten(),
                    dtype=np.float32).reshape(-1, 4)[:, :2]
mov_flat = np.array(table_meta.column("speedMask").combine_chunks().flatten(), dtype=bool)
N_ALL    = len(xy_all)
del table_meta

# On reconstruit le mask par ligne (1 valeur de speedMask par ligne, stockée en list<bool>)
pf = pq.ParquetFile("data/M1199_PAG_stride4_win108_test.parquet")
sm_col = pf.read(columns=["speedMask"]).column("speedMask")
speed_mask = np.array([bool(sm_col[i][0]) for i in range(N_ALL)], dtype=bool)
moving_idx  = np.where(speed_mask)[0]
del sm_col

xy_mov   = xy_all[moving_idx]          # (N_mov, 2)
arms_mov = get_arm(xy_mov)
N_MOV    = len(moving_idx)
print(f"  Total={N_ALL}, Moving={N_MOV}")

# ─── Step 2: Pour chaque shank, charger les waveforms ────────────────────────
# On lit colonne par colonne pour limiter la mémoire
# Pour chaque window, on prend la waveform MOYENNE sur les spikes → (N_mov, n_ch*32)
# + on collecte des spikes individuels (sampleés) pour t-SNE

pf = pq.ParquetFile("data/M1199_PAG_stride4_win108_test.parquet")

mean_wf     = {}   # shank → (N_mov, n_ch*32) float32
indiv_wf    = {}   # shank → (N_sample, n_ch*32) float32 (sampleés)
indiv_xy    = {}   # shank → (N_sample, 2)
raw_shapes  = {}   # shank → list of (n_spikes, n_ch, 32) for mean/std figure

for g, n_ch in enumerate(N_CH):
    print(f"  Loading group{g} ({n_ch}ch) …")
    col = pf.read(columns=[f"group{g}"]).column(f"group{g}")
    wf_dim = n_ch * 32

    # Calcul mean waveform par window (moving only)
    mw_list  = np.zeros((N_MOV, wf_dim), dtype=np.float32)
    spk_all  = []   # tous les spikes (pour stats)
    indiv_list = []
    indiv_xy_list = []

    for ii, idx in enumerate(moving_idx):
        flat = np.array(col[idx], dtype=np.float32)
        n_spikes = len(flat) // wf_dim
        if n_spikes > 0:
            wf = flat.reshape(n_spikes, wf_dim)
            mw_list[ii] = wf.mean(axis=0)
            # Collecte pour t-SNE (reservoir sampling, max 10 spikes par window)
            n_take = min(n_spikes, 10)
            idx_spk = np.random.choice(n_spikes, n_take, replace=False)
            indiv_list.append(wf[idx_spk])
            indiv_xy_list.append(np.tile(xy_mov[ii], (n_take, 1)))
            if ii < 200:  # garde premières windows pour visualisation shapes
                spk_all.append(wf.reshape(n_spikes, n_ch, 32))
        else:
            # pas de spike → laisse à zéro (sera filtré si besoin)
            pass

        if (ii + 1) % 5000 == 0:
            print(f"    {ii+1}/{N_MOV}")

    del col  # libère la colonne

    mean_wf[g]    = mw_list
    raw_shapes[g] = np.concatenate(spk_all, axis=0) if spk_all else None  # (N_spk, n_ch, 32)

    # Reservoir sampling pour t-SNE
    if indiv_list:
        all_indiv = np.concatenate(indiv_list, axis=0)
        all_xy_i  = np.concatenate(indiv_xy_list, axis=0)
        if len(all_indiv) > N_TSNE_SAMPLES:
            perm = np.random.permutation(len(all_indiv))[:N_TSNE_SAMPLES]
            all_indiv = all_indiv[perm]
            all_xy_i  = all_xy_i[perm]
        indiv_wf[g]  = all_indiv
        indiv_xy[g]  = all_xy_i
        print(f"    → mean_wf: {mw_list.shape}, indiv: {all_indiv.shape}")

# Mask: windows avec au moins 1 spike dans TOUS les shanks (pour PCA cohérente)
valid_mask = (mean_wf[0].sum(axis=1) != 0) & \
             (mean_wf[1].sum(axis=1) != 0) & \
             (mean_wf[2].sum(axis=1) != 0) & \
             (mean_wf[3].sum(axis=1) != 0)
print(f"\n  Windows with all shanks active: {valid_mask.sum()} / {N_MOV}")

xy_v   = xy_mov[valid_mask]
arms_v = arms_mov[valid_mask]

# ─── PCA par shank ────────────────────────────────────────────────────────────
print("\nFitting PCA …")
pcas       = {}   # shank → fitted PCA object
coords     = {}   # shank → (N_valid, N_PCA)
scalers    = {}

for g in range(4):
    X = mean_wf[g][valid_mask]
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)
    pca    = PCA(n_components=N_PCA, random_state=SEED)
    Z      = pca.fit_transform(X_sc)
    pcas[g]    = pca
    coords[g]  = Z
    scalers[g] = scaler
    var_80 = np.searchsorted(np.cumsum(pca.explained_variance_ratio_), 0.80) + 1
    var_95 = np.searchsorted(np.cumsum(pca.explained_variance_ratio_), 0.95) + 1
    print(f"  S{g}: PC1={pca.explained_variance_ratio_[0]:.1%}  "
          f"80%→{var_80} comps  95%→{var_95} comps")

# ─── Figure 1: Mean waveforms ± std ──────────────────────────────────────────
print("\nFig 1: Mean waveforms …")
fig, axes = plt.subplots(4, 1, figsize=(14, 12))
fig.suptitle("Formes d'onde moyennes par shank — M1199_PAG", fontsize=13, fontweight="bold")
cmap = plt.cm.tab10

for g, n_ch in enumerate(N_CH):
    ax    = axes[g]
    spks  = raw_shapes[g]  # (N_spk, n_ch, 32)
    t     = np.arange(32)
    for ch in range(n_ch):
        wf_ch  = spks[:, ch, :]          # (N_spk, 32)
        mu     = wf_ch.mean(axis=0)
        sigma  = wf_ch.std(axis=0)
        col    = cmap(ch / max(n_ch - 1, 1))
        ax.plot(t + ch * 35, mu, color=col, lw=1.5, label=f"ch{ch}")
        ax.fill_between(t + ch * 35, mu - sigma, mu + sigma,
                        color=col, alpha=0.2)
        # Separator
        if ch < n_ch - 1:
            ax.axvline(ch * 35 + 32 + 1.5, color="gray", lw=0.5, ls="--", alpha=0.4)

    ax.set_title(f"S{g} ({n_ch} ch) — {len(spks)} spikes (premières 200 windows)")
    ax.set_xlabel("Sample (par canal, 32 pts/canal)"); ax.set_ylabel("Amplitude")
    ax.legend(fontsize=7, loc="upper right", ncol=n_ch)
    ax.grid(True, alpha=0.2)
    # X ticks = canal labels
    ax.set_xticks([ch * 35 + 16 for ch in range(n_ch)])
    ax.set_xticklabels([f"ch{ch}" for ch in range(n_ch)])

plt.tight_layout()
plt.savefig(f"{OUT}/01_mean_waveforms.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved {OUT}/01_mean_waveforms.png")

# ─── Figure 2: PCA variance expliquée ────────────────────────────────────────
print("Fig 2: PCA variance …")
fig, axes = plt.subplots(2, 4, figsize=(18, 7))
fig.suptitle("Variance expliquée PCA (waveforms moyennes par fenêtre)", fontsize=13, fontweight="bold")

for g in range(4):
    pca = pcas[g]
    evr = pca.explained_variance_ratio_
    cumev = np.cumsum(evr)
    k = np.arange(1, len(evr) + 1)

    # Scree plot
    ax = axes[0, g]
    ax.bar(k, evr * 100, color=cmap(g / 3.), edgecolor="white")
    ax.set_title(f"S{g} — Scree plot")
    ax.set_xlabel("Composante"); ax.set_ylabel("Variance expliquée (%)")
    ax.set_xlim(0.5, N_PCA + 0.5)

    # Cumulative
    ax2 = axes[1, g]
    ax2.plot(k, cumev * 100, "o-", color=cmap(g / 3.), lw=2)
    for thr in [0.80, 0.90, 0.95]:
        idx = np.searchsorted(cumev, thr) + 1
        ax2.axhline(thr * 100, ls="--", color="gray", lw=0.8, alpha=0.7)
        ax2.annotate(f"{int(thr*100)}% → PC{idx}", xy=(idx, thr*100),
                     xytext=(idx + 0.5, thr * 100 - 3), fontsize=7, color="gray")
    ax2.set_title(f"S{g} — Cumulative")
    ax2.set_xlabel("Nb composantes"); ax2.set_ylabel("Variance cumulée (%)")
    ax2.set_xlim(0.5, N_PCA + 0.5); ax2.set_ylim(0, 102)
    ax2.grid(True, alpha=0.25)

plt.tight_layout()
plt.savefig(f"{OUT}/02_pca_variance.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved {OUT}/02_pca_variance.png")

# ─── Figure 3: PCA scatter PC1 vs PC2 ────────────────────────────────────────
print("Fig 3: PCA scatter …")
fig = plt.figure(figsize=(18, 12))
fig.suptitle("PCA waveforms — PC1 vs PC2 par shank (windows mouvantes)", fontsize=13, fontweight="bold")
gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.45, wspace=0.35)

for g in range(4):
    Z = coords[g]

    # Col 0: coloré par x
    ax = fig.add_subplot(gs[g, 0])
    sc = ax.scatter(Z[:, 0], Z[:, 1], c=xy_v[:, 0], cmap="RdYlBu_r",
                    s=5, alpha=0.4, rasterized=True)
    plt.colorbar(sc, ax=ax, fraction=0.046, label="x")
    ax.set_title(f"S{g} — coloré par x")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")

    # Col 1: coloré par y
    ax = fig.add_subplot(gs[g, 1])
    sc = ax.scatter(Z[:, 0], Z[:, 1], c=xy_v[:, 1], cmap="RdYlBu_r",
                    s=5, alpha=0.4, rasterized=True)
    plt.colorbar(sc, ax=ax, fraction=0.046, label="y")
    ax.set_title(f"S{g} — coloré par y")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")

    # Col 2: coloré par bras
    ax = fig.add_subplot(gs[g, 2])
    for a, (aname, acol) in enumerate(zip(ARM_NAMES, ARM_COLORS)):
        mask = arms_v == a
        ax.scatter(Z[mask, 0], Z[mask, 1], c=acol, s=5, alpha=0.4,
                   label=f"{aname} ({mask.sum()})", rasterized=True)
    ax.legend(fontsize=7, loc="upper right", markerscale=3)
    ax.set_title(f"S{g} — coloré par bras")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")

plt.savefig(f"{OUT}/03_pca_scatter.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved {OUT}/03_pca_scatter.png")

# ─── Figure 4: Corrélations Spearman PC1-5 vs x, y ───────────────────────────
print("Fig 4: PC-position correlations …")
N_PC_CORR = 8
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
fig.suptitle("Corrélations Spearman |ρ| : PC → position", fontsize=13, fontweight="bold")

corr_table = {}
for g in range(4):
    Z = coords[g]
    rho_x = [abs(spearmanr(Z[:, k], xy_v[:, 0])[0]) for k in range(N_PC_CORR)]
    rho_y = [abs(spearmanr(Z[:, k], xy_v[:, 1])[0]) for k in range(N_PC_CORR)]
    corr_table[g] = {"rho_x": rho_x, "rho_y": rho_y}

    k_labels = [f"PC{k+1}" for k in range(N_PC_CORR)]
    x_k = np.arange(N_PC_CORR)

    ax = axes[0, g]
    bars = ax.bar(x_k - 0.2, rho_x, 0.4, label="ρ(x)", color="#4292c6")
    bars2 = ax.bar(x_k + 0.2, rho_y, 0.4, label="ρ(y)", color="#e6550d")
    ax.set_xticks(x_k); ax.set_xticklabels(k_labels, fontsize=8)
    ax.set_ylabel("|ρ| Spearman"); ax.set_title(f"S{g} — ρ par PC")
    ax.legend(fontsize=8); ax.set_ylim(0, 0.7); ax.grid(True, alpha=0.3, axis="y")

    # Best PC heatmap
    ax2 = axes[1, g]
    best_x = int(np.argmax(rho_x))
    best_y = int(np.argmax(rho_y))
    sc = ax2.scatter(xy_v[:, 0], xy_v[:, 1], c=Z[:, best_y], cmap="plasma",
                     s=5, alpha=0.5, rasterized=True)
    plt.colorbar(sc, ax=ax2, fraction=0.046, label=f"PC{best_y+1}")
    ax2.set_title(f"S{g} — meilleure PC pour y: PC{best_y+1} (ρ={rho_y[best_y]:.3f})")
    ax2.set_xlabel("x"); ax2.set_ylabel("y")

plt.tight_layout()
plt.savefig(f"{OUT}/04_pc_position_corr.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved {OUT}/04_pc_position_corr.png")

# ─── Figure 5: t-SNE ─────────────────────────────────────────────────────────
print("Fig 5: t-SNE (peut prendre 1-2 min) …")
fig, axes = plt.subplots(4, 2, figsize=(13, 18))
fig.suptitle("t-SNE waveforms individuels par shank", fontsize=13, fontweight="bold")

for g in range(4):
    X_i   = indiv_wf[g]
    xy_i  = indiv_xy[g]
    arm_i = get_arm(xy_i)

    # Normalize
    X_sc = StandardScaler().fit_transform(X_i)
    # PCA pré-processing pour t-SNE (50 dims)
    n_pre = min(50, X_sc.shape[1], X_sc.shape[0] - 1)
    Z_pre = PCA(n_components=n_pre, random_state=SEED).fit_transform(X_sc)
    # t-SNE
    Z_tsne = TSNE(n_components=2, perplexity=30, random_state=SEED,
                  max_iter=500).fit_transform(Z_pre)

    print(f"  S{g}: t-SNE done on {len(X_i)} spikes")

    ax = axes[g, 0]
    for a, (aname, acol) in enumerate(zip(ARM_NAMES, ARM_COLORS)):
        m = arm_i == a
        ax.scatter(Z_tsne[m, 0], Z_tsne[m, 1], c=acol, s=4, alpha=0.4,
                   label=f"{aname} ({m.sum()})", rasterized=True)
    ax.legend(fontsize=7, markerscale=3)
    ax.set_title(f"S{g} — t-SNE coloré par bras")
    ax.set_xlabel("tSNE-1"); ax.set_ylabel("tSNE-2")

    ax = axes[g, 1]
    sc = ax.scatter(Z_tsne[:, 0], Z_tsne[:, 1], c=xy_i[:, 1], cmap="RdYlBu_r",
                    s=4, alpha=0.4, rasterized=True)
    plt.colorbar(sc, ax=ax, fraction=0.046, label="y_pos")
    ax.set_title(f"S{g} — t-SNE coloré par y")
    ax.set_xlabel("tSNE-1"); ax.set_ylabel("tSNE-2")

plt.tight_layout()
plt.savefig(f"{OUT}/05_tsne.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved {OUT}/05_tsne.png")

# ─── Print summary ────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY — Analysis 03")
print("=" * 60)
for g in range(4):
    pca   = pcas[g]
    cumev = np.cumsum(pca.explained_variance_ratio_)
    v80   = np.searchsorted(cumev, 0.80) + 1
    v95   = np.searchsorted(cumev, 0.95) + 1
    ct    = corr_table[g]
    best_x_pc = int(np.argmax(ct["rho_x"])) + 1
    best_y_pc = int(np.argmax(ct["rho_y"])) + 1
    print(f"  S{g}: PC1={pca.explained_variance_ratio_[0]:.1%}"
          f"  80%→PC{v80}  95%→PC{v95}"
          f"  best_ρ(x)=PC{best_x_pc}({ct['rho_x'][best_x_pc-1]:.3f})"
          f"  best_ρ(y)=PC{best_y_pc}({ct['rho_y'][best_y_pc-1]:.3f})")

print("\nDone.")

"""Analysis 04 — Discriminabilité des bras (LDA / SVM / Feature importance).

Question centrale : les features (waveforms PCA + spike count) permettent-elles
de discriminer Left / Top / Right avec quelle précision ?

Figures → artifacts/analysis_04/
  01_lda_projection.png    : LDA 2D des features par shank + all-shank
  02_confusion_matrix.png  : Matrices de confusion (LDA, SVM) par shank
  03_feature_importance.png: Feature importance (coefficients LDA + SVM)
  04_cross_val_bars.png    : Accuracy CV 5-fold par shank + combinaisons
  05_boundary_map.png      : Décision LDA projetée sur le maze (heatmap proba)

Usage : python notebooks/analysis_04_arm_discriminability.py
"""
import os, sys
from pathlib import Path
ROOT = Path(__file__).parent.parent; sys.path.insert(0, str(ROOT)); os.chdir(ROOT)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pyarrow.parquet as pq
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

OUT  = "artifacts/analysis_04"
N_CH = [6, 4, 6, 4]
SEED = 42
np.random.seed(SEED)
os.makedirs(OUT, exist_ok=True)

ARM_NAMES  = ["Left", "Top", "Right"]
ARM_COLORS = ["#e41a1c", "#4daf4a", "#377eb8"]

# ─── Arm detection ────────────────────────────────────────────────────────────
def get_arm(xy):
    x, y = xy[:, 0], xy[:, 1]
    arm = np.full(len(x), -1, dtype=np.int8)
    arm[(x < 0.35) & (y > 0.45)] = 0
    arm[(x > 0.35) & (x < 0.65) & (y > 0.45)] = 1
    arm[(x > 0.65) & (y > 0.45)] = 2
    arm[arm == -1] = 1
    return arm

# ─── Step 1: Load pos + speedMask ────────────────────────────────────────────
print("Loading pos + speedMask …")
pf = pq.ParquetFile("data/M1199_PAG_stride4_win108_test.parquet")
sm_col = pf.read(columns=["speedMask", "pos"])
xy_all = np.array(sm_col.column("pos").combine_chunks().flatten(),
                  dtype=np.float32).reshape(-1, 4)[:, :2]
N_ALL  = len(xy_all)
speed_mask = np.array([bool(sm_col.column("speedMask")[i][0]) for i in range(N_ALL)], dtype=bool)
moving_idx = np.where(speed_mask)[0]
xy_mov  = xy_all[moving_idx]
arms_mov = get_arm(xy_mov)
N_MOV   = len(moving_idx)
print(f"  Moving={N_MOV}  |  Left={( arms_mov==0).sum()}  Top={(arms_mov==1).sum()}  Right={(arms_mov==2).sum()}")

# ─── Step 2: Build features per shank ────────────────────────────────────────
# Features = [mean_waveform (n_ch*32), spike_count (1)] par shank
# On utilise ça comme représentation compacte multi-dimensionnelle
print("Loading waveforms …")
mean_wf_all = {}   # g → (N_mov, n_ch*32)
sc_all      = {}   # g → (N_mov,)  spike count

for g, n_ch in enumerate(N_CH):
    print(f"  group{g} …")
    col  = pf.read(columns=[f"group{g}"]).column(f"group{g}")
    dim  = n_ch * 32
    mw   = np.zeros((N_MOV, dim), dtype=np.float32)
    sc   = np.zeros(N_MOV, dtype=np.float32)
    for ii, idx in enumerate(moving_idx):
        flat = np.array(col[idx], dtype=np.float32)
        n_sp = len(flat) // dim
        if n_sp > 0:
            wf = flat.reshape(n_sp, dim)
            mw[ii] = wf.mean(axis=0)
            sc[ii] = n_sp
    del col
    mean_wf_all[g] = mw
    sc_all[g]      = sc
    print(f"    mean spikes: {sc.mean():.2f}")

# Valid mask (toutes shanks actives)
valid = np.ones(N_MOV, dtype=bool)
for g in range(4):
    valid &= (mean_wf_all[g].sum(axis=1) != 0)
print(f"  Valid windows: {valid.sum()}")

xy_v   = xy_mov[valid]
arms_v = arms_mov[valid]

# Features par shank (PCA 20 comps pour réduire la dim + spike count)
print("Building features (PCA 20) …")
feats = {}
pcas  = {}
for g in range(4):
    X   = mean_wf_all[g][valid]
    sc  = sc_all[g][valid].reshape(-1, 1)
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)
    pca    = PCA(n_components=20, random_state=SEED)
    Z      = pca.fit_transform(X_sc)
    # concat PCA coords + log(spike_count)
    F      = np.hstack([Z, np.log1p(sc)])
    feats[g] = F
    pcas[g]  = pca
    print(f"  S{g}: feature shape = {F.shape}")

# All-shank concatenated
feats["all"] = np.hstack([feats[g] for g in range(4)])
print(f"  All-shank: {feats['all'].shape}")

# ─── Step 3: LDA 2D projection ───────────────────────────────────────────────
print("Fitting LDA …")
lda_models = {}
lda_coords = {}
for key in [0, 1, 2, 3, "all"]:
    lda = LinearDiscriminantAnalysis(n_components=2)
    Z   = lda.fit_transform(feats[key], arms_v)
    lda_models[key] = lda
    lda_coords[key] = Z

# ─── Figure 1: LDA 2D projection ─────────────────────────────────────────────
print("Fig 1: LDA projection …")
keys   = [0, 1, 2, 3, "all"]
titles = [f"S{g}" for g in range(4)] + ["All shanks"]
fig, axes = plt.subplots(1, 5, figsize=(22, 4))
fig.suptitle("Projection LDA 2D — discriminabilité des bras", fontsize=13, fontweight="bold")

for ax, key, title in zip(axes, keys, titles):
    Z = lda_coords[key]
    for a, (aname, acol) in enumerate(zip(ARM_NAMES, ARM_COLORS)):
        m = arms_v == a
        ax.scatter(Z[m, 0], Z[m, 1], c=acol, s=4, alpha=0.4,
                   label=f"{aname} ({m.sum()})", rasterized=True)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("LD1"); ax.set_ylabel("LD2")
    ax.legend(fontsize=7, markerscale=3, loc="best")
    ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig(f"{OUT}/01_lda_projection.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved {OUT}/01_lda_projection.png")

# ─── Step 4: Cross-validation accuracy ───────────────────────────────────────
print("Cross-validation …")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
cv_results = {}

for key in keys:
    F = feats[key]
    # LDA
    pipe_lda = Pipeline([("sc", StandardScaler()), ("lda", LinearDiscriminantAnalysis())])
    acc_lda  = cross_val_score(pipe_lda, F, arms_v, cv=cv, scoring="accuracy")
    # SVM RBF
    pipe_svm = Pipeline([("sc", StandardScaler()),
                          ("svm", SVC(kernel="rbf", C=1.0, random_state=SEED))])
    acc_svm  = cross_val_score(pipe_svm, F, arms_v, cv=cv, scoring="accuracy")

    cv_results[key] = {"lda_mean": acc_lda.mean(), "lda_std": acc_lda.std(),
                        "svm_mean": acc_svm.mean(), "svm_std": acc_svm.std()}
    label = f"S{key}" if isinstance(key, int) else "All"
    print(f"  {label:4s}  LDA={acc_lda.mean():.3f}±{acc_lda.std():.3f}  "
          f"SVM={acc_svm.mean():.3f}±{acc_svm.std():.3f}")

# ─── Figure 2: Cross-val accuracy bars ───────────────────────────────────────
print("Fig 2: CV accuracy bars …")
fig, ax = plt.subplots(figsize=(10, 5))
fig.suptitle("Accuracy 5-fold CV — classification bras (chance=33.3%)",
             fontsize=13, fontweight="bold")

labels   = [f"S{g}" for g in range(4)] + ["All"]
x        = np.arange(len(keys))
w        = 0.35
lda_m    = [cv_results[k]["lda_mean"] for k in keys]
lda_s    = [cv_results[k]["lda_std"]  for k in keys]
svm_m    = [cv_results[k]["svm_mean"] for k in keys]
svm_s    = [cv_results[k]["svm_std"]  for k in keys]

b1 = ax.bar(x - w/2, lda_m, w, yerr=lda_s, capsize=4,
            label="LDA", color="#4292c6", edgecolor="white")
b2 = ax.bar(x + w/2, svm_m, w, yerr=svm_s, capsize=4,
            label="SVM RBF", color="#e6550d", edgecolor="white")

ax.axhline(1/3, ls="--", color="gray", lw=1.2, label="Chance (33.3%)")
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=11)
ax.set_ylabel("Accuracy"); ax.set_ylim(0, 1.0)
ax.legend(fontsize=10); ax.grid(True, alpha=0.3, axis="y")

for bar, val in zip(list(b1) + list(b2), lda_m + svm_m):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.015,
            f"{val:.2f}", ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.savefig(f"{OUT}/02_cross_val_bars.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved {OUT}/02_cross_val_bars.png")

# ─── Figure 3: Confusion matrices ────────────────────────────────────────────
print("Fig 3: Confusion matrices …")
fig, axes = plt.subplots(2, 5, figsize=(22, 8))
fig.suptitle("Matrices de confusion (5-fold CV) — LDA (haut) · SVM (bas)",
             fontsize=13, fontweight="bold")

for col, (key, title) in enumerate(zip(keys, titles)):
    F = feats[key]
    for row, (model_name, pipe) in enumerate([
        ("LDA", Pipeline([("sc", StandardScaler()), ("lda", LinearDiscriminantAnalysis())])),
        ("SVM", Pipeline([("sc", StandardScaler()), ("svm", SVC(kernel="rbf", C=1.0, random_state=SEED))]))
    ]):
        preds = cross_val_predict(pipe, F, arms_v, cv=cv)
        cm    = confusion_matrix(arms_v, preds, normalize="true")
        ax    = axes[row, col]
        im    = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1, aspect="equal")
        ax.set_xticks(range(3)); ax.set_xticklabels(ARM_NAMES, fontsize=8)
        ax.set_yticks(range(3)); ax.set_yticklabels(ARM_NAMES, fontsize=8)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{cm[i,j]:.2f}", ha="center", va="center",
                        fontsize=9, color="white" if cm[i,j] > 0.6 else "black")
        acc = np.diag(cm).mean()
        ax.set_title(f"{title} — {model_name}\nacc={acc:.2f}", fontsize=9)
        if col == 0:
            ax.set_ylabel("Vrai");
        if row == 1:
            ax.set_xlabel("Prédit")

plt.tight_layout()
plt.savefig(f"{OUT}/03_confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved {OUT}/03_confusion_matrices.png")

# ─── Figure 4: LDA feature importance (coefficients) ─────────────────────────
print("Fig 4: Feature importance …")
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle("Importance des PCs (coefficients LD1, LD2 — LDA)",
             fontsize=13, fontweight="bold")

for g in range(4):
    F = feats[g]
    scaler = StandardScaler().fit(F)
    F_sc   = scaler.transform(F)
    lda    = LinearDiscriminantAnalysis(n_components=2).fit(F_sc, arms_v)
    coef1  = np.abs(lda.coef_).mean(axis=0)  # mean over 3 classes

    ax = axes[g // 2, g % 2]
    k_labels = [f"PC{k+1}" for k in range(20)] + ["log_sc"]
    k_x = np.arange(len(k_labels))
    colors = ["#4292c6"] * 20 + ["#e6550d"]
    ax.bar(k_x, coef1, color=colors, edgecolor="white")
    ax.set_xticks(k_x[::2]); ax.set_xticklabels(k_labels[::2], fontsize=7, rotation=45)
    ax.set_title(f"S{g} — |coef| LDA moyen sur 3 classes")
    ax.set_ylabel("|coef|"); ax.grid(True, alpha=0.2, axis="y")
    ax.axvline(19.5, color="red", lw=1.5, ls="--", alpha=0.6, label="spike_count")
    ax.legend(fontsize=8)
    # Annotate top 3
    top3 = np.argsort(coef1)[-3:][::-1]
    for t in top3:
        ax.annotate(k_labels[t], (t, coef1[t]), textcoords="offset points",
                    xytext=(0, 3), ha="center", fontsize=7, color="red")

plt.tight_layout()
plt.savefig(f"{OUT}/04_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved {OUT}/04_feature_importance.png")

# ─── Figure 5: Decision map LDA (all-shank) sur le maze ──────────────────────
print("Fig 5: Decision boundary map …")
# Utilise LDA sur all-shank pour prédire chaque position dans le maze
# Projette les probabilités sur une grille 50×50
lda_all = Pipeline([("sc", StandardScaler()),
                    ("lda", LinearDiscriminantAnalysis())])
lda_all.fit(feats["all"], arms_v)
proba_all = lda_all.predict_proba(feats["all"])   # (N, 3)

GRID  = 50
occ   = np.zeros((GRID, GRID))
p_arm = np.zeros((GRID, GRID, 3))

ci = np.clip((xy_v[:, 0] * GRID).astype(int), 0, GRID - 1)
cj = np.clip((xy_v[:, 1] * GRID).astype(int), 0, GRID - 1)
np.add.at(occ, (ci, cj), 1)
for a in range(3):
    np.add.at(p_arm[:, :, a], (ci, cj), proba_all[:, a])

valid_cells = occ >= 3
p_norm = np.where(valid_cells[:, :, None], p_arm / np.maximum(occ[:, :, None], 1), np.nan)

SKEL = np.array([[.15,.0,.15,.85],[.15,.85,.85,.85],[.85,.85,.85,.0]])
def skel_overlay(ax):
    for x1,y1,x2,y2 in SKEL: ax.plot([x1,x2],[y1,y2], c="white", lw=2, alpha=0.8)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Probabilité LDA (all-shank) d'appartenance à chaque bras",
             fontsize=13, fontweight="bold")

for a, (aname, acol) in enumerate(zip(ARM_NAMES, ARM_COLORS)):
    ax  = axes[a]
    pm  = p_norm[:, :, a]
    im  = ax.imshow(pm.T, origin="lower", extent=[0,1,0,1], cmap="hot",
                    aspect="equal", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, label="P(bras)")
    skel_overlay(ax)
    ax.set_title(f"P({aname})")
    ax.set_xlabel("x"); ax.set_ylabel("y")

plt.tight_layout()
plt.savefig(f"{OUT}/05_decision_map.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved {OUT}/05_decision_map.png")

# ─── Print full summary ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY — Analysis 04")
print("=" * 60)
print(f"{'Key':6s}  {'LDA':>10s}  {'SVM':>10s}")
for key, label in zip(keys, labels):
    r = cv_results[key]
    print(f"  {label:4s}  LDA={r['lda_mean']:.3f}±{r['lda_std']:.3f}"
          f"  SVM={r['svm_mean']:.3f}±{r['svm_std']:.3f}")
print("\nDone.")


"""Analysis 02 — Rate Maps par shank (smoothed + selectivity).

Figures → artifacts/analysis_02/
  rate_maps.png      : raw + lissé + sélectivité × 4 shanks
  summary.png        : comparaison inter-shanks + corrélations spatiales

Usage : python notebooks/analysis_02_rate_maps.py
"""
import os, sys
from pathlib import Path
ROOT = Path(__file__).parent.parent; sys.path.insert(0, str(ROOT)); os.chdir(ROOT)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pyarrow.parquet as pq
from scipy.ndimage import gaussian_filter
from scipy.stats import pearsonr

OUT   = "artifacts/analysis_02"
GRID  = 50
SIGMA = 1.5   # lissage gaussien (cells)
MIN_OCC = 3
SKEL  = np.array([[.15,.0,.15,.85],[.15,.85,.85,.85],[.85,.85,.85,.0]])
N_CH  = [6,4,6,4]
os.makedirs(OUT, exist_ok=True)

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading …")
table = pq.ParquetFile("data/M1199_PAG_stride4_win108_test.parquet").read(
    columns=["pos","speedMask","groups"])
xy   = np.array(table.column("pos").combine_chunks().flatten(),
                dtype=np.float32).reshape(-1,4)[:,:2]
mov  = np.array(table.column("speedMask").combine_chunks().flatten(), dtype=bool)
grp  = table.column("groups").combine_chunks()
offs = np.array(grp.offsets, dtype=np.int64)
vals = np.array(grp.values,  dtype=np.int64)
N    = len(xy)
sc   = np.bincount(np.repeat(np.arange(N,dtype=np.int64),np.diff(offs))*4+vals,
                   minlength=N*4).reshape(N,4).astype(np.int32)

mi   = np.where(mov)[0]; sp = int(len(mi)*.9)
txy  = xy[mi[:sp]]; tsc = sc[mi[:sp]]

def skel_overlay(ax, lw=1.5, c="red", a=.7):
    for x1,y1,x2,y2 in SKEL: ax.plot([x1,x2],[y1,y2],c=c,lw=lw,alpha=a)

def rate_map(xy, sc_g, grid=GRID, min_occ=MIN_OCC):
    occ=np.zeros((grid,grid)); ssum=np.zeros((grid,grid))
    ci=np.clip((xy[:,0]*grid).astype(int),0,grid-1)
    cj=np.clip((xy[:,1]*grid).astype(int),0,grid-1)
    np.add.at(occ,(ci,cj),1); np.add.at(ssum,(ci,cj),sc_g)
    r=np.full((grid,grid),np.nan)
    v=occ>=min_occ; r[v]=ssum[v]/occ[v]
    return r, occ, v

# ── Compute rate maps ─────────────────────────────────────────────────────────
rms, smrms, occs, valids = [], [], [], []
stats = []
for g in range(4):
    rm, occ, v = rate_map(txy, tsc[:,g])
    # smooth: NaN → 0 before filter, then re-mask
    rm0 = np.where(np.isfinite(rm), rm, 0.0)
    sm  = gaussian_filter(rm0, sigma=SIGMA)
    sm  = np.where(v, sm, np.nan)
    # selectivity index = peak / mean (> 1 = spatially selective)
    mean_r = np.nanmean(rm); peak_r = np.nanmax(rm)
    sel_idx = peak_r / mean_r if mean_r > 0 else 0
    pi,pj = np.unravel_index(np.nanargmax(rm), rm.shape)
    n_field = np.sum(np.isfinite(rm) & (rm > .5*peak_r))
    rms.append(rm); smrms.append(sm); occs.append(occ); valids.append(v)
    stats.append(dict(g=g, peak=peak_r, mean=mean_r, sel=sel_idx,
                      px=(pi+.5)/GRID, py=(pj+.5)/GRID, n_field=n_field))
    print(f"  S{g}: peak={peak_r:.2f} mean={mean_r:.2f} sel={sel_idx:.2f} "
          f"peak@({(pi+.5)/GRID:.2f},{(pj+.5)/GRID:.2f}) n_field={n_field}")

# ── Fig 1: Rate maps (raw | smoothed | sélectivité) ──────────────────────────
fig = plt.figure(figsize=(16, 4*4))
fig.suptitle("Rate Maps par shank — M1199_PAG", fontsize=14, fontweight="bold")
gs  = gridspec.GridSpec(4, 3, figure=fig, hspace=.35, wspace=.3)
PAL = plt.cm.tab10(np.linspace(0,.4,4))
EXT = [0,1,0,1]

for g in range(4):
    rm, sm = rms[g], smrms[g]
    st     = stats[g]

    # Raw
    ax = fig.add_subplot(gs[g, 0])
    im = ax.imshow(rm.T, origin="lower", extent=EXT, cmap="hot", aspect="equal",
                   vmin=0, vmax=np.nanpercentile(rm,98))
    plt.colorbar(im, ax=ax, fraction=.046, label="spk/win")
    skel_overlay(ax)
    ax.scatter([st["px"]], [st["py"]], c="cyan", s=80, marker="*",
               zorder=5, label=f"peak={st['peak']:.1f}")
    ax.legend(fontsize=7, loc="lower right")
    ax.set_title(f"S{g} ({N_CH[g]}ch) — brut"); ax.set_xlabel("x"); ax.set_ylabel("y")

    # Smoothed
    ax = fig.add_subplot(gs[g, 1])
    im = ax.imshow(sm.T, origin="lower", extent=EXT, cmap="viridis", aspect="equal")
    plt.colorbar(im, ax=ax, fraction=.046, label="spk/win (lissé)")
    skel_overlay(ax)
    ax.set_title(f"S{g} — lissé (σ={SIGMA})"); ax.set_xlabel("x"); ax.set_ylabel("y")

    # Sélectivité: log(rate / mean_rate), highlight hot zones
    ax = fig.add_subplot(gs[g, 2])
    sel_map = np.where(np.isfinite(rm) & (rm > 0),
                       np.log2(rm / st["mean"]), np.nan)
    vext = np.nanpercentile(np.abs(sel_map[np.isfinite(sel_map)]), 95)
    im = ax.imshow(sel_map.T, origin="lower", extent=EXT, cmap="RdBu_r",
                   aspect="equal", vmin=-vext, vmax=vext)
    plt.colorbar(im, ax=ax, fraction=.046, label="log₂(rate/mean)")
    skel_overlay(ax)
    ax.set_title(f"S{g} — sélectivité  (idx={st['sel']:.2f})")
    ax.set_xlabel("x"); ax.set_ylabel("y")

plt.savefig(f"{OUT}/rate_maps.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {OUT}/rate_maps.png")

# ── Fig 2: Summary inter-shanks ───────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle("Comparaison inter-shanks — M1199_PAG", fontsize=13, fontweight="bold")

# 2a — Selectivity index bar
ax = axes[0, 0]
bars = ax.bar(range(4), [s["sel"] for s in stats], color=PAL, edgecolor="white")
ax.set_xticks(range(4)); ax.set_xticklabels([f"S{g}" for g in range(4)])
ax.set_ylabel("Sélectivité (peak/mean)"); ax.set_title("Indice de sélectivité spatiale")
best = np.argmax([s["sel"] for s in stats])
bars[best].set_edgecolor("red"); bars[best].set_linewidth(2.5)
for b, s in zip(bars, stats):
    ax.text(b.get_x()+b.get_width()/2, s["sel"]+.02, f"{s['sel']:.2f}",
            ha="center", va="bottom", fontsize=9)

# 2b — Peak rate vs mean rate scatter
ax = axes[0, 1]
for s in stats:
    ax.scatter(s["mean"], s["peak"], color=PAL[s["g"]], s=120,
               label=f"S{s['g']}", zorder=3)
    ax.annotate(f"S{s['g']}", (s["mean"], s["peak"]),
                textcoords="offset points", xytext=(5,3), fontsize=9)
ax.set_xlabel("Mean rate (spk/win)"); ax.set_ylabel("Peak rate (spk/win)")
ax.set_title("Peak vs Mean rate par shank"); ax.grid(True, alpha=.3)

# 2c — n_field (cells firing > 50% peak)
ax = axes[0, 2]
ax.bar(range(4), [s["n_field"] for s in stats], color=PAL, edgecolor="white")
ax.set_xticks(range(4)); ax.set_xticklabels([f"S{g}" for g in range(4)])
ax.set_ylabel("# cellules > 50% peak"); ax.set_title("Étendue des champs de lieu")
for i, s in enumerate(stats):
    ax.text(i, s["n_field"]+2, str(s["n_field"]), ha="center", va="bottom", fontsize=9)

# 2d — Corrélations croisées entre shanks (Pearson sur cellules valides communes)
ax = axes[1, 0]
valid_union = valids[0] & valids[1] & valids[2] & valids[3]
corr = np.zeros((4,4))
for i in range(4):
    for j in range(4):
        a = rms[i][valid_union]; b = rms[j][valid_union]
        if len(a) > 10:
            corr[i,j], _ = pearsonr(a, b)
im = ax.imshow(corr, cmap="coolwarm", vmin=0, vmax=1, aspect="equal")
plt.colorbar(im, ax=ax, fraction=.046)
ax.set_xticks(range(4)); ax.set_xticklabels([f"S{g}" for g in range(4)])
ax.set_yticks(range(4)); ax.set_yticklabels([f"S{g}" for g in range(4)])
ax.set_title("Corrélations de Pearson\nentre rate maps")
for i in range(4):
    for j in range(4):
        ax.text(j, i, f"{corr[i,j]:.2f}", ha="center", va="center", fontsize=9,
                color="black" if corr[i,j] < .7 else "white")

# 2e — Overlay des 4 smoothed maps normalisées
ax = axes[1, 1]
cmaps = ["Reds","Blues","Greens","Oranges"]
for g in range(4):
    sm = smrms[g].copy()
    sm_norm = (sm - np.nanmin(sm)) / (np.nanmax(sm) - np.nanmin(sm) + 1e-9)
    sm_norm = np.where(np.isfinite(smrms[g]), sm_norm, 0)
    rgba = plt.cm.get_cmap(cmaps[g])(sm_norm)
    rgba[...,3] = np.where(valids[g], 0.4, 0.0)
    ax.imshow(rgba.transpose(1,0,2), origin="lower", extent=EXT, aspect="equal")
skel_overlay(ax, lw=2, c="black")
patches = [plt.Rectangle((0,0),1,1, fc=plt.cm.get_cmap(cmaps[g])(.7), label=f"S{g}") for g in range(4)]
ax.legend(handles=patches, fontsize=8, loc="lower right")
ax.set_title("Overlay shanks (normalisé)")
ax.set_xlabel("x"); ax.set_ylabel("y")

# 2f — Peak locations
ax = axes[1, 2]
occ0,_,_ = rate_map(txy, tsc[:,0])
ax.imshow(occs[0].T, origin="lower", extent=EXT, cmap="Greys", aspect="equal", alpha=.4)
skel_overlay(ax)
for s in stats:
    ax.scatter([s["px"]], [s["py"]], color=PAL[s["g"]], s=200, marker="*",
               zorder=5, label=f"S{s['g']} ({s['px']:.2f},{s['py']:.2f})", edgecolors="black")
ax.legend(fontsize=8); ax.set_title("Localisation des pics par shank")
ax.set_xlabel("x"); ax.set_ylabel("y")

plt.tight_layout()
plt.savefig(f"{OUT}/summary.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {OUT}/summary.png")

# ── Print cross-correlations ──────────────────────────────────────────────────
print("\n  Corrélations Pearson entre rate maps (cellules valides communes):")
for i in range(4):
    row = "  " + "".join(f"  S{j}={corr[i,j]:.3f}" for j in range(4))
    print(f"  S{i}: {row}")
print("\nDone.")


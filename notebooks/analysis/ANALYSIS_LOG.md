# Analysis Log — M1199_PAG Decoding

## Dataset

| Mouse | Shanks [ch] | Total | Moving | Train | Test | Stride/Win |
|---|---|---|---|---|---|---|
| M1199_PAG | 4 · [6,4,6,4] | 62 257 | 22 974 (36.9%) | 20 676 | 2 298 | 4 / 108 |

**Cible** : `median_mge ≈ 0.031` · `p90_mge ≈ 0.089` (pipeline TF labo, non reproduite)
**Baseline k-NN** (PCA-80, k=20) : médian eucl. = **0.337** — à battre largement

---

## 00 — Maze Artifacts (`scripts/build_maze.py`)

| Grille | Cellules valides | Couverture | Usage |
|---|---|---|---|
| 50×50 | 1 277 | 51.1% | Head A (classification) |
| 100×100 | 1 649 | 16.5% | Feasibility loss (distance transform) |

---

## 01 — Spatial Information (`artifacts/analysis_01/`)

| Shank | Ch | spk/win | SI b/spk | SI b/spl | ρ(x) | ρ(y) |
|---|---|---|---|---|---|---|
| S0 | 6 | 13.86 | 0.0017 | 0.0233 | +0.052 | +0.192 |
| S1 | 4 | 12.76 | 0.0016 | 0.0210 | +0.011 | +0.029 |
| S2 | 6 | 18.04 | 0.0012 | 0.0220 | -0.069 | +0.237 |
| **S3** | **4** | **11.76** | **0.0023** | **0.0268** | -0.015 | +0.158 |

**Bras** : Left=37.7% · Top=32.4% · Right=29.8%

→ **S3 meilleur SI** (qualité > quantité) · **y mieux encodé que x** partout · corrélations faibles → DL non-linéaire nécessaire

---

## 02 — Rate Maps (`artifacts/analysis_02/`)

| Shank | Peak rate | Sélectivité | Champ >50% |
|---|---|---|---|
| S0 | 25.25 | 1.83 | 935 cells |
| S1 | 26.00 | 2.05 | 605 cells |
| S2 | **38.00** | 2.12 | 518 cells |
| S3 | 24.00 | 2.07 | 596 cells |

**Corrélations inter-shank** : S0-S2 r=0.654 (redondants) · S1-S3 r=0.344 (complémentaires)

→ **S1+S3 à conserver ensemble** (info complémentaire malgré SI faible de S1) · S0-S2 quasi-redondants

---

## 03 — Waveform PCA / t-SNE (`artifacts/analysis_03/`)

| Shank | PC1 var% | 80% → n comps | best ρ(y) [PC] |
|---|---|---|---|
| S0 | 36.1% | 13 | 0.082 [PC4] |
| S1 | 32.4% | 11 | 0.080 [PC1] |
| S2 | 27.8% | 14 | **0.216 [PC6]** |
| **S3** | 29.2% | **9** | **0.269 [PC8]** |

→ PC1 waveform = 27–36% (vs ≥95.7% pour spike count PCA) → **espace waveform multi-dim** confirmé
→ Signal position dans PCs d'ordre élevé (PC4–PC8) → régression linéaire insuffisante, DL justifié
→ **S3 encode le mieux y** (ρ=0.269) · x faiblement encodé partout (max ρ=0.093)

---

## 04 — Discriminabilité des bras LDA/SVM (`artifacts/analysis_04/`)

**Distribution** : Left=5 135 · Top=12 551 · Right=5 288 ← class imbalance Top×2

| Input | LDA | SVM RBF |
|---|---|---|
| Shank seul (moy.) | ~55% | ~58% |
| **All-shank** | **58.8%** | **71.0%** |
| Chance | 33.3% | 33.3% |

→ Fusion non-linéaire multi-shank = +12% SVM vs +4% LDA → **attention apprise indispensable**
→ Shanks individuels quasi-équivalents → pas de shank dominant, tous utiles
→ `class_weight = [2.22, 0.91, 2.16]` (Left, Top, Right) pour CrossEntropy

---

## 05 — Zones difficiles (`artifacts/analysis_05/`)

**Score composite** = kNN×0.35 + var_neurale×0.35 + CV_spike×0.15 + cov×0.15

| Bras | kNN err | Diff. composite | Top-25% hard cells |
|---|---|---|---|
| Left | 0.329±0.130 | 0.481 | 60 |
| **Top** | **0.351±0.141** | **0.531** | **166** |
| Right | 0.328±0.125 | 0.537 | 104 |

→ **Top arm le plus dur** (jonctions + zone centrale hétérogène) → Kalman + geodesic loss prioritaires ici
→ Right arm haute variance neurale → signal peu reproductible malgré kNN similaire
→ Left arm le plus stable et bien couvert → signal fiable

---

## 06 — Dynamique temporelle & vitesse (`artifacts/analysis_06/`)

**Speed–firing** : ρ ≈ 0 pour tous les shanks (max |ρ|=0.015) → **vitesse ≠ confound neural**

| Signal | τ½ autocorr |
|---|---|
| Position x, y | **>160 ms** (très persistant) |
| Neural PCA | **~6 ms** (décore vite) |

**kNN error vs speed** : ρ = -0.016 global → vitesse n'affecte pas la décodabilité

→ Position persistante → Kalman `process_noise ≈ 0.001` confirmé (pas de sauts brusques)
→ Neural décore en 6ms → features par-fenêtre suffisantes, **pas besoin de LSTM longue mémoire**
→ **Direction tuning faible** → direction inutile comme feature
→ Top arm légèrement mieux décodé à haute vitesse (0.343 vs 0.354)

---

## 07 — Feature Ablation & Ranking par canal (`artifacts/analysis_07/`)

**Protocole** : split 90/10 · k-NN k=20 · médiane euclidean · PCA elbow sur [1…50] comps/shank

| Feature set | Dims | Médiane | p90 |
|---|---|---|---|
| Spike count | 4 | 0.420 | 0.583 |
| Peak amp/ch | 20 | 0.378 | 0.574 |
| PCA-20 waveform | 80 | 0.362 | 0.554 |
| Full waveform | 640 | 0.380 | 0.564 |
| **Handcrafted (HC)** | **73** | **0.337** | **0.554** |
| **HC + PCA-20** | **153** | **0.331** | **0.551** |

**PCA elbow** : optimum à **10 comps/shank** (médiane=0.360) — au-delà le bruit l'emporte

**Top-5 canaux ρ(y)** : S3.ch1/e22 (0.314·F=241) · S3.ch2/e23 (0.293) · S2.ch4/e13 (0.242) · S3.ch3/e11 (0.221) · S2.ch3/e14 (0.202)

→ **HC > PCA-20 > Full** : features domain-knowledge battent la PCA brute ; DL doit comprimer les waveforms
→ **HC + PCA-20 = meilleur combo** (0.331) → garder peak/trough/std comme features auxiliaires MLP
→ **S3 domine** : 3 des 5 meilleurs canaux sont dans S3 (e22 = canal #1 absolu) → surpondérer S3.ch1-3

---

## Actions pour le modèle — Bilan consolidé

| # | Insight | Action |
|---|---|---|
| 1 | Waveforms multi-dim (80% var → 9-14 PCs) | CNN sur waveforms brutes, pas projection linéaire |
| 2 | Fusion non-linéaire multi-shank critique (+12% SVM) | Attention shank-wise apprise |
| 3 | S3 meilleur (SI, ρ-y, top canaux e22/e23/e11) | Surpondérer S3 dans SpikeProjection |
| 4 | S1-S3 complémentaires (r=0.344 rate maps) | Garder S1 malgré SI faible |
| 5 | HC features (0.337) > PCA-20 (0.362) | Garder peak/trough/std comme features auxiliaires |
| 6 | HC + PCA-20 meilleur combo (0.331) | Concat HC + waveform embedding en entrée MLP |
| 7 | PCA optimum = 10/shank (elbow) | Dim cible pour projection waveform = 10-20 |
| 8 | y >> x dans tous les signaux | Loss : `w_y=2, w_x=1` |
| 9 | Top arm 2× plus fréquent + hardest | `class_weight=[2.22, 0.91, 2.16]` + geodesic loss |
| 10 | Position τ½ > 160ms · neural τ½ ≈ 6ms | Kalman `Q ≈ 0.001` · pas de RNN |
| 11 | Vitesse et direction ≠ informatifs | Ne pas ajouter speed/direction comme features |


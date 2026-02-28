<div align="center">
  <img src="ICM.png" alt="Institut du Cerveau — ICM" width="280"/>
  <br/><br/>

  # Neural Position Decoding from Multi-Shank Spike Sequences
  ### A Hierarchical Transformer with Uncertainty Quantification

  <br/>

  ![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
  ![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)
  ![License](https://img.shields.io/badge/License-MIT-green)

  **Theta Gang** · ICM Hackathon 2026
</div>

---

## Scientific Context

The hippocampus and parahippocampal structures encode spatial position through the coordinated activity of place cells, enabling an animal to build an internal map of its environment. Decoding this neural activity — reconstructing where an animal is from spike data alone — is both a fundamental neuroscience problem and a benchmark for neural population coding models.

This project tackles **real-time position decoding** from extracellular spike recordings collected as a mouse navigates a U-shaped maze. Rather than operating on firing rates over long time windows, we decode from **raw waveform sequences** within short 108 ms sliding windows, capturing both the temporal structure of population activity and the fine-grained morphology of individual spikes.

Our approach departs from classical linear decoders (PCA + k-NN, Bayesian) by learning directly from spike sequences with a **hierarchical Transformer**, jointly performing:
- **2D position regression** (x, y) with per-axis Gaussian uncertainty
- **Zone classification** (Left / Top / Right arm)
- **Curvilinear distance regression** along the maze skeleton

---

## The Maze & Dataset

The recording session (mouse `M1199_PAG`) was conducted in a U-shaped maze. Positions are normalised to [0, 1]². The maze skeleton is approximated by three connected segments:

```
  (0.15, 0.85) ──────────────── (0.85, 0.85)
       │                               │
  Left arm                        Right arm
       │                               │
  (0.15, 0.00)                   (0.85, 0.00)
```

| Parameter | Value |
|---|---|
| Mouse | M1199_PAG |
| Shanks | 4 · [6, 4, 6, 4] channels |
| Total windows | 62,257 |
| Moving windows (speedMask) | 22,974 (36.9%) |
| Train / Test split | 20,676 / 2,298 (temporal 90/10) |
| Window stride / duration | stride 4 / 108 ms |
| Baseline (k-NN, PCA-80, k=20) | median Eucl. = **0.337** |

---

## Exploratory Data Analysis

Seven analyses were conducted prior to model design to inform every architectural choice. Full results are in [`notebooks/analysis/ANALYSIS_LOG.md`](notebooks/analysis/ANALYSIS_LOG.md).

### 1 · Spatial Information per Shank

Skaggs spatial information (SI, bits/spike) measures how selectively a shank's firing rate encodes position.

| Shank | Channels | SI (b/spk) | ρ(x) | ρ(y) |
|---|---|---|---|---|
| S0 | 6 | 0.0017 | +0.052 | +0.192 |
| S1 | 4 | 0.0016 | +0.011 | +0.029 |
| S2 | 6 | 0.0012 | −0.069 | +0.237 |
| **S3** | **4** | **0.0023** | −0.015 | +0.158 |

> **Key insight:** y-position is consistently better encoded than x across all shanks. Linear correlations are weak (ρ < 0.27), confirming that a non-linear decoder is necessary.

### 2 · Rate Maps & Inter-Shank Redundancy

Occupancy-normalised firing rate maps reveal spatially selective responses. Cross-shank correlation analysis shows:
- S0–S2: r = 0.654 → **near-redundant**
- S1–S3: r = 0.344 → **complementary**

> **Key insight:** Retaining all four shanks is beneficial. S1 and S3 carry independent spatial information despite S1's lower SI.

### 3 · Waveform Dimensionality (PCA)

PC1 of spike waveforms explains only 27–36% of variance (vs. >95% for spike count PCA), confirming that **waveform space is high-dimensional**. Position information is concentrated in **high-order PCs** (PC4–PC8):

| Shank | PC1 var% | PCs for 80% var | Best ρ(y) |
|---|---|---|---|
| S0 | 36.1% | 13 | 0.082 [PC4] |
| S1 | 32.4% | 11 | 0.080 [PC1] |
| S2 | 27.8% | 14 | 0.216 [PC6] |
| **S3** | 29.2% | **9** | **0.269 [PC8]** |

> **Key insight:** Linear projection (PCA) is insufficient to extract position from waveforms. A 1-D CNN is used to learn task-relevant, non-linear waveform embeddings.

### 4 · Multi-Shank Fusion (LDA vs. SVM)

Comparing single-shank vs. all-shank decoders for arm classification:

| Decoder | Single-shank | All-shank | Gain |
|---|---|---|---|
| LDA (linear) | ~55% | 58.8% | +4% |
| SVM RBF (non-linear) | ~58% | **71.0%** | **+12%** |

> **Key insight:** Non-linear multi-shank fusion provides a 12-point improvement. This directly motivates the cross-shank Transformer attention mechanism.

### 5 · Difficult Zones

A composite difficulty score (k-NN error × 0.35 + neural variance × 0.35 + spike CV × 0.15 + coverage × 0.15) reveals spatial heterogeneity in decoding difficulty:

| Zone | k-NN error | Composite difficulty | Hard cells (top 25%) |
|---|---|---|---|
| Left | 0.329 ± 0.130 | 0.481 | 60 |
| **Top** | **0.351 ± 0.141** | **0.531** | **166** |
| Right | 0.328 ± 0.125 | 0.537 | 104 |

> **Key insight:** The Top arm (corridor) is hardest due to junction ambiguity and heterogeneous neural responses. This motivates the **feasibility loss** to keep predictions inside the corridor.

### 6 · Temporal Dynamics & Speed

| Signal | τ½ autocorrelation |
|---|---|
| Position (x, y) | **> 160 ms** — highly persistent |
| Neural PCA | **~6 ms** — decorrelates rapidly |

Speed–firing rate correlation: |ρ| < 0.015 for all shanks.

> **Key insight:** Position is a slowly-varying signal while neural activity decorrelates in ~6 ms. Per-window features are sufficient; recurrent models (LSTM) would not provide additional temporal memory. Speed and direction are not confounds and should not be added as features.

### 7 · Feature Ablation & Channel Ranking

| Feature set | Dims | Median Eucl. | p90 |
|---|---|---|---|
| Spike count | 4 | 0.420 | 0.583 |
| Peak amplitude / ch | 20 | 0.378 | 0.574 |
| PCA-20 waveform | 80 | 0.362 | 0.554 |
| Full waveform | 640 | 0.380 | 0.564 |
| Handcrafted (HC) | 73 | 0.337 | 0.554 |
| **HC + PCA-20** | **153** | **0.331** | **0.551** |

Top-5 channels by ρ(y): **S3.ch1/e22** (ρ=0.314), S3.ch2/e23, S2.ch4/e13, S3.ch3/e11, S2.ch3/e14.

> **Key insight:** Domain-knowledge features outperform raw PCA. The CNN must learn to compress waveforms more efficiently than PCA, focusing on channels identified as most informative (S3 dominates with 3 of the top 5 channels).

---

## Model Architecture

### Design Rationale

Each EDA finding is directly mapped to an architectural choice:

| Finding | Design choice |
|---|---|
| Non-linear position encoding in waveforms | Per-shank 1-D CNN encoder |
| High-dimensional waveform space (9–14 PCs) | CNN with learned compression |
| Non-linear multi-shank fusion critical (+12%) | Cross-shank Transformer attention |
| y encoded better than x | Per-axis Gaussian heads (independent σ) |
| Top arm hardest (junction ambiguity) | Feasibility loss + conditional zone heads |
| Neural τ½ ≈ 6 ms (no long memory needed) | Transformer (not LSTM) |

### Architecture

```
Spike sequence  (variable length T ≤ 128 spikes)
       │
       ▼
╔══════════════════════════════════════════════════╗
║         Per-shank CNN Encoder (×4 shanks)        ║
║  Conv1D(C→32, k=5) → ReLU                       ║
║  Conv1D(32→64, k=3) → ReLU → AdaptiveAvgPool    ║
╚══════════════════════════════════════════════════╝
       │  embed_dim = 64 per spike
       │
       + Shank embedding (learned, dim=64)
       + Sinusoidal positional encoding
       │
       ▼
╔══════════════════════════════════════════════════╗
║     Transformer Encoder  (2 layers, 4 heads)     ║
║  Self-attention across all spikes                ║
║  dim_feedforward = 256,  dropout = 0.2           ║
╚══════════════════════════════════════════════════╝
       │
     Masked mean pooling (ignore padding)
       │
  ┌────┴──────────────────────────────────┐
  ▼                                       ▼
Zone Classifier                  3 × Conditional heads
(3-way softmax)                  (one per zone: Left/Top/Right)
CrossEntropy loss                mu ∈ ℝ²,  log_sigma ∈ ℝ²
                                 GaussianNLL loss

  └────────────────┬──────────────────────┘
                   ▼
         Mixture of Gaussians
         (zone-probability weighted)
         ────────────────────────
         μ = Σₖ P(zₖ) · μₖ          (law of total expectation)
         σ² = Σₖ P(zₖ)(σₖ² + μₖ²) − μ²   (law of total variance)
                   │
              (x̂, ŷ, σ̂ₓ, σ̂ᵧ)

  Additional head: Curvilinear distance d ∈ [0,1]
                   Sigmoid + MSE loss

  Additional loss: FeasibilityLoss
                   = E[max(0, dist_to_skeleton − w)²]
                   penalises predictions outside the corridor
```

**Total parameters:** 163,344

### Loss Function

$$\mathcal{L} = \mathcal{L}_{\text{CE}} + \mathcal{L}_{\text{NLL}} + \lambda_d \cdot \mathcal{L}_{\text{MSE}}(d) + \lambda_{\text{feas}} \cdot \mathcal{L}_{\text{feas}}$$

| Term | Weight | Role |
|---|---|---|
| CrossEntropy | 1 | Zone classification |
| GaussianNLL | 1 | Conditional position regression per zone |
| MSE(d) | λ_d = 1.0 | Curvilinear distance supervision |
| FeasibilityLoss | λ_feas = 10.0 | Corridor adherence |

### Data Augmentation (training only)

| Technique | Parameter | Effect |
|---|---|---|
| Spike dropout | p = 0.15 | Robustness to missing spikes |
| Waveform noise | σ = 0.5 | Robustness to waveform variability |

---

## Training Protocol

**Cross-validation:** 5-fold KFold on the training set (temporal 90/10 pre-split).
**Ensemble:** Final predictions are the mean over 5 fold models; uncertainty combines aleatoric (mean σ²) and epistemic (variance of μ across folds) components.

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 1 × 10⁻³ (OneCycleLR) |
| Weight decay | 1 × 10⁻⁴ |
| Batch size | 64 |
| Max epochs | 30 |
| Early stopping patience | 7 epochs |
| Gradient clipping | max norm = 1.0 |

---

## Repository Structure

```
theta-gang/
│
├── src/                          # Core Python package
│   ├── config.py                 # All hyperparameters & paths (single source of truth)
│   ├── geometry.py               # U-maze skeleton geometry & curvilinear distance
│   ├── dataset.py                # Data loading, sequence reconstruction, PyTorch Dataset
│   ├── model.py                  # SpikeTransformerHierarchical architecture
│   ├── losses.py                 # FeasibilityLoss
│   └── trainer.py                # train_epoch / eval_epoch loops
│
├── scripts/                      # Executable entry points
│   ├── visualize_data.py         # Dataset EDA → figures/data/
│   ├── train.py                  # 5-fold training → checkpoints/ + figures/training/
│   └── evaluate.py               # Ensemble evaluation → figures/evaluation/ + outputs/
│
├── notebooks/                    # Reference Jupyter notebooks
│   ├── 02i_transformer_combined.ipynb   ← main notebook (original source)
│   └── analysis/                 # Pre-model exploratory analysis
│       ├── ANALYSIS_LOG.md       ← full analysis results
│       └── analysis_0{1–7}_*.py
│
├── figures/
│   ├── data/                     # EDA figures
│   ├── training/                 # Loss curves & LR schedule
│   ├── evaluation/               # All evaluation figures (13 total)
│   └── reference/                # Figures from the original notebook
│
├── artifacts/                    # Pre-computed maze artifacts (.npy, .png)
│   └── analysis_{01–07}/         # Per-analysis output figures
│
├── checkpoints/                  # Trained model weights (.pt) — gitignored
├── outputs/                      # Ensemble predictions (.npy) — gitignored
├── data/                         # Raw data files — gitignored
│
├── requirements.txt
└── .gitignore
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place data files in data/
#    data/M1199_PAG_stride4_win108_test.parquet
#    data/M1199_PAG.json

# 3. Explore the dataset  (~1 min)
python scripts/visualize_data.py

# 4. Train  (~30 min on GPU / ~3 h on CPU)
python scripts/train.py

# 5. Evaluate & generate all figures
python scripts/evaluate.py
```

All figures are saved automatically to `figures/` — no display required.

---

## Generated Figures

### Data exploration (`figures/data/`)
| Figure | Content |
|---|---|
| `01_maze_geometry.png` | Maze skeleton · curvilinear d · zone map |
| `02_curvilinear_dist.png` | Distribution of d with zone boundaries · zone pie chart |
| `03_spike_statistics.png` | Spikes per shank · sequence-length distribution |

### Training (`figures/training/`)
| Figure | Content |
|---|---|
| `01_total_loss.png` | Train / val total loss per fold |
| `02_loss_breakdown.png` | Per-component loss (CE · NLL · MSE-d · Feasibility) |
| `03_lr_schedule.png` | OneCycleLR learning-rate schedule |

### Evaluation (`figures/evaluation/`)
| Figure | Content |
|---|---|
| `01_scatter_pred_vs_true.png` | X, Y, d predicted vs. true scatter (R²) |
| `02_trajectory_uncertainty.png` | 2D trajectory · X/Y time-series with 2σ bands |
| `03_spatial_heatmaps.png` | Error / σ / corridor-distance heatmaps |
| `04_corridor_adherence.png` | Skeleton adherence scatter · distance histogram |
| `05_zone_dynamics.png` | Zone probabilities over time · curvilinear d · confusion map |
| `06_zone_heatmaps.png` | Predicted zone · error · confidence spatial heatmaps |
| `07_confusion_matrix.png` | 3-class confusion matrix (counts + normalised) |
| `08_uncertainty_decomposition.png` | Aleatoric vs. epistemic uncertainty decomposition |
| `09_fold_agreement.png` | Inter-fold zone agreement spatial map + histogram |
| `10_sigma_distribution.png` | Predicted σ distributions for X and Y axes |
| `11_uncertainty_calibration.png` | Spatial calibration: error vs. σ heatmaps |
| `12_zone_error_violin.png` ★ | Per-zone error distribution (violin + p25/50/75/90) |
| `13_calibration_curve.png` ★ | Reliability diagram: binned uncertainty calibration curve |

★ Figures not present in the original notebook.

---

## Uncertainty Quantification

The model produces **calibrated uncertainty estimates** by combining:

- **Aleatoric uncertainty** — intrinsic noise in the neural signal, captured by the per-head `log_sigma` outputs
- **Epistemic uncertainty** — model uncertainty due to limited data, captured by the variance of predictions across the 5 fold models

$$\sigma^2_{\text{total}} = \underbrace{\mathbb{E}[\sigma^2_k]}_{\text{aleatoric}} + \underbrace{\text{Var}[\mu_k]}_{\text{epistemic}}$$

A well-calibrated model satisfies: ~39% of errors < 1σ, ~86% < 2σ, ~99% < 3σ (2D Gaussian coverage).

---

<div align="center">
  <sub>Theta Gang · Institut du Cerveau (ICM) · Hackathon 2026</sub>
</div>

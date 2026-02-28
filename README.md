# Spike Transformer — Neural Position Decoding in a U-Maze

> **Hackathon project** — Theta Gang
> Decode mouse position in a U-shaped maze from multi-shank spike recordings using a hierarchical Transformer with uncertainty quantification.

---

## Problem

Given a 108-ms window of electrophysiology data (4 shanks, up to 128 spikes), predict:
- **(x, y)** — 2D normalised position of the mouse in the maze
- **zone** — which arm of the U the mouse is in (Left / Top / Right)
- **d** — normalised curvilinear distance along the U-skeleton

The model must also output **calibrated uncertainty** (σ per axis) and keep predictions **inside the maze corridor** (feasibility constraint).

---

## Dataset

| Mouse | Shanks [ch] | Total | Moving | Stride / Window |
|---|---|---|---|---|
| M1199_PAG | 4 · [6, 4, 6, 4] | 62 257 | 22 974 (36.9%) | 4 / 108 |

Place the data files in the `data/` directory (not committed):
```
data/
├── M1199_PAG_stride4_win108_test.parquet
└── M1199_PAG.json
```

---

## Model Architecture

```
Spike sequence (variable length, up to 128 spikes)
        │
        ▼
Per-shank CNN encoder (SpikeEncoder)
  Conv1D(C→32, k=5) → ReLU → Conv1D(32→64, k=3) → ReLU → AdaptiveAvgPool
        │
        ▼
+ Shank embedding  (learned per-shank identity vector)
+ Sinusoidal positional encoding
        │
        ▼
Transformer Encoder  (2 layers, 4 heads, embed_dim=64)
        │
     Mean pool over active (non-padded) tokens
        │
    ┌───┴─────────────────────────────────────────┐
    ▼                                             ▼
Zone classifier (3-way)               3 × conditional regression heads
CrossEntropy loss                     (mu, log-sigma) per zone
                                      GaussianNLL loss
    │                                             │
    └──────────────── Weighted mixture ───────────┘
                   (law of total expectation / variance)
                              │
                        (x̂, ŷ, σ̂)

Additional head: Curvilinear distance d ∈ [0,1] (Sigmoid + MSE)
Additional loss: FeasibilityLoss — penalise predictions outside the corridor
```

**Data augmentation (training only)**
- Spike dropout (p = 0.15): randomly mask individual spikes
- Additive Gaussian noise on waveforms (σ = 0.5)

---

## Repository Structure

```
theta-gang/
│
├── src/                        # Python package (importable)
│   ├── config.py               # All hyperparameters and paths
│   ├── geometry.py             # U-maze skeleton geometry
│   ├── dataset.py              # Data loading + SpikeSequenceDataset
│   ├── model.py                # SpikeTransformerHierarchical
│   ├── losses.py               # FeasibilityLoss
│   └── trainer.py              # train_epoch / eval_epoch
│
├── scripts/                    # Entry-point scripts
│   ├── visualize_data.py       # EDA: maze geometry + spike statistics
│   ├── train.py                # 5-fold cross-validation training
│   └── evaluate.py             # Ensemble evaluation + all figures
│
├── notebooks/                  # Reference notebooks
│   ├── 02i_transformer_combined.ipynb   ← main notebook (source)
│   ├── pipeline_pytorch_v4.ipynb
│   ├── pipeline_v2_transformer.ipynb
│   ├── pipeline_v3.ipynb
│   └── analysis/               # Exploratory analysis scripts
│       ├── ANALYSIS_LOG.md
│       └── analysis_0{1-7}_*.py
│
├── figures/                    # All generated figures
│   ├── data/                   # EDA figures (visualize_data.py)
│   ├── training/               # Loss curves (train.py)
│   ├── evaluation/             # Evaluation figures (evaluate.py)
│   └── reference/              # Figures from the original notebook
│
├── artifacts/                  # Pre-computed maze artifacts
│   ├── maze_meta.json
│   ├── maze_heatmap_{50,100}.png
│   ├── maze_*.npy
│   └── analysis_{01-07}/       # Per-analysis figures
│
├── outputs/                    # Saved predictions (.npy) — gitignored
├── checkpoints/                # Model weights (.pt) — gitignored
├── data/                       # Raw data — gitignored
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

# 3. Visualise the data (optional but recommended)
python scripts/visualize_data.py

# 4. Train (5-fold cross-validation, ~30 min on GPU)
python scripts/train.py

# 5. Evaluate ensemble + generate all figures
python scripts/evaluate.py
```

---

## Training Configuration

All hyperparameters live in [`src/config.py`](src/config.py):

| Parameter | Value | Description |
|---|---|---|
| `EMBED_DIM` | 64 | Transformer embedding dimension |
| `NHEAD` | 4 | Number of attention heads |
| `NUM_LAYERS` | 2 | Transformer encoder layers |
| `DROPOUT` | 0.2 | Dropout rate |
| `SPIKE_DROPOUT` | 0.15 | Fraction of spikes randomly masked |
| `NOISE_STD` | 0.5 | Gaussian waveform noise σ |
| `LR` | 1e-3 | Peak learning rate (OneCycleLR) |
| `EPOCHS` | 30 | Maximum training epochs |
| `PATIENCE` | 7 | Early-stopping patience |
| `N_FOLDS` | 5 | Number of cross-validation folds |
| `LAMBDA_D` | 1.0 | Weight on curvilinear-distance loss |
| `LAMBDA_FEAS` | 10.0 | Weight on feasibility loss |

---

## Generated Figures

### Data exploration (`figures/data/`)
| File | Content |
|---|---|
| `01_maze_geometry.png` | Skeleton overlay, curvilinear d, zone classification |
| `02_curvilinear_dist.png` | Distribution of d + zone pie chart |
| `03_spike_statistics.png` | Spikes per shank, sequence-length histogram |

### Training (`figures/training/`)
| File | Content |
|---|---|
| `01_total_loss.png` | Train / val total loss per fold |
| `02_loss_breakdown.png` | Per-component loss curves (CE, NLL, MSE-d, Feas) |
| `03_lr_schedule.png` | OneCycleLR learning-rate schedule |

### Evaluation (`figures/evaluation/`)
| File | Content |
|---|---|
| `01_scatter_pred_vs_true.png` | X, Y, d scatter plots |
| `02_trajectory_uncertainty.png` | 2D trajectory + X/Y uncertainty bands |
| `03_spatial_heatmaps.png` | Error / σ / corridor-distance heatmaps |
| `04_corridor_adherence.png` | Skeleton adherence scatter + histogram |
| `05_zone_dynamics.png` | Zone probabilities over time + d curves |
| `06_zone_heatmaps.png` | Zone / error / confidence spatial heatmaps |
| `07_confusion_matrix.png` | 3-class confusion matrix |
| `08_uncertainty_decomposition.png` | Aleatoric vs epistemic uncertainty |
| `09_fold_agreement.png` | Inter-fold zone prediction agreement |
| `10_sigma_distribution.png` | Predicted σ distribution for X and Y |
| `11_uncertainty_calibration.png` | Spatial error vs σ calibration |
| `12_zone_error_violin.png` ⭐ | Per-zone error violin + percentile bars |
| `13_calibration_curve.png` ⭐ | Reliability diagram (binned calibration curve) |

⭐ New figures not in the original notebook.

---

## Key Insights (from exploratory analysis)

| Finding | Implication |
|---|---|
| Shank S3 has highest spatial information (SI=0.0023 b/spk) | Architecture uses per-shank CNN encoders |
| Multi-shank non-linear fusion: +12% vs single shank (SVM) | Learned cross-shank attention is critical |
| y-position encoded better than x everywhere | Loss weighting: `w_y=2, w_x=1` possible extension |
| Top arm: 2× more samples + hardest to decode | Feasibility loss prioritises junction regions |
| Neural autocorrelation τ½ ≈ 6 ms (fast) | Per-window features sufficient; no LSTM needed |

---

## Results (ensemble of 5 folds)

| Metric | Value |
|---|---|
| Euclidean error (median) | see `figures/evaluation/` |
| R² X | see `figures/evaluation/` |
| R² Y | see `figures/evaluation/` |
| Zone accuracy | see `figures/evaluation/` |
| Outside corridor | see `figures/evaluation/` |

*Run `python scripts/evaluate.py` to populate results.*

---

## Team

**Theta Gang** — Hackathon 2026

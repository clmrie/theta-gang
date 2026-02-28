<div align="center">
  <img src="ICM.png" alt="Institut du Cerveau — ICM" width="240"/>
  <br/><br/>

  # Neural Position Decoding — Spike Transformer
  **Theta Gang · ICM Hackathon 2026**

  ![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
  ![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)
</div>

---

## Problem

Decode the 2D position of a mouse navigating a U-shaped maze from short windows (108 ms) of multi-shank extracellular spike recordings. The model must also output **calibrated uncertainty** and keep predictions inside the maze corridor.

**Baseline** (k-NN, PCA-80): median Euclidean error = 0.337 · **Dataset**: 22,974 moving windows, 4 shanks [6, 4, 6, 4] channels.

---

## Key Techniques

### Hierarchical zone decomposition
The U-maze is split into **3 zones** (Left / Top / Right arm) using a curvilinear distance `d ∈ [0, 1]` computed along the skeleton. The model jointly:
1. **classifies** the zone (3-way softmax, CrossEntropy)
2. **regresses** position within each zone via **3 conditional heads** (one per zone)
3. **regresses** `d` directly (MSE, Sigmoid head)

Final position = zone-probability-weighted **Gaussian mixture** (law of total expectation / variance), producing both a mean `μ` and uncertainty `σ` per axis.

### Feasibility loss
A custom `FeasibilityLoss` penalises predictions that fall **outside the corridor**: for each predicted point, the excess distance beyond the corridor half-width is squared and averaged. Weighted at λ = 10 to strongly constrain predictions to the maze geometry.

### Data augmentation
Two techniques are applied **during training only**:
- **Spike dropout** (p = 0.15): randomly mask individual spikes to force robustness to missing activity
- **Gaussian waveform noise** (σ = 0.5): additive noise on raw waveforms to prevent overfitting on waveform shapes

### Transformer backbone
Per-shank **1-D CNN** encoders compress each spike waveform into a 64-dim embedding. A **Transformer encoder** (2 layers, 4 heads) then attends across all spikes in the window, learning cross-shank interactions that linear decoders cannot capture (+12% zone accuracy vs. SVM on single shank).

### Ensemble & uncertainty
**5-fold cross-validation** trains 5 independent models. At inference, predictions are averaged and uncertainty is decomposed into **aleatoric** (intrinsic signal noise, from the per-head sigma outputs) and **epistemic** (model uncertainty, from the variance of predictions across folds).

---

## Repository Structure

```
src/              ← Python package (config, geometry, dataset, model, losses, trainer)
scripts/          ← train.py · evaluate.py · visualize_data.py
notebooks/        ← 02i_transformer_combined.ipynb (main) + analysis/
figures/          ← data/ · training/ · evaluation/ · reference/
artifacts/        ← pre-computed maze artifacts
checkpoints/      ← model weights (gitignored)
outputs/          ← predictions .npy (gitignored)
data/             ← raw data (gitignored)
```

The main notebook [`notebooks/02i_transformer_combined.ipynb`](notebooks/02i_transformer_combined.ipynb) is the original source; the `src/` and `scripts/` folders are its clean, modular equivalent.

---

## Quick Start

```bash
pip install -r requirements.txt

# Place data files in data/
#   M1199_PAG_stride4_win108_test.parquet
#   M1199_PAG.json

python scripts/visualize_data.py   # EDA figures → figures/data/
python scripts/train.py            # 5-fold training → checkpoints/
python scripts/evaluate.py         # Evaluation + all figures → figures/evaluation/
```

---

<div align="center">
  <sub>Theta Gang · Institut du Cerveau (ICM) · 2026</sub>
</div>


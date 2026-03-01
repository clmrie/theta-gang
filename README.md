<div align="center">
  <img src="ICM.png" alt="Institut du Cerveau — ICM" height="80" style="vertical-align:middle"/>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="hacktion.png" alt="Hacktion" height="80" style="vertical-align:middle"/>
  <br/><br/>

  # Spike Transformer — Neural Position Decoding
  **Theta Gang · ICM Hackathon 2026**

  ![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
  ![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)
  ![License](https://img.shields.io/badge/License-MIT-green)

  **[Live Demo & Presentation](https://clmrie.github.io/theta-gang/)**
</div>

---

Decode the 2D position of a mouse navigating a U-shaped maze from 108 ms windows of multi-shank spike recordings. A hierarchical Transformer jointly classifies the maze zone, regresses position with calibrated uncertainty, and constrains predictions to the corridor geometry.

| | Baseline (k-NN + PCA-80) | Spike Transformer |
|---|---|---|
| Median Euclidean error | 0.337 | **0.118** |
| Zone classification | — | 90.2% |
| Predictions inside corridor | — | 97.8% |

---

## Quick Start

```bash
pip install -r requirements.txt

# Place data in data/
python scripts/visualize_data.py   # EDA
python scripts/train.py            # 5-fold CV training
python scripts/evaluate.py         # Ensemble evaluation + figures
```

---

## Structure

```
src/          config, dataset, model, geometry, losses, trainer
scripts/      train.py · evaluate.py · visualize_data.py
notebooks/    development notebooks + 7 feature analyses
figures/      reference · data · training · evaluation
artifacts/    pre-computed maze masks and distance maps
```

---

<div align="center">
  <sub>Theta Gang · Institut du Cerveau (ICM) · 2026</sub>
</div>

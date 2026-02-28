"""Global configuration — hyperparameters, paths, and maze geometry constants.

All scripts import from this file so that a single edit is enough to change any
parameter across the entire project.
"""
from pathlib import Path

import numpy as np

# ── Root directory (one level above src/) ────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent

# ── Data / output paths ───────────────────────────────────────────────────────
DATA_DIR        = ROOT / "data"
OUTPUTS_DIR     = ROOT / "outputs"
CHECKPOINTS_DIR = ROOT / "checkpoints"
FIGURES_DIR     = ROOT / "figures"

PARQUET_NAME = "M1199_PAG_stride4_win108_test.parquet"
JSON_NAME    = "M1199_PAG.json"

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42

# ── U-maze skeleton geometry ──────────────────────────────────────────────────
# Each row: [x1, y1, x2, y2]  (normalised coordinates in [0, 1])
SKELETON_SEGMENTS = np.array(
    [
        [0.15, 0.00, 0.15, 0.85],   # Left arm   (bottom → top)
        [0.15, 0.85, 0.85, 0.85],   # Top corridor (left  → right)
        [0.85, 0.85, 0.85, 0.00],   # Right arm  (top    → bottom)
    ],
    dtype=np.float32,
)
CORRIDOR_HALF_WIDTH = 0.15
N_ZONES             = 3
ZONE_NAMES          = ["Left", "Top", "Right"]

# ── Model architecture ────────────────────────────────────────────────────────
MAX_SEQ_LEN   = 128
MAX_CHANNELS  = 6       # maximum channels across all shanks (shank 0 and 2 have 6)
EMBED_DIM     = 64
NHEAD         = 4
NUM_LAYERS    = 2
DROPOUT       = 0.2
SPIKE_DROPOUT = 0.15    # probability of randomly masking a spike during training
NOISE_STD     = 0.5     # std of additive Gaussian noise on waveforms

# ── Training schedule ─────────────────────────────────────────────────────────
LR           = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS       = 30
PATIENCE     = 7        # early-stopping patience
BATCH_SIZE   = 64
N_FOLDS      = 5        # number of cross-validation folds

# ── Loss weights ──────────────────────────────────────────────────────────────
LAMBDA_D    = 1.0       # weight on curvilinear-distance MSE loss
LAMBDA_FEAS = 10.0      # weight on feasibility (corridor adherence) loss

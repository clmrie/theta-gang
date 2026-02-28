"""U-maze skeleton geometry: curvilinear distance and zone classification.

The U-maze is approximated as three connected line segments:
    Left arm  → Top corridor → Right arm
All coordinates are normalised to [0, 1].
"""
import numpy as np

from src.config import CORRIDOR_HALF_WIDTH, N_ZONES, SKELETON_SEGMENTS

# ── Pre-computed segment properties ──────────────────────────────────────────
_SEG_LENGTHS  = np.array(
    [np.sqrt((s[2] - s[0]) ** 2 + (s[3] - s[1]) ** 2) for s in SKELETON_SEGMENTS]
)
_TOTAL_LENGTH = float(_SEG_LENGTHS.sum())        # ≈ 2.40 (normalised units)
_CUM_LENGTHS  = np.concatenate([[0.0], np.cumsum(_SEG_LENGTHS)])

# Zone thresholds along the normalised curvilinear axis d ∈ [0, 1]
D_LEFT_END    = float(_CUM_LENGTHS[1] / _TOTAL_LENGTH)   # ≈ 0.354
D_RIGHT_START = float(_CUM_LENGTHS[2] / _TOTAL_LENGTH)   # ≈ 0.646


# ── Core geometry functions ───────────────────────────────────────────────────

def project_point_on_segment(px, py, x1, y1, x2, y2):
    """Project point (px, py) onto segment [(x1,y1) → (x2,y2)].

    Returns
    -------
    t       : float  — normalised position along segment [0, 1]
    dist    : float  — distance from the point to its projection
    proj_x  : float  — x-coordinate of the projection
    proj_y  : float  — y-coordinate of the projection
    """
    dx, dy     = x2 - x1, y2 - y1
    seg_len_sq = dx ** 2 + dy ** 2
    if seg_len_sq < 1e-12:
        return 0.0, float(np.hypot(px - x1, py - y1)), float(x1), float(y1)
    t      = float(np.clip(((px - x1) * dx + (py - y1) * dy) / seg_len_sq, 0.0, 1.0))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return t, float(np.hypot(px - proj_x, py - proj_y)), proj_x, proj_y


def compute_curvilinear_distance(x, y):
    """Normalised curvilinear distance d ∈ [0, 1] along the U-skeleton."""
    best_dist = np.inf
    best_d    = 0.0
    for i, (x1, y1, x2, y2) in enumerate(SKELETON_SEGMENTS):
        t, dist, _, _ = project_point_on_segment(x, y, x1, y1, x2, y2)
        if dist < best_dist:
            best_dist = dist
            best_d    = (_CUM_LENGTHS[i] + t * _SEG_LENGTHS[i]) / _TOTAL_LENGTH
    return float(best_d)


def compute_distance_to_skeleton(x, y):
    """Minimum Euclidean distance from (x, y) to the U-skeleton."""
    return min(
        project_point_on_segment(x, y, *seg)[1]
        for seg in SKELETON_SEGMENTS
    )


def d_to_zone(d):
    """Map normalised curvilinear distance to zone index.

    0 = Left arm  (d < D_LEFT_END)
    1 = Top corridor
    2 = Right arm (d >= D_RIGHT_START)
    """
    if d < D_LEFT_END:
        return 0
    if d < D_RIGHT_START:
        return 1
    return 2


# ── Batch helpers ─────────────────────────────────────────────────────────────

def compute_geometry(positions):
    """Batch-compute curvilinear distances and zone labels.

    Parameters
    ----------
    positions : (N, 2) float array of (x, y) positions

    Returns
    -------
    curvilinear_d : (N,) float32
    zone_labels   : (N,) int64
    """
    curvilinear_d = np.array(
        [compute_curvilinear_distance(x, y) for x, y in positions],
        dtype=np.float32,
    )
    zone_labels = np.array([d_to_zone(d) for d in curvilinear_d], dtype=np.int64)
    return curvilinear_d, zone_labels


def print_geometry_stats(positions, curvilinear_d, zone_labels):
    """Print a short summary of geometry statistics."""
    from src.config import ZONE_NAMES
    dist_to_skel = np.array(
        [compute_distance_to_skeleton(x, y) for x, y in positions]
    )
    print(f"\nCurvilinear d : min={curvilinear_d.min():.4f}  "
          f"max={curvilinear_d.max():.4f}  mean={curvilinear_d.mean():.4f}")
    print(f"Zone thresholds : Left d<{D_LEFT_END:.4f}  "
          f"Top {D_LEFT_END:.4f}–{D_RIGHT_START:.4f}  Right d>={D_RIGHT_START:.4f}")
    print("\nZone distribution:")
    for z in range(N_ZONES):
        n = (zone_labels == z).sum()
        print(f"  {ZONE_NAMES[z]:6s} (class {z}): {n}  ({n / len(zone_labels):.1%})")
    print(f"\nDistance to skeleton : mean={dist_to_skel.mean():.4f}  "
          f"max={dist_to_skel.max():.4f}  "
          f"in corridor={( dist_to_skel < CORRIDOR_HALF_WIDTH).mean():.1%}")

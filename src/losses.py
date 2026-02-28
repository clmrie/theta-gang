"""Custom loss functions for the Spike Transformer."""
import torch
import torch.nn as nn


class FeasibilityLoss(nn.Module):
    """Penalise position predictions that fall outside the U-shaped corridor.

    For each predicted (x, y), the loss computes the minimum distance to the
    skeleton and returns the mean squared excess beyond `corridor_half_width`.

    A prediction inside the corridor contributes zero; one outside contributes
    (distance - corridor_half_width)^2.
    """

    def __init__(self, skeleton_segments, corridor_half_width: float):
        """
        Parameters
        ----------
        skeleton_segments    : (S, 4) array-like — [x1, y1, x2, y2] per segment
        corridor_half_width  : float — half-width of the allowed corridor
        """
        super().__init__()
        self.register_buffer(
            "segments",
            torch.tensor(skeleton_segments, dtype=torch.float32),
        )
        self.corridor_half_width = corridor_half_width

    def forward(self, xy_pred: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        xy_pred : (B, 2) tensor — predicted (x, y) positions

        Returns
        -------
        Scalar feasibility loss (mean squared corridor violation).
        """
        px, py    = xy_pred[:, 0], xy_pred[:, 1]
        distances = []

        for i in range(self.segments.shape[0]):
            x1, y1, x2, y2 = self.segments[i]
            dx, dy  = x2 - x1, y2 - y1
            t       = ((px - x1) * dx + (py - y1) * dy) / (dx ** 2 + dy ** 2 + 1e-8)
            t       = t.clamp(0.0, 1.0)
            proj_x  = x1 + t * dx
            proj_y  = y1 + t * dy
            dist    = torch.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2 + 1e-8)
            distances.append(dist)

        min_dist = torch.stack(distances, dim=1).min(dim=1).values
        return torch.relu(min_dist - self.corridor_half_width).pow(2).mean()


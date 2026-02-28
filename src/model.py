"""Hierarchical Spike Transformer for neural position decoding.

Architecture overview
---------------------
Input  : variable-length sequence of multi-channel spike waveforms
         (one waveform per spike, recorded across 4 shanks)

Encoder
  1. Per-shank 1-D CNN  (SpikeEncoder)   : waveform → dense embedding
  2. Shank embedding                      : add learned shank identity
  3. Sinusoidal positional encoding       : add temporal position
  4. Transformer encoder                  : attend across all spikes

Heads (applied to mean-pooled Transformer output)
  • Zone classifier (3-way softmax)       : Left / Top / Right
  • 3 × conditional regression heads      : one (mu, sigma) per zone
  • Curvilinear-distance head             : predict d ∈ [0, 1] via Sigmoid

Prediction (at inference)
  Final position = zone-probability-weighted mixture of the 3 mu heads
  (law of total expectation / total variance)

Data augmentation (training only)
  • Spike dropout   : randomly mask a fraction of spikes
  • Waveform noise  : add Gaussian noise to raw waveforms
"""
import math

import torch
import torch.nn as nn

from src import config as cfg


# ── Waveform encoder ──────────────────────────────────────────────────────────

class SpikeEncoder(nn.Module):
    """1-D CNN that encodes a multi-channel waveform into a fixed-size embedding.

    Input  : (N, C, 32)  — N spikes, C channels, 32 waveform samples
    Output : (N, embed_dim)
    """

    def __init__(self, n_channels: int, embed_dim: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, embed_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x):
        return self.conv(x).squeeze(-1)


# ── Positional encoding ───────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (Vaswani et al., 2017)."""

    def __init__(self, embed_dim: int, max_len: int = 256):
        super().__init__()
        pe       = torch.zeros(max_len, embed_dim)
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float()
            * (-math.log(10_000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, embed_dim)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


# ── Main model ────────────────────────────────────────────────────────────────

class SpikeTransformerHierarchical(nn.Module):
    """Hierarchical Transformer: zone classification + conditional regression."""

    def __init__(
        self,
        nGroups: int,
        nChannelsPerGroup: list,
        n_zones: int       = cfg.N_ZONES,
        embed_dim: int     = cfg.EMBED_DIM,
        nhead: int         = cfg.NHEAD,
        num_layers: int    = cfg.NUM_LAYERS,
        dropout: float     = cfg.DROPOUT,
        spike_dropout: float = cfg.SPIKE_DROPOUT,
        noise_std: float   = cfg.NOISE_STD,
        max_channels: int  = cfg.MAX_CHANNELS,
    ):
        super().__init__()
        self.nGroups       = nGroups
        self.embed_dim     = embed_dim
        self.n_zones       = n_zones
        self.spike_dropout = spike_dropout
        self.noise_std     = noise_std

        # --- Per-shank waveform encoders ---
        self.spike_encoders  = nn.ModuleList(
            [SpikeEncoder(max_channels, embed_dim) for _ in range(nGroups)]
        )
        self.shank_embedding = nn.Embedding(nGroups, embed_dim)
        self.pos_encoding    = PositionalEncoding(embed_dim)

        # --- Transformer backbone ---
        encoder_layer    = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )

        # --- Zone classification head ---
        self.cls_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(embed_dim, n_zones),
        )

        # --- Per-zone position regression heads (mean and log-sigma) ---
        self.mu_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(embed_dim, 2),
            )
            for _ in range(n_zones)
        ])
        self.log_sigma_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(embed_dim, 2),
            )
            for _ in range(n_zones)
        ])

        # --- Curvilinear distance regression head ---
        self.d_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(embed_dim, 1), nn.Sigmoid(),
        )

    # ── Data augmentation ─────────────────────────────────────────────────────

    def _apply_spike_dropout(self, mask: torch.Tensor) -> torch.Tensor:
        """Randomly mask additional spikes during training."""
        if not self.training or self.spike_dropout <= 0:
            return mask
        drop      = torch.rand_like(mask.float()) < self.spike_dropout
        active    = ~mask
        new_drops = drop & active
        # Guarantee at least one real spike per sequence
        all_gone  = (active & ~new_drops).sum(dim=1) == 0
        if all_gone.any():
            new_drops[all_gone] = False
        return mask | new_drops

    def _apply_waveform_noise(self, waveforms: torch.Tensor) -> torch.Tensor:
        if not self.training or self.noise_std <= 0:
            return waveforms
        return waveforms + torch.randn_like(waveforms) * self.noise_std

    # ── Shared encoder ────────────────────────────────────────────────────────

    def _encode(self, waveforms, shank_ids, mask):
        """CNN encode → positional + shank embedding → Transformer → mean pool."""
        B, T = waveforms.shape[:2]
        mask      = self._apply_spike_dropout(mask)
        waveforms = self._apply_waveform_noise(waveforms)

        embeddings = torch.zeros(B, T, self.embed_dim, device=waveforms.device)
        for g in range(self.nGroups):
            grp_mask = (shank_ids == g) & (~mask)
            if grp_mask.any():
                embeddings[grp_mask] = self.spike_encoders[g](waveforms[grp_mask])

        embeddings = embeddings + self.shank_embedding(shank_ids)
        embeddings = self.pos_encoding(embeddings)
        encoded    = self.transformer(embeddings, src_key_padding_mask=mask)

        # Masked mean pooling (ignore padding tokens)
        active = (~mask).unsqueeze(-1).float()
        pooled = (encoded * active).sum(dim=1) / (active.sum(dim=1) + 1e-8)
        return pooled

    # ── Forward / predict ─────────────────────────────────────────────────────

    def forward(self, waveforms, shank_ids, mask):
        """Raw forward pass returning all head outputs (used during training)."""
        pooled     = self._encode(waveforms, shank_ids, mask)
        cls_logits = self.cls_head(pooled)
        mus        = [h(pooled)              for h in self.mu_heads]
        sigmas     = [torch.exp(h(pooled))   for h in self.log_sigma_heads]
        d_pred     = self.d_head(pooled)
        return cls_logits, mus, sigmas, d_pred

    def predict(self, waveforms, shank_ids, mask):
        """Combined prediction via zone-probability-weighted Gaussian mixture.

        Uses the law of total expectation and total variance to aggregate
        per-zone mu/sigma into a single mu and sigma.

        Returns
        -------
        mu    : (B, 2) — predicted (x, y)
        sigma : (B, 2) — predicted uncertainty
        probs : (B, 3) — zone softmax probabilities
        d_pred: (B, 1) — predicted curvilinear distance
        """
        cls_logits, mus, sigmas, d_pred = self.forward(waveforms, shank_ids, mask)
        probs = torch.softmax(cls_logits, dim=1)          # (B, 3)

        mu_stack    = torch.stack(mus,    dim=1)          # (B, 3, 2)
        sigma_stack = torch.stack(sigmas, dim=1)          # (B, 3, 2)
        p           = probs.unsqueeze(-1)                 # (B, 3, 1)

        mu  = (p * mu_stack).sum(dim=1)                   # law of total expectation
        var = (p * (sigma_stack ** 2 + mu_stack ** 2)).sum(dim=1) - mu ** 2
        sigma = torch.sqrt(var.clamp(min=1e-8))

        return mu, sigma, probs, d_pred


# ── Factory ───────────────────────────────────────────────────────────────────

def build_model(nGroups: int, nChannelsPerGroup: list) -> SpikeTransformerHierarchical:
    """Instantiate a model with the default config hyperparameters."""
    model = SpikeTransformerHierarchical(nGroups, nChannelsPerGroup)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters  "
          f"({cfg.N_ZONES} zones, {cfg.NUM_LAYERS} transformer layers, "
          f"embed_dim={cfg.EMBED_DIM})")
    return model

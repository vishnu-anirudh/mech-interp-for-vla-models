"""
Matryoshka Sparse Autoencoder (MSAE) for multi-scale feature extraction.
Addresses critique requirement: Fix fragmentation by preserving syntax while steering semantics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MatryoshkaSAE(nn.Module):
    """
    Matryoshka Sparse Autoencoder with nested structure.
    Coarse features (first k dims) encode high-level semantics.
    Fine features (remaining dims) encode syntax/grammar.

    This allows steering semantics without disrupting syntax.
    """

    def __init__(
        self,
        d_model: int,
        expansion_factor: int = 4,
        k: int = 16,
        matryoshka_levels: list | None = None,
    ):
        """
        Args:
            d_model: Hidden dimension of the input activations
            expansion_factor: Multiplier for latent dimension
            k: Number of active features to keep (top-k sparsity)
            matryoshka_levels: List of truncation indices for nested structure
        """
        if matryoshka_levels is None:
            matryoshka_levels = [64, 256, 1024, 4096]
        super().__init__()
        self.d_model = d_model
        self.d_latents = d_model * expansion_factor
        self.k = k
        self.matryoshka_levels = sorted(matryoshka_levels)

        # Ensure max level doesn't exceed d_latents
        self.matryoshka_levels = [m for m in self.matryoshka_levels if m <= self.d_latents]
        if not self.matryoshka_levels:
            self.matryoshka_levels = [self.d_latents]

        self.encoder = nn.Linear(d_model, self.d_latents)
        self.encoder_bias = nn.Parameter(torch.zeros(self.d_latents))

        # Decoder for each matryoshka level
        self.decoders = nn.ModuleList(
            [nn.Linear(level, d_model, bias=False) for level in self.matryoshka_levels]
        )

        # Initialize decoders
        for decoder in self.decoders:
            decoder.weight.data = F.normalize(decoder.weight.data, dim=0)

    def encode(self, x):
        """
        Encode input activations into sparse latent representation.

        Args:
            x: Input activations of shape [batch, d_model] or [batch, seq_len, d_model]

        Returns:
            Sparse latents with only top-k features active
        """
        pre_act = self.encoder(x) + self.encoder_bias
        topk_values, topk_indices = torch.topk(pre_act, self.k, dim=-1)
        sparse_latents = torch.zeros_like(pre_act)
        sparse_latents.scatter_(-1, topk_indices, F.relu(topk_values))
        return sparse_latents

    def decode(self, latents, level_idx: int | None = None):
        """
        Decode latents using specified matryoshka level.

        Args:
            latents: Sparse latent representation
            level_idx: Which matryoshka level to use (None = use finest, 0 = coarsest)

        Returns:
            Reconstructed activations
        """
        if level_idx is None:
            level_idx = len(self.matryoshka_levels) - 1

        # Ensure level_idx is valid
        level_idx = max(0, min(level_idx, len(self.matryoshka_levels) - 1))

        level = self.matryoshka_levels[level_idx]
        truncated_latents = latents[..., :level]
        decoder = self.decoders[level_idx]
        return decoder(truncated_latents)

    def forward(self, x, level_idx: int | None = None):
        """
        Forward pass: encode and decode.

        Args:
            x: Input activations
            level_idx: Which matryoshka level to use for decoding

        Returns:
            reconstructed: Reconstructed activations
            latents: Sparse latent representation
        """
        latents = self.encode(x)
        reconstructed = self.decode(latents, level_idx)
        return reconstructed, latents

    def compute_matryoshka_loss(self, x, latents):
        """
        Compute nested loss for all matryoshka levels.

        Args:
            x: Input activations
            latents: Encoded sparse latents

        Returns:
            Dictionary of losses for each level
        """
        losses = {}
        weights = [1.0, 0.5, 0.25, 0.125]  # Decreasing weights for finer levels

        for i, level in enumerate(self.matryoshka_levels):
            recon = self.decode(latents, i)
            mse = F.mse_loss(recon, x)
            weight = weights[i] if i < len(weights) else 0.1
            losses[f"level_{level}"] = weight * mse

        return losses

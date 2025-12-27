"""
Sparse Autoencoder (SAE) for robotics feature discovery.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RoboticsSAE(nn.Module):
    """
    Sparse Autoencoder for discovering interpretable features in vision-language model activations.

    Uses top-k sparsity to enforce sparse latent representations.
    """

    def __init__(self, d_model: int, expansion_factor: int = 4, k: int = 16):
        """
        Args:
            d_model: Hidden dimension of the input activations
            expansion_factor: Multiplier for latent dimension (d_latents = d_model * expansion_factor)
            k: Number of active features to keep (top-k sparsity)
        """
        super().__init__()
        self.d_model = d_model
        self.d_latents = d_model * expansion_factor
        self.k = k

        self.encoder = nn.Linear(d_model, self.d_latents)
        self.encoder_bias = nn.Parameter(torch.zeros(self.d_latents))
        self.decoder = nn.Linear(self.d_latents, d_model, bias=False)
        self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=0)

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

    def forward(self, x):
        """
        Forward pass: encode and decode.

        Returns:
            reconstructed: Reconstructed activations
            latents: Sparse latent representation
        """
        latents = self.encode(x)
        reconstructed = self.decoder(latents)
        return reconstructed, latents

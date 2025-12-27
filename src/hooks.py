"""
Hook functions for intercepting and modifying model activations.
"""

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from sae import RoboticsSAE


def create_data_hook(activations_buffer: list):
    """
    Create a forward hook to collect activations from transformer layers.

    Args:
        activations_buffer: List to append collected activations to

    Returns:
        Hook function that can be registered with register_forward_hook()
    """

    def data_hook(module, input, output):
        """
        Hook function to extract activations from layer outputs.
        Handles different output formats (tuple vs tensor) and shapes (2D vs 3D).
        """
        # Handle different output formats: tuple or tensor, 2D or 3D
        act_tensor = output[0] if isinstance(output, tuple) else output

        # Handle different tensor shapes and ensure consistent output shape [1, hidden_size]
        if act_tensor.dim() == 3:
            # 3D: [batch_size, seq_len, hidden_size] - take last token from last sequence position
            act = act_tensor[:, -1, :].detach().to(torch.float32).cpu()
        elif act_tensor.dim() == 2:
            # 2D: [batch_size, hidden_size] or [seq_len, hidden_size] - take last row
            act = act_tensor[-1:, :].detach().to(torch.float32).cpu()
        else:
            # Fallback: flatten to 2D and take last row
            act_flat = act_tensor.detach().to(torch.float32).cpu()
            if act_flat.dim() > 2:
                act_flat = act_flat.view(-1, act_flat.shape[-1])
            act = act_flat[-1:, :] if act_flat.shape[0] > 0 else act_flat

        # Ensure shape is [1, hidden_size] for consistent concatenation
        if act.dim() == 1:
            act = act.unsqueeze(0)
        elif act.shape[0] > 1:
            act = act[-1:, :]  # Take only the last one if multiple

        activations_buffer.append(act)

    return data_hook


def create_steering_hook(sae: RoboticsSAE, fragile_feat_idx: int):
    """
    Create a forward hook to steer model behavior by injecting activations.

    Args:
        sae: Trained sparse autoencoder
        fragile_feat_idx: Index of the feature to activate (e.g., fragility feature)

    Returns:
        Hook function that can be registered with register_forward_hook()
    """

    def steering_hook(module, input, output):
        """
        Hook function to modify hidden states by steering through SAE.
        Injects high activation into the specified feature dimension.
        """
        # Handle both tuple and tensor outputs
        hiddens = output[0] if isinstance(output, tuple) else output

        dtype_orig = hiddens.dtype
        original_shape = hiddens.shape

        with torch.no_grad():
            h_float = hiddens.to(torch.float32)

            # Handle different shapes: flatten if 3D, keep if 2D
            if h_float.dim() == 3:
                # 3D: [batch_size, seq_len, hidden_size] - flatten to [batch*seq, hidden_size]
                _batch_size, _seq_len, hidden_size = h_float.shape
                h_flat = h_float.view(-1, hidden_size)
            else:
                # 2D: [batch_size, hidden_size] or [seq_len, hidden_size]
                h_flat = h_float

            z = sae.encode(h_flat)

            # STEER: Inject high activation into the discovered feature
            # We use a high coefficient to make the effect visible with few samples
            z[..., fragile_feat_idx] = z[..., fragile_feat_idx] + 20.0

            rec = sae.decoder(z)

            # Reshape back to original shape
            if len(original_shape) == 3:
                h_steered = rec.view(original_shape).to(dtype_orig)
            else:
                h_steered = rec.to(dtype_orig)

        # Return in the same format as input
        if isinstance(output, tuple):
            return (h_steered, *output[1:])
        else:
            return h_steered

    return steering_hook

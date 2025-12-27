"""
Patch-based hook functions for extracting all visual patch tokens.

This module provides hooks that extract ALL visual patch tokens from transformer layers,
enabling patch-based training that scales datasets by 100× (4k images → 400k patch tokens).

For standard hooks (single token extraction), see hooks.py.
"""

import torch


def create_data_hook_patch_based(activations_buffer: list):
    """
    Create a forward hook to collect ALL visual patch tokens from transformer layers.
    This enables patch-based training (~81 patches per image instead of 1 token).

    Args:
        activations_buffer: List to append collected activations to

    Returns:
        Hook function that can be registered with register_forward_hook()
    """

    def data_hook(module, input, output):
        """
        Hook function to extract ALL patch tokens from layer outputs.
        Extracts all visual tokens (typically ~81 patches) instead of just the last token.
        """
        # Handle different output formats: tuple or tensor, 2D or 3D
        act_tensor = output[0] if isinstance(output, tuple) else output

        # Handle different tensor shapes
        if act_tensor.dim() == 3:
            # 3D: [batch_size, seq_len, hidden_size]
            # Extract ALL visual patch tokens (typically first ~81 tokens for vision)
            # For SmolVLM, visual tokens are typically the first tokens in the sequence
            # We extract all tokens except the last few (which are text tokens)
            _batch_size, seq_len, hidden_size = act_tensor.shape

            # Estimate visual tokens: typically first ~81 tokens for 384x384 images
            # Text tokens start after visual tokens
            # Conservative estimate: take first 100 tokens (covers visual + some overlap)
            num_visual_tokens = min(100, seq_len - 1)  # Leave last token for text

            # Extract all visual patch tokens
            visual_tokens = act_tensor[:, :num_visual_tokens, :]  # [batch, num_visual, hidden]

            # Flatten to [batch * num_visual, hidden] for training
            visual_tokens_flat = visual_tokens.reshape(-1, hidden_size)

            # Detach and move to CPU
            act = visual_tokens_flat.detach().to(torch.float32).cpu()

        elif act_tensor.dim() == 2:
            # 2D: [batch_size, hidden_size] or [seq_len, hidden_size]
            # If 2D, we still extract all rows (treat as sequence)
            act = act_tensor.detach().to(torch.float32).cpu()
        else:
            # Fallback: flatten to 2D
            act_flat = act_tensor.detach().to(torch.float32).cpu()
            if act_flat.dim() > 2:
                act_flat = act_flat.view(-1, act_flat.shape[-1])
            act = act_flat

        # Append all tokens (not just one)
        activations_buffer.append(act)

    return data_hook

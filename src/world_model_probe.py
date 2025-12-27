"""
World Model Probe for predicting future states (S_{t+1}).
Addresses critique requirement: Don't wait for Phase 3 - add world model analysis now.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


class WorldModelProbe:
    """
    Linear probe to predict future observation embeddings from current activations.
    Based on Molinari et al. (2025) methodology.
    """

    def __init__(self, d_model: int, embedding_dim: int):
        """
        Args:
            d_model: Hidden dimension of activations
            embedding_dim: Dimension of observation embeddings
        """
        self.d_model = d_model
        self.embedding_dim = embedding_dim
        self.probe = nn.Linear(d_model, embedding_dim)

    def train(
        self,
        activations: list[torch.Tensor],
        next_embeddings: list[torch.Tensor],
        num_epochs: int = 100,
        lr: float = 1e-3,
    ):
        """
        Train the probe to predict next observation embeddings.

        Args:
            activations: List of activation tensors at time t
            next_embeddings: List of observation embeddings at time t+1
            num_epochs: Number of training epochs
            lr: Learning rate

        Returns:
            Training history (losses, R^2 scores)
        """
        # Convert to tensors
        X = torch.stack(activations)  # [N, d_model]
        y = torch.stack(next_embeddings)  # [N, embedding_dim]

        optimizer = optim.Adam(self.probe.parameters(), lr=lr)

        losses = []
        r2_scores = []

        for epoch in tqdm(range(num_epochs), desc="Training world model probe"):
            optimizer.zero_grad()

            # Forward pass
            predictions = self.probe(X)
            loss = nn.MSELoss()(predictions, y)

            # Compute R^2 score
            ss_res = torch.sum((y - predictions) ** 2)
            ss_tot = torch.sum((y - y.mean(dim=0)) ** 2)
            r2 = 1 - (ss_res / (ss_tot + 1e-8))
            r2_scores.append(r2.item())

            # Backward pass
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            if epoch % 20 == 0:
                print(f"Epoch {epoch}: Loss={loss.item():.4f}, R²={r2.item():.4f}")

        return {
            "losses": losses,
            "r2_scores": r2_scores,
            "final_r2": r2_scores[-1] if r2_scores else 0.0,
        }

    def predict(self, activations: torch.Tensor) -> torch.Tensor:
        """
        Predict next observation embedding from current activations.

        Args:
            activations: Activation tensor at time t

        Returns:
            Predicted observation embedding at time t+1
        """
        with torch.no_grad():
            return self.probe(activations)

    def compute_hallucination_metric(self, predicted: torch.Tensor, actual: torch.Tensor) -> float:
        """
        Compute hallucination metric: distance between predicted and actual S_{t+1}.

        Args:
            predicted: Predicted observation embedding
            actual: Actual observation embedding

        Returns:
            L2 distance (hallucination score)
        """
        return torch.norm(predicted - actual).item()


def create_drop_probe_dataset(vla, env, num_samples: int = 100):
    """
    Create dataset for "Drop Probe" - predicting state after drop event.

    Args:
        vla: Vision-language model wrapper
        env: Robotics environment
        num_samples: Number of samples to collect

    Returns:
        Tuple of (activations_holding, embeddings_after_drop)
    """
    activations_holding = []
    embeddings_after_drop = []

    # Hook to capture activations
    activations_buffer = []
    from hooks import create_data_hook

    data_hook = create_data_hook(activations_buffer)

    target_layer_idx = min(16, len(vla.layers) - 2)
    hook_handle = vla.get_layer(target_layer_idx).register_forward_hook(data_hook)

    print("\n📸 Collecting data for Drop Probe...")
    for i in tqdm(range(num_samples)):
        is_fragile = i % 2 == 0
        env.reset()
        env.spawn_object_decorrelated(is_fragile, "red" if i % 4 < 2 else "blue")

        # State 1: Robot holding object
        img_holding = env.get_image()
        vla.forward_pass(img_holding, "Describe the object.")

        if activations_buffer:
            act_holding = activations_buffer[-1].to(vla.device)
            activations_holding.append(act_holding.squeeze(0))

        # Simulate drop: Remove object (or spawn broken version)
        # For now, we'll use a different image as proxy for "after drop"
        # In real implementation, would simulate actual drop physics
        env.reset()
        # Spawn "broken" version (or empty scene)
        env.get_image()

        # Get embedding of "after drop" state
        # This would ideally be the embedding of the broken object
        with torch.no_grad():
            # Use vision encoder to get embedding
            # Simplified: use forward pass and extract embedding
            # In practice, would extract from vision encoder directly
            pass  # Placeholder - would need access to vision encoder

    hook_handle.remove()

    return activations_holding, embeddings_after_drop

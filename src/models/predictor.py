import torch
import torch.nn as nn
import torch.nn.functional as F


class Predictor(nn.Module):
    """
    JEPA-style latent predictor.

    Purpose:
    - Transform visual embeddings into a semantic latent space
    - Serve as the primary trainable component for Phase 3
    - Maintain clean cosine geometry for event detection

    Design principles:
    - No language conditioning
    - No temporal memory
    - No shortcuts
    - Geometry over generation
    """

    def __init__(self, embed_dim: int = 256):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [B, embed_dim]

        Returns:
            Normalized semantic embedding [B, embed_dim]
        """
        s = self.mlp(x)

        # Normalize to keep cosine distance meaningful
        s = F.normalize(s, dim=-1)
        return s

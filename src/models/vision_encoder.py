import torch
import torch.nn as nn
import torch.nn.functional as F


class VisionEncoder(nn.Module):
    """
    Lightweight vision encoder for VL-JEPA-style latent systems.

    Purpose:
    - Convert raw video frames into normalized latent embeddings
    - Preserve semantic structure, not fine-grained visual detail
    - Serve as a replaceable backbone (Phase 3+)

    This encoder is intentionally simple:
    - No pretrained weights
    - No temporal modeling
    - No language conditioning

    Architecture > encoder choice (for this project).
    """

    def __init__(self, embed_dim: int = 256):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.projection = nn.Linear(64, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [B, 3, H, W]

        Returns:
            Normalized embedding tensor of shape [B, embed_dim]
        """
        features = self.cnn(x)
        features = features.view(features.size(0), -1)
        embeddings = self.projection(features)

        # Normalize for cosine similarity geometry
        embeddings = F.normalize(embeddings, dim=-1)
        return embeddings

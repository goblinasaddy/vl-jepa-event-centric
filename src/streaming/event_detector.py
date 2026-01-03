import torch
import torch.nn.functional as F


class SemanticChangeDetector:
    """
    Stateful semantic change detector operating in latent space.

    Purpose:
    - Monitor a stream of latent embeddings over time
    - Detect meaningful semantic changes using cosine distance
    - Trigger events sparsely and conservatively

    This module is:
    - NOT trainable
    - NOT tuned per video
    - SHARED across Phase 1, 2, and 3

    Any change here invalidates baseline comparisons.
    """

    def __init__(self, threshold: float = 0.15):
        """
        Args:
            threshold: Cosine distance threshold for triggering an event.
                       Higher = more conservative (fewer events).
        """
        self.threshold = threshold
        self.prev_embedding = None

    def reset(self):
        """
        Reset internal state.
        Call this before starting a new video stream.
        """
        self.prev_embedding = None

    def step(self, embedding: torch.Tensor) -> bool:
        """
        Process one timestep.

        Args:
            embedding: Tensor of shape [D] (already normalized)

        Returns:
            True if a semantic event is detected, else False
        """
        # First frame always initializes the stream
        if self.prev_embedding is None:
            self.prev_embedding = embedding
            return True

        # Cosine distance as semantic change signal
        distance = 1.0 - F.cosine_similarity(
            embedding.unsqueeze(0),
            self.prev_embedding.unsqueeze(0)
        ).item()

        # Update state
        self.prev_embedding = embedding

        # Event decision
        return distance > self.threshold

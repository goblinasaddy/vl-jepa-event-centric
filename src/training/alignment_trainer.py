import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def alignment_loss(
    video_emb: torch.Tensor,
    text_emb: torch.Tensor,
    labels: torch.Tensor
) -> torch.Tensor:
    """
    Compute alignment loss between video embeddings and narration embeddings.

    Args:
        video_emb: [B, D] normalized video embeddings
        text_emb:  [B, D] normalized text embeddings
        labels:    [B] (1 = positive, 0 = negative)

    Returns:
        Scalar loss tensor
    """
    similarity = F.cosine_similarity(video_emb, text_emb)

    # Positive samples: maximize similarity
    if (labels == 1).any():
        pos_loss = (1.0 - similarity[labels == 1]).mean()
    else:
        pos_loss = torch.tensor(0.0, device=video_emb.device)

    # Negative samples: minimize similarity
    if (labels == 0).any():
        neg_loss = similarity[labels == 0].mean()
    else:
        neg_loss = torch.tensor(0.0, device=video_emb.device)

    return pos_loss + neg_loss


class AlignmentTrainer:
    """
    Trainer for Phase 3 latent–narration alignment.

    This trainer:
    - Trains ONLY the Predictor by default
    - Keeps VisionEncoder frozen (optional unfreeze later)
    - Uses narrated moments as semantic anchors
    - Does NOT modify inference logic
    """

    def __init__(
        self,
        vision_encoder,
        predictor,
        dataset,
        collate_fn,
        device: str = "cpu",
        batch_size: int = 4,
        lr: float = 1e-3
    ):
        self.device = device

        self.vision_encoder = vision_encoder.to(device)
        self.predictor = predictor.to(device)

        # Freeze vision encoder by default
        for p in self.vision_encoder.parameters():
            p.requires_grad = False

        self.dataset = dataset
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn
        )

        self.optimizer = torch.optim.Adam(
            self.predictor.parameters(),
            lr=lr
        )

    def train(self, epochs: int = 5):
        """
        Run alignment training for a fixed number of epochs.
        """
        self.vision_encoder.train()
        self.predictor.train()

        for epoch in range(epochs):
            total_loss = 0.0

            for frames, text_embs, labels in self.dataloader:
                frames = frames.to(self.device)
                text_embs = text_embs.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()

                with torch.no_grad():
                    visual_emb = self.vision_encoder(frames)

                semantic_emb = self.predictor(visual_emb)

                loss = alignment_loss(
                    semantic_emb,
                    text_embs,
                    labels
                )

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            print(f"[Epoch {epoch+1}] Alignment Loss: {total_loss:.4f}")

    def unfreeze_vision_encoder(self):
        """
        Optional: unfreeze the vision encoder for fine-tuning.
        Use ONLY after initial alignment converges.
        """
        for p in self.vision_encoder.parameters():
            p.requires_grad = True

        self.optimizer = torch.optim.Adam(
            list(self.vision_encoder.parameters()) +
            list(self.predictor.parameters()),
            lr=1e-4
        )

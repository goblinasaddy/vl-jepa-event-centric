import os
import torch

from sentence_transformers import SentenceTransformer

from src.models.vision_encoder import VisionEncoder
from src.models.predictor import Predictor
from src.datasets.ego4d_narrated import Ego4DNarratedDataset, narrated_collate_fn
from src.training.alignment_trainer import AlignmentTrainer


def main():
    # -------------------------
    # Configuration (EDIT HERE)
    # -------------------------
    ANNOTATION_PATH = "data/ego4d/annotations/narrations_processed.json"
    VIDEO_DIR = "data/ego4d/videos"

    BATCH_SIZE = 4
    EPOCHS = 5
    LR = 1e-3

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # -------------------------
    # Sanity checks
    # -------------------------
    if not os.path.exists(ANNOTATION_PATH):
        raise FileNotFoundError(f"Annotation file not found: {ANNOTATION_PATH}")

    if not os.path.isdir(VIDEO_DIR):
        raise FileNotFoundError(f"Video directory not found: {VIDEO_DIR}")

    print(f"[INFO] Using device: {DEVICE}")

    # -------------------------
    # Frozen text encoder
    # -------------------------
    print("[INFO] Loading frozen text encoder...")
    text_encoder = SentenceTransformer("all-MiniLM-L6-v2")
    text_encoder.eval()

    # -------------------------
    # Dataset
    # -------------------------
    print("[INFO] Loading Ego4D narrated dataset...")
    dataset = Ego4DNarratedDataset(
        annotation_path=ANNOTATION_PATH,
        video_dir=VIDEO_DIR,
        text_encoder=text_encoder
    )

    print(f"[INFO] Number of narrated moments: {len(dataset)}")

    # -------------------------
    # Models
    # -------------------------
    vision_encoder = VisionEncoder(embed_dim=256)
    predictor = Predictor(embed_dim=256)

    # -------------------------
    # Trainer
    # -------------------------
    trainer = AlignmentTrainer(
        vision_encoder=vision_encoder,
        predictor=predictor,
        dataset=dataset,
        collate_fn=narrated_collate_fn,
        device=DEVICE,
        batch_size=BATCH_SIZE,
        lr=LR
    )

    # -------------------------
    # Training
    # -------------------------
    print("[INFO] Starting Phase 3 alignment training...")
    trainer.train(epochs=EPOCHS)

    # -------------------------
    # Save trained weights
    # -------------------------
    os.makedirs("results/checkpoints", exist_ok=True)

    torch.save(
        predictor.state_dict(),
        "results/checkpoints/predictor_phase3.pt"
    )

    print("[INFO] Training complete.")
    print("[INFO] Saved predictor to results/checkpoints/predictor_phase3.pt")


if __name__ == "__main__":
    main()

import json
import random
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import Dataset
import cv2


def load_frame_at_time(
    video_path: str,
    time_sec: float,
    resize: Tuple[int, int] = (224, 224)
) -> torch.Tensor:
    """
    Load a single frame from a video at a specific timestamp.

    Args:
        video_path: Path to video file
        time_sec: Timestamp in seconds
        resize: Spatial size (H, W)

    Returns:
        Frame tensor [3, H, W] normalized to [0, 1], or None if failed
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_id = int(time_sec * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return None

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, resize)
    frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0

    return frame


class Ego4DNarratedDataset(Dataset):
    """
    Ego4D Narrated Moments Dataset.

    Each dataset item corresponds to ONE narrated moment and returns
    multiple samples:
      - Positive frames near the narration timestamp
      - Negative frames far from the narration timestamp

    Returned samples are used for latent alignment training.
    """

    def __init__(
        self,
        annotation_path: str,
        video_dir: str,
        text_encoder,
        pos_window: float = 2.0,
        neg_window: float = 5.0,
        frames_per_region: int = 2,
        resize: Tuple[int, int] = (224, 224)
    ):
        """
        Args:
            annotation_path: Path to processed narration JSON
            video_dir: Directory containing Ego4D videos
            text_encoder: Frozen text encoder (SentenceTransformer / CLIP)
            pos_window: ± seconds around narration for positives
            neg_window: minimum distance (sec) from narration for negatives
            frames_per_region: frames sampled per positive/negative region
            resize: Frame resize size
        """
        self.annotation_path = annotation_path
        self.video_dir = Path(video_dir)
        self.text_encoder = text_encoder
        self.pos_window = pos_window
        self.neg_window = neg_window
        self.frames_per_region = frames_per_region
        self.resize = resize

        with open(annotation_path, "r") as f:
            self.annotations = json.load(f)

        # Pre-encode narration texts (frozen semantic anchors)
        texts = [ann["narration_text"] for ann in self.annotations]
        with torch.no_grad():
            self.text_embeddings = self.text_encoder.encode(
                texts,
                convert_to_tensor=True,
                normalize_embeddings=True
            )

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int):
        ann = self.annotations[idx]

        video_uid = ann["video_uid"]
        timestamp = ann["timestamp_sec"]
        text_emb = self.text_embeddings[idx]

        video_path = self.video_dir / f"{video_uid}.mp4"

        samples = []

        # -------- POSITIVE SAMPLES --------
        for _ in range(self.frames_per_region):
            delta = random.uniform(-self.pos_window, self.pos_window)
            t = max(timestamp + delta, 0.0)

            frame = load_frame_at_time(
                str(video_path), t, resize=self.resize
            )

            if frame is not None:
                samples.append((frame, text_emb, 1))

        # -------- NEGATIVE SAMPLES --------
        for _ in range(self.frames_per_region):
            delta = random.choice([
                random.uniform(-30.0, -self.neg_window),
                random.uniform(self.neg_window, 30.0)
            ])
            t = max(timestamp + delta, 0.0)

            frame = load_frame_at_time(
                str(video_path), t, resize=self.resize
            )

            if frame is not None:
                samples.append((frame, text_emb, 0))

        return samples


def narrated_collate_fn(batch: List[List[Tuple[torch.Tensor, torch.Tensor, int]]]):
    """
    Collate function for Ego4DNarratedDataset.

    Flattens samples from multiple narrations into a single batch.
    """
    frames = []
    text_embs = []
    labels = []

    for sample_list in batch:
        for frame, text_emb, label in sample_list:
            frames.append(frame)
            text_embs.append(text_emb)
            labels.append(label)

    frames = torch.stack(frames)
    text_embs = torch.stack(text_embs)
    labels = torch.tensor(labels, dtype=torch.long)

    return frames, text_embs, labels

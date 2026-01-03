import cv2
import torch


def stream_video_frames(
    video_path: str,
    fps: int = 1,
    resize: tuple = (224, 224)
):
    """
    Stream frames from a video at a fixed sampling rate.

    Purpose:
    - Provide a deterministic temporal stream for latent processing
    - Decouple video I/O from semantic logic
    - Ensure identical sampling across all experiments

    Args:
        video_path: Path to video file
        fps: Target frames per second to sample
        resize: Spatial size (H, W) for frames

    Yields:
        Tensor of shape [1, 3, H, W] normalized to [0, 1]
    """

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        cap.release()
        raise ValueError("Invalid FPS reported by video file.")

    stride = max(int(video_fps // fps), 1)

    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % stride == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, resize)

            frame = torch.from_numpy(frame)
            frame = frame.permute(2, 0, 1).float() / 255.0
            yield frame.unsqueeze(0)

        frame_idx += 1

    cap.release()

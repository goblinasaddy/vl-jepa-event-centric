import csv
import json
from pathlib import Path

ANNOTATIONS_DIR = Path("data/epic_kitchens/annotations")
OUTPUT_PATH = ANNOTATIONS_DIR / "narrations_processed.json"


def parse_csv(csv_path):
    narrations = []

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            video_id = row["video_id"]
            start = float(row["start_timestamp"])
            end = float(row["stop_timestamp"])
            narration = row["narration"].strip().lower()

            # Use midpoint as semantic anchor
            timestamp = (start + end) / 2.0

            narrations.append({
                "video_uid": video_id,
                "timestamp_sec": timestamp,
                "narration_text": narration
            })

    return narrations


def main():
    all_narrations = []

    for csv_file in [
        ANNOTATIONS_DIR / "EPIC_100_train.csv",
        ANNOTATIONS_DIR / "EPIC_100_validation.csv"
    ]:
        if csv_file.exists():
            all_narrations.extend(parse_csv(csv_file))

    print(f"[INFO] Total narrations: {len(all_narrations)}")

    all_narrations = sorted(
        all_narrations,
        key=lambda x: (x["video_uid"], x["timestamp_sec"])
    )

    with open(OUTPUT_PATH, "w") as f:
        json.dump(all_narrations, f, indent=2)

    print(f"[INFO] Saved processed narrations to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

"""
Project scaffold generator for:
vl-jepa-event-centric

Usage:
    python template.py

This script creates the full folder + file structure
without overwriting existing files.
"""

from pathlib import Path

# Root directory (current working directory)
ROOT = Path(__file__).parent


# Folder + file structure definition
STRUCTURE = {
    "data": {
        "ego4d": {
            "videos": {},
            "annotations": {
                "narrations_raw.json": None,
                "narrations_processed.json": None,
            },
        },
        "mock": {
            "narrations.json": None,
        },
    },
    "notebooks": {
        "01_streaming_semantics.ipynb": None,
        "02_phase2_baselines.ipynb": None,
        "03_phase3_training.ipynb": None,
    },
    "src": {
        "__init__.py": None,
        "models": {
            "__init__.py": None,
            "vision_encoder.py": None,
            "predictor.py": None,
        },
        "streaming": {
            "__init__.py": None,
            "sampler.py": None,
            "event_detector.py": None,
        },
        "datasets": {
            "__init__.py": None,
            "ego4d_narrated.py": None,
        },
        "training": {
            "__init__.py": None,
            "alignment_trainer.py": None,
        },
        "evaluation": {
            "__init__.py": None,
            "phase2_baselines.py": None,
            "phase3_metrics.py": None,
        },
        "utils": {
            "__init__.py": None,
            "video.py": None,
            "plotting.py": None,
        },
    },
    "experiments": {
        "run_phase1_streaming.py": None,
        "run_phase2_baselines.py": None,
        "run_phase3_training.py": None,
    },
    "results": {
        "figures": {
            "phase1_distance_plot.png": None,
            "phase2_event_timeline.png": None,
            "phase3_alignment_plot.png": None,
        },
        "logs": {
            "training.log": None,
        },
    },
    "README.md": None,
    "requirements.txt": None,
    ".gitignore": None,
}


def create_structure(base_path: Path, tree: dict):
    """
    Recursively create folders and files.
    """
    for name, content in tree.items():
        path = base_path / name

        # Directory
        if isinstance(content, dict):
            path.mkdir(parents=True, exist_ok=True)
            create_structure(path, content)

        # File
        else:
            if not path.exists():
                path.touch()


def main():
    print("📁 Creating VL-JEPA project structure...")
    create_structure(ROOT, STRUCTURE)
    print("✅ Project structure created successfully.")
    print("⚠️ Existing files were NOT overwritten.")


if __name__ == "__main__":
    main()

"""Shared settings for the RGB + thermal detection project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
IMAGES = DATA / "images"
LABELS = DATA / "labels"
LISTS = DATA / "lists"
RUNS = ROOT / "runs"
WEIGHTS = ROOT / "weights"

# The three classes the challenge asks for.
# armoured_vehicle has no open data yet: the slot is ready, the model
# simply never sees it until labelled images are added (see README, step 6).
CLASSES = ["person", "vehicle", "armoured_vehicle"]

MODALITIES = ["rgb", "thermal"]

# Source dataset: M3FD, 4-channel TIFF (colour + thermal), YOLO labels.
HF_REPO = "Frencis/M3FD_RGBT"
# M3FD class ids -> our class ids. 'lamp' (4) is dropped.
M3FD_TO_OURS = {0: 0, 1: 1, 2: 1, 3: 1, 5: 1}

# COCO class ids -> our class ids, for scoring the stock model.
COCO_TO_OURS = {0: 0, 2: 1, 3: 1, 5: 1, 7: 1}


def modality_of(path) -> str:
    """Image files are named <id>_rgb.jpg or <id>_thermal.jpg."""
    stem = Path(path).stem
    return "thermal" if stem.endswith("_thermal") else "rgb"


def label_path(image_path) -> Path:
    p = Path(image_path)
    return LABELS / p.parent.name / (p.stem + ".txt")

"""Step 3: fine-tune the COCO-trained YOLOv8n on the balanced RGB + thermal mix.

    python scripts/04_train.py --epochs 50                 # on a GPU (Colab)
    python scripts/04_train.py --epochs 8 --imgsz 512      # quick CPU run

The result lands in runs/shared/weights/best.pt
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rgbt.common import LISTS, RUNS, WEIGHTS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(WEIGHTS / "yolov8n.pt"))
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--name", default="shared")
    ap.add_argument("--device", default=None, help="0 for the first GPU, cpu, or mps on a Mac")
    args = ap.parse_args()

    if not (LISTS / "mixed.yaml").exists():
        sys.exit("run scripts/02_make_lists.py first")

    from ultralytics import YOLO
    model = YOLO(args.weights)
    model.train(
        data=str(LISTS / "mixed.yaml"),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(RUNS),
        name=args.name,
        exist_ok=True,
        seed=0,
        # Colour tricks mean nothing on a thermal image, so keep them mild.
        hsv_h=0.0,
        hsv_s=0.3,
        hsv_v=0.4,
        plots=False,
    )


if __name__ == "__main__":
    main()

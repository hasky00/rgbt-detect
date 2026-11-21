"""Steps 1 and 4: score a model separately on RGB and on thermal.

    python scripts/03_evaluate.py                       # stock COCO YOLOv8n (the "before")
    python scripts/03_evaluate.py --weights runs/shared/weights/best.pt --name shared

Every run is saved to runs/scores.json and the comparison table is rewritten
to runs/results.md.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rgbt.common import COCO_TO_OURS, LISTS, MODALITIES, RUNS, WEIGHTS  # noqa: E402
from rgbt.scoring import score  # noqa: E402


def table(scores):
    rows = ["| model | RGB mAP50 | thermal mAP50 | gap | RGB mAP50-95 | thermal mAP50-95 |",
            "|---|---|---|---|---|---|"]
    for name, s in scores.items():
        r, t = s["rgb"], s["thermal"]
        rows.append(f"| {name} | {r['mAP50']:.3f} | {t['mAP50']:.3f} | "
                    f"{abs(r['mAP50'] - t['mAP50']):.3f} | {r['mAP50-95']:.3f} | {t['mAP50-95']:.3f} |")
    return "\n".join(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(WEIGHTS / "yolov8n.pt"))
    ap.add_argument("--name", default="baseline (COCO)")
    ap.add_argument("--coco", action="store_true",
                    help="model uses COCO class ids (automatic for the default weights)")
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.weights)
    is_coco = args.coco or len(model.names) == 80
    cmap = COCO_TO_OURS if is_coco else None

    result = {}
    for m in MODALITIES:
        result[m] = score(model, LISTS / f"val_{m}.txt", class_map=cmap, imgsz=args.imgsz)
        s = result[m]
        per = ", ".join(f"{k} {v:.3f}" for k, v in s["per_class"].items())
        print(f"{args.name:20s} {m:8s} mAP50 {s['mAP50']:.3f}  mAP50-95 {s['mAP50-95']:.3f}  ({per})")

    RUNS.mkdir(exist_ok=True)
    f = RUNS / "scores.json"
    scores = json.loads(f.read_text()) if f.exists() else {}
    scores[args.name] = result
    f.write_text(json.dumps(scores, indent=2))
    (RUNS / "results.md").write_text(table(scores) + "\n")
    print("\n" + table(scores))


if __name__ == "__main__":
    main()

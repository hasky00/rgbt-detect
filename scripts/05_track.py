"""Step 5: detect and track on a video, RGB or thermal.

    python scripts/05_track.py clip.mp4
    python scripts/05_track.py clip.mp4 --weights runs/shared/weights/best.pt --out tracked.mp4

Writes an annotated video plus a CSV with one row per object per frame:
frame, track id, class, confidence, box. ByteTrack keeps each object's id
stable from frame to frame.
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rgbt.common import RUNS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--weights", default=str(RUNS / "shared" / "weights" / "best.pt"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.weights)
    src = Path(args.video)
    out = Path(args.out) if args.out else src.with_name(src.stem + "_tracked.mp4")

    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    cap.release()

    writer, rows, ids = None, [], set()
    stream = model.track(source=str(src), tracker="bytetrack.yaml", conf=args.conf,
                         imgsz=args.imgsz, stream=True, verbose=False, persist=True)
    for n, r in enumerate(stream):
        frame = r.plot()
        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        writer.write(frame)
        if r.boxes.id is None:
            continue
        for tid, cls, cf, box in zip(r.boxes.id.int().tolist(), r.boxes.cls.int().tolist(),
                                     r.boxes.conf.tolist(), r.boxes.xyxy.tolist()):
            ids.add(tid)
            rows.append([n, tid, model.names[cls], round(cf, 3)] + [round(v, 1) for v in box])
    if writer is not None:
        writer.release()

    csv_path = out.with_suffix(".csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "track_id", "class", "conf", "x1", "y1", "x2", "y2"])
        w.writerows(rows)
    print(f"{n + 1} frames, {len(ids)} tracked objects")
    print(f"video: {out}\ntracks: {csv_path}")


if __name__ == "__main__":
    main()

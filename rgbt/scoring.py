"""One scoring path for every model, so the numbers are comparable.

The stock COCO model and the fine-tuned model speak different class ids.
Both are translated to our classes first, then scored with the same
mAP code (Ultralytics' own ap_per_class).
"""
from pathlib import Path

import numpy as np
import torch
from ultralytics.utils.metrics import ap_per_class, box_iou

from .common import CLASSES, label_path

IOUV = torch.linspace(0.5, 0.95, 10)


def load_truth(image_path, w, h):
    """YOLO label file -> (classes, boxes in pixels xyxy)."""
    p = label_path(image_path)
    rows = [l.split() for l in p.read_text().splitlines() if l.strip()] if p.exists() else []
    if not rows:
        return torch.zeros(0), torch.zeros(0, 4)
    a = torch.tensor([[float(v) for v in r] for r in rows])
    cx, cy, bw, bh = a[:, 1] * w, a[:, 2] * h, a[:, 3] * w, a[:, 4] * h
    return a[:, 0], torch.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)


def match(pred_boxes, pred_cls, true_boxes, true_cls):
    """True/false positive flags per prediction at 10 IoU thresholds."""
    correct = np.zeros((len(pred_boxes), len(IOUV)), dtype=bool)
    if not len(pred_boxes) or not len(true_boxes):
        return correct
    iou = box_iou(true_boxes, pred_boxes)
    iou = iou * (true_cls[:, None] == pred_cls[None, :])
    iou = iou.numpy()
    for i, thr in enumerate(IOUV.tolist()):
        gi, pi = np.nonzero(iou >= thr)
        if not len(gi):
            continue
        m = np.stack([gi, pi, iou[gi, pi]], 1)
        m = m[m[:, 2].argsort()[::-1]]
        m = m[np.unique(m[:, 1], return_index=True)[1]]
        # Predictions arrive sorted by confidence, so when two of them claim
        # the same object the more confident one keeps it (as Ultralytics does).
        m = m[np.unique(m[:, 0], return_index=True)[1]]
        correct[m[:, 1].astype(int), i] = True
    return correct


def score(model, image_list, class_map=None, imgsz=640, batch=16, device=None):
    """Score a model on a list of images.

    class_map: {model class id -> our class id}; predictions of any other
    class are dropped. None means the model already uses our ids.
    Returns {'mAP50', 'mAP50-95', 'images', 'per_class': {name: AP50}}.
    """
    paths = [l for l in Path(image_list).read_text().splitlines() if l.strip()]
    tp, conf, pcls, tcls = [], [], [], []
    for i in range(0, len(paths), batch):
        results = model.predict(paths[i:i + batch], imgsz=imgsz, conf=0.001, iou=0.7,
                                max_det=300, verbose=False, device=device)
        for path, r in zip(paths[i:i + batch], results):
            h, w = r.orig_shape
            t_cls, t_box = load_truth(path, w, h)
            boxes, cls, cf = r.boxes.xyxy.cpu(), r.boxes.cls.cpu(), r.boxes.conf.cpu()
            if class_map is not None:
                keep = torch.tensor([int(c) in class_map for c in cls], dtype=torch.bool)
                boxes, cf = boxes[keep], cf[keep]
                cls = torch.tensor([float(class_map[int(c)]) for c in cls[keep]])
            order = cf.argsort(descending=True)
            boxes, cls, cf = boxes[order], cls[order], cf[order]
            tp.append(match(boxes, cls, t_box, t_cls))
            conf.append(cf.numpy())
            pcls.append(cls.numpy())
            tcls.append(t_cls.numpy())

    tp, conf, pcls, tcls = (np.concatenate(x, 0) for x in (tp, conf, pcls, tcls))
    out = {"images": len(paths), "mAP50": 0.0, "mAP50-95": 0.0, "per_class": {}}
    if len(tp) and len(tcls):
        res = ap_per_class(tp, conf, pcls, tcls, plot=False, names={i: c for i, c in enumerate(CLASSES)})
        ap, classes = res[5], res[6]
        out["mAP50"] = float(ap[:, 0].mean())
        out["mAP50-95"] = float(ap.mean())
        out["per_class"] = {CLASSES[int(c)]: float(a[0]) for c, a in zip(classes, ap)}
    return out

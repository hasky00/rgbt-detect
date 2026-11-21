"""Build the image lists the other steps read.

    data/lists/train.txt        balanced training mix (same count per modality)
    data/lists/val_rgb.txt      validation, colour only
    data/lists/val_thermal.txt  validation, thermal only
    data/lists/val_all.txt      both, used while training
    data/lists/*.yaml           dataset files for Ultralytics

Balancing: if one camera type has fewer images, its images are repeated until
both sides are equal. That is the guard against "neglecting either modality".
Any image dropped into data/images/train/ named *_rgb.jpg or *_thermal.jpg
(with a label file of the same name in data/labels/train/) is picked up.
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rgbt.common import CLASSES, IMAGES, LISTS, MODALITIES, modality_of  # noqa: E402

EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def images(split):
    return sorted(p for p in (IMAGES / split).iterdir() if p.suffix.lower() in EXT)


def write_list(name, paths):
    (LISTS / f"{name}.txt").write_text("\n".join(str(p) for p in paths) + "\n")


def write_yaml(name, train, val):
    names = "\n".join(f"  {i}: {c}" for i, c in enumerate(CLASSES))
    (LISTS / f"{name}.yaml").write_text(
        f"train: {LISTS / (train + '.txt')}\nval: {LISTS / (val + '.txt')}\nnames:\n{names}\n")


def main():
    LISTS.mkdir(parents=True, exist_ok=True)
    random.seed(0)

    by_mod = {m: [p for p in images("train") if modality_of(p) == m] for m in MODALITIES}
    target = max(len(v) for v in by_mod.values())
    train = []
    for m, items in by_mod.items():
        if not items:
            sys.exit(f"no {m} training images found")
        extra = [random.choice(items) for _ in range(target - len(items))]
        print(f"train {m:8s} {len(items)} images" + (f" + {len(extra)} repeats" if extra else ""))
        train += items + extra
    random.shuffle(train)
    write_list("train", train)

    val = images("val")
    for m in MODALITIES:
        part = [p for p in val if modality_of(p) == m]
        print(f"val   {m:8s} {len(part)} images")
        write_list(f"val_{m}", part)
    write_list("val_all", val)

    write_yaml("mixed", "train", "val_all")
    write_yaml("rgb", "train", "val_rgb")
    write_yaml("thermal", "train", "val_thermal")
    print(f"lists written to {LISTS}")


if __name__ == "__main__":
    main()

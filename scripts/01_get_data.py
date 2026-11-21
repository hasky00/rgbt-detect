"""Step 2: download M3FD and split every scene into an RGB and a thermal image.

Each source file is one 4-channel TIFF: colour (3 channels) + thermal (1).
We write two JPEGs per scene with the same boxes, so the dataset is an exact
50/50 mix of the two camera types.

    python scripts/01_get_data.py --train 400 --val 150
    python scripts/01_get_data.py --train 0 --val 0      # 0 = everything
"""
import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rgbt.common import HF_REPO, IMAGES, LABELS, M3FD_TO_OURS, RAW  # noqa: E402


def pick_evenly(items, n):
    """Spread the picks over the whole set: neighbours are near-identical frames."""
    if n <= 0 or n >= len(items):
        return items
    idx = np.linspace(0, len(items) - 1, n).round().astype(int)
    return [items[i] for i in sorted(set(idx))]


def convert(split, stem, tiff_path, label_src):
    img = cv2.imread(str(tiff_path), cv2.IMREAD_UNCHANGED)
    if img is None or img.ndim != 3 or img.shape[2] != 4:
        return f"skip {stem}: not a 4-channel image"
    colour, thermal = img[..., :3], img[..., 3]
    out_i, out_l = IMAGES / split, LABELS / split
    cv2.imwrite(str(out_i / f"{stem}_rgb.jpg"), colour, [cv2.IMWRITE_JPEG_QUALITY, 92])
    cv2.imwrite(str(out_i / f"{stem}_thermal.jpg"), cv2.merge([thermal] * 3),
                [cv2.IMWRITE_JPEG_QUALITY, 92])

    rows = []
    for line in Path(label_src).read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cls = M3FD_TO_OURS.get(int(parts[0]))
        if cls is not None:
            rows.append(" ".join([str(cls)] + parts[1:]))
    text = "\n".join(rows) + ("\n" if rows else "")
    (out_l / f"{stem}_rgb.txt").write_text(text)
    (out_l / f"{stem}_thermal.txt").write_text(text)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=400, help="scenes for training, 0 = all")
    ap.add_argument("--val", type=int, default=150, help="scenes for validation, 0 = all")
    ap.add_argument("--keep-raw", action="store_true", help="keep the downloaded TIFFs")
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download, list_repo_files

    files = list_repo_files(HF_REPO, repo_type="dataset")
    for split, n in (("train", args.train), ("val", args.val)):
        stems = sorted(Path(f).stem for f in files if f.startswith(f"{split}/images/"))
        stems = pick_evenly(stems, n)
        (IMAGES / split).mkdir(parents=True, exist_ok=True)
        (LABELS / split).mkdir(parents=True, exist_ok=True)
        print(f"{split}: {len(stems)} scenes -> {2 * len(stems)} images")

        def one(stem, split=split):
            if (IMAGES / split / f"{stem}_thermal.jpg").exists():
                return None
            kw = dict(repo_id=HF_REPO, repo_type="dataset", local_dir=str(RAW))
            # The Hub rate-limits anonymous downloads (HTTP 429). Wait and try
            # again instead of letting one refused file stop the whole run.
            for attempt in range(8):
                try:
                    tiff = hf_hub_download(filename=f"{split}/images/{stem}.tiff", **kw)
                    lab = hf_hub_download(filename=f"{split}/labels/{stem}.txt", **kw)
                    break
                except Exception as e:  # noqa: BLE001
                    if attempt == 7:
                        return f"FAILED {stem}: {e}"
                    time.sleep(min(60, 5 * 2 ** attempt))
            msg = convert(split, stem, tiff, lab)
            if not args.keep_raw:
                Path(tiff).unlink(missing_ok=True)
            return msg

        failed = 0
        with ThreadPoolExecutor(8) as pool:
            for i, msg in enumerate(pool.map(one, stems), 1):
                if msg:
                    print(msg)
                    failed += msg.startswith("FAILED")
                if i % 50 == 0:
                    print(f"  {i}/{len(stems)}")
        got = len(list((IMAGES / split).glob("*_rgb.jpg")))
        print(f"{split}: {got} of {len(stems)} scenes on disk, {failed} failed")
        if failed:
            sys.exit("some files failed to download: run this script again, it skips what is already there")
    print("done")


if __name__ == "__main__":
    main()

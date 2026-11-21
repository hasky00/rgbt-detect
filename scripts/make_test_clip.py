"""Make a short fake clip by sliding a window across one still image.

Only for checking that 05_track.py runs. It is not real footage: nothing in
the scene moves, the camera just pans. Use a real video for a real test.

    python scripts/make_test_clip.py data/images/val/01104_thermal.jpg test_clip.mp4
"""
import sys

import cv2

src, out = sys.argv[1], sys.argv[2]
img = cv2.imread(src)
h, w = img.shape[:2]
win_w, frames = int(w * 0.7), 60
writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), 15, (win_w, h))
for i in range(frames):
    x = int((w - win_w) * i / (frames - 1))
    writer.write(img[:, x:x + win_w])
writer.release()
print(f"{frames} frames -> {out}")

# RGB + Thermal Detection and Tracking

Personal take on EDTH Paris 2026, challenge 07: one detector that works on
both colour and thermal images, with tracking on top.

One YOLOv8n model, fine-tuned on an exact 50/50 mix of colour and thermal
images, scored separately on each.

## First result

150 validation scenes from M3FD (each scene as one colour and one thermal
image), 400 scenes for training, 8 epochs on a 2-core CPU at 512 px.

| model | RGB mAP50 | thermal mAP50 | gap |
|---|---|---|---|
| stock YOLOv8n (COCO) | 0.585 | 0.478 | 0.107 |
| shared model, 8 epochs | 0.683 | 0.681 | 0.002 |

The stock model is clearly weaker on thermal. After the balanced fine-tune
both sides improve and the gap closes. This is a small, short run: expect
higher numbers from the full dataset and 50 epochs on a GPU.

## Run it on Colab

Open [train_colab.ipynb](https://colab.research.google.com/github/hasky00/rgbt-detect/blob/main/train_colab.ipynb), set the runtime to T4 GPU, and choose Run all. It downloads the full dataset, trains for 50 epochs and scores both camera types.

## Run it locally

```bash
pip install -r requirements.txt

python scripts/01_get_data.py --train 400 --val 150   # 0 0 = the full dataset
python scripts/02_make_lists.py
python scripts/03_evaluate.py                         # the "before" numbers
python scripts/04_train.py --epochs 50                # add --device 0 on a GPU, mps on a Mac
python scripts/03_evaluate.py --weights runs/shared/weights/best.pt --name shared
python scripts/05_track.py some_video.mp4
```

The comparison table is rewritten to `runs/results.md` after every
evaluation. The weights from the run above are in `weights/shared_8ep_cpu.pt`.

## What each file does

| file | job |
|---|---|
| `scripts/01_get_data.py` | downloads M3FD and splits every scene into `<id>_rgb.jpg` and `<id>_thermal.jpg` |
| `scripts/02_make_lists.py` | builds the balanced training list and the per-camera validation lists |
| `scripts/03_evaluate.py` | scores any model on RGB and thermal separately |
| `scripts/04_train.py` | fine-tunes the COCO model on the mix |
| `scripts/05_track.py` | ByteTrack on a video, writes an annotated video and a CSV of tracks |
| `rgbt/scoring.py` | the single scoring path used for every model |
| `rgbt/common.py` | classes, paths and class-id mappings |

## Classes

`person`, `vehicle`, `armoured_vehicle`.

M3FD has people, cars, buses, motorcycles and trucks, so the first two are
covered. The `armoured_vehicle` slot exists but has no training data yet.

## Adding more data (step 6)

Drop images into `data/images/train/` (or `val/`) named `something_rgb.jpg`
or `something_thermal.jpg`, with a YOLO label file of the same name in
`data/labels/train/`. Class ids: 0 person, 1 vehicle, 2 armoured_vehicle.
Then rerun `02_make_lists.py`. If one camera type ends up with fewer images,
they are repeated until both sides match.

## Known limits

- Trained and tested on M3FD only, which is ground-level street footage, not
  drone footage. Scores on aerial video will be lower until aerial data is added.
- Colour and thermal images of a scene share one set of boxes. An object
  that is invisible in one of the two still counts as a miss there.
- The tracker was only checked on a synthetic clip (a pan across a still
  image, `scripts/make_test_clip.py`). It needs real video for a real test.
- The 150 validation scenes are a sample, so treat differences under about
  0.02 mAP as noise.

## Data source

M3FD (Liu et al., CVPR 2022), fetched from the `Frencis/M3FD_RGBT` mirror on
Hugging Face, which lists it as GPL-3.0. Check the original licence terms
before any use beyond research.

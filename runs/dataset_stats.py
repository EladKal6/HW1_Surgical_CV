#!/usr/bin/env python3
"""
dataset_stats.py  –  Quick EDA for a YOLO-format dataset (hard-coded paths)

Outputs (saved in OUT_DIR):
  • boxes_per_image_hist.png
  • class_frequency_bar.png
  • image_sizes.csv
"""

import os, glob, csv, collections
import matplotlib.pyplot as plt
import cv2                     # pip install opencv-python

# ─── Hard-coded paths ────────────────────────────────────────────────────────
LABELS_DIR = "/home/student/runs/data/labels/train"
IMAGES_DIR = "/home/student/runs/data/images/train"    # set to None to skip image sizes
OUT_DIR    = "/home/student/runs/data/stats"

# ─── Gather stats from label files ───────────────────────────────────────────
label_files = glob.glob(os.path.join(LABELS_DIR, "*.txt"))
if not label_files:
    raise RuntimeError(f"No .txt files found in {LABELS_DIR}")

boxes_per_img = []
class_counter = collections.Counter()

for lf in label_files:
    with open(lf) as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    boxes_per_img.append(len(lines))
    for ln in lines:
        cls_id = ln.split()[0]          # first token = class id
        class_counter[cls_id] += 1

# ─── Create output folder ────────────────────────────────────────────────────
os.makedirs(OUT_DIR, exist_ok=True)

# ─── Plot: histogram of boxes per image ──────────────────────────────────────
plt.figure()
plt.hist(boxes_per_img,
         bins=range(0, max(boxes_per_img) + 2),
         rwidth=0.9)
plt.title("Boxes per Image")
plt.xlabel("# bounding boxes")
plt.ylabel("# images")
plt.savefig(os.path.join(OUT_DIR, "boxes_per_image_hist.png"), dpi=200)
plt.close()

# ─── Plot: class-frequency bar plot ──────────────────────────────────────────
classes = sorted(class_counter.keys(), key=int)   # ['0','1',...]
freqs   = [class_counter[c] for c in classes]

plt.figure()
plt.bar(classes, freqs)
plt.title("Class Frequency")
plt.xlabel("class id")
plt.ylabel("# bounding boxes")
plt.savefig(os.path.join(OUT_DIR, "class_frequency_bar.png"), dpi=200)
plt.close()

print("✔  Saved plots in:", OUT_DIR)

# ─── Optional: CSV of image sizes ────────────────────────────────────────────
if IMAGES_DIR is not None and os.path.isdir(IMAGES_DIR):
    rows = []
    for img_path in glob.glob(os.path.join(IMAGES_DIR, "*.*")):
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        rows.append((os.path.basename(img_path), w, h))

    rows.sort()
    csv_path = os.path.join(OUT_DIR, "image_sizes.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "width", "height"])
        writer.writerows(rows)
    print(f"✔  Wrote {len(rows)} image sizes to {csv_path}")
else:
    print("ℹ  IMAGES_DIR not provided or not found – skipping image-size CSV")

# ─── Console summary ────────────────────────────────────────────────────────
total_boxes = sum(freqs)
print("\nSummary")
print("-------")
print(f"Images parsed : {len(boxes_per_img)}")
print(f"Total boxes   : {total_boxes}")
for cls, n in zip(classes, freqs):
    pct = 100 * n / total_boxes
    print(f"  class {cls:>2}: {n:>4}  ({pct:5.1f} %)")

#!/usr/bin/env python3
"""
plot_yolo_metrics.py  –  Visualise YOLOv8 training logs.

Outputs
-------
loss_curve.png         –  train vs val total-loss per epoch
map_curve.png          –  train vs val mAP@0.5 per epoch
"""

import os
import pandas as pd          # pip install pandas
import matplotlib.pyplot as plt

# ─── Edit these three lines if your paths differ ──────────────────────────
RUN_DIR  = "/home/student/runs/augmented/runs/detect/train_with_tmp_yaml"
CSV_PATH     = os.path.join(RUN_DIR, "results.csv")
OUT_DIR      = os.path.join(RUN_DIR, "plots")
# -------------------------------------------------------------------------

os.makedirs(OUT_DIR, exist_ok=True)

# ─── Load CSV ─────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)

# Ultralytics columns look like:
#   epoch,      box_loss, cls_loss, dfl_loss,
#   metrics/precision, metrics/recall,
#   metrics/mAP50, metrics/mAP50-95,
#   val/box_loss, val/cls_loss, val/dfl_loss, ...
#
# We’ll sum the three loss components for train & val to get “total loss”.

train_total = df["box_loss"] + df["cls_loss"] + df["dfl_loss"]
val_total   = df["val/box_loss"] + df["val/cls_loss"] + df["val/dfl_loss"]
epochs      = df["epoch"]

# ─── Plot: loss ───────────────────────────────────────────────────────────
plt.figure()
plt.plot(epochs, train_total, label="train loss")
plt.plot(epochs, val_total,   label="val loss")
plt.xlabel("epoch")
plt.ylabel("total loss")
plt.title("YOLOv8 – Train vs Val Loss")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "loss_curve.png"), dpi=200)
plt.close()

# ─── Plot: mAP@0.5 (and 0.5‒0.95) ────────────────────────────────────────
plt.figure()
plt.plot(epochs, df["metrics/mAP50"],        label="train mAP@0.5")
plt.plot(epochs, df["val/metrics/mAP50"],    label="val mAP@0.5")
# Optional stricter metric – uncomment if you want both curves
# plt.plot(epochs, df["metrics/mAP50-95"],     label="train mAP@0.5-0.95", ls="--")
# plt.plot(epochs, df["val/metrics/mAP50-95"], label="val mAP@0.5-0.95",   ls="--")
plt.xlabel("epoch")
plt.ylabel("mAP")
plt.title("YOLOv8 – Train vs Val mAP")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "map_curve.png"), dpi=200)
plt.close()

print(f"✔  Plots saved in  {OUT_DIR}")

#!/usr/bin/env python3
"""
dataset_stats.py  –  Quick EDA for a YOLO-format dataset.

Generates:
  • boxes_per_image_hist.png      – histogram of how many boxes each image has
  • class_frequency_bar.png       – bar plot of total boxes per class
  • image_sizes.csv  (optional)   – filename,width,height for every image

Example:
    python dataset_stats.py \
        --labels_dir  data/labels/train \
        --images_dir  data/images/train \
        --out_dir     stats
"""

import os, glob, csv, collections, argparse
import matplotlib.pyplot as plt
import cv2                             # pip install opencv-python

# ──────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--labels_dir", required=True,
                   help="folder with YOLO *.txt label files")
    p.add_argument("--images_dir", default=None,
                   help="matching images folder (optional – for size CSV)")
    p.add_argument("--out_dir", default="stats",
                   help="where to save plots / CSV")
    return p.parse_args()

# ──────────────────────────────────────────────────────────────────────────────
def gather_stats(label_files):
    """Return list[#boxes per image] and Counter(class_id → count)."""
    boxes_per_img = []
    class_counter = collections.Counter()

    for lf in label_files:
        with open(lf) as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        boxes_per_img.append(len(lines))
        for ln in lines:
            cls_id = ln.split()[0]
            class_counter[cls_id] += 1
    return boxes_per_img, class_counter

# ──────────────────────────────────────────────────────────────────────────────
def save_image_sizes(images_dir, out_csv):
    rows = []
    for img_path in glob.glob(os.path.join(images_dir, "*.*")):
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        rows.append((os.path.basename(img_path), w, h))

    rows.sort()
    with open(out_csv, "w", newline="") as f:
        csv.writer(f).writerows([("filename", "width", "height"), *rows])
    print(f"✔  Wrote image sizes to: {out_csv}")

# ──────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    label_files = glob.glob(os.path.join(args.labels_dir, "*.txt"))
    if not label_files:
        raise RuntimeError(f"No .txt files found in {args.labels_dir}")

    boxes_per_img, class_counter = gather_stats(label_files)

    # ── Plot #boxes / image ──────────────────────────────────────────────────
    plt.figure()
    plt.hist(boxes_per_img,
             bins=range(0, max(boxes_per_img) + 2),
             rwidth=0.9)
    plt.title("Boxes per Image")
    plt.xlabel("# bounding boxes")
    plt.ylabel("# images")
    plt.savefig(os.path.join(args.out_dir, "boxes_per_image_hist.png"), dpi=200)
    plt.close()
    print("✔  Saved boxes-per-image histogram")

    # ── Plot class frequency ─────────────────────────────────────────────────
    classes = sorted(class_counter.keys(), key=int)   # ['0', '1', ...]
    freqs   = [class_counter[c] for c in classes]

    plt.figure()
    plt.bar(classes, freqs)
    plt.title("Class Frequency")
    plt.xlabel("class id")
    plt.ylabel("# bounding boxes")
    plt.savefig(os.path.join(args.out_dir, "class_frequency_bar.png"), dpi=200)
    plt.close()
    print("✔  Saved class-frequency bar plot")

    # ── Optional CSV of image sizes ──────────────────────────────────────────
    if args.images_dir and os.path.isdir(args.images_dir):
        out_csv = os.path.join(args.out_dir, "image_sizes.csv")
        save_image_sizes(args.images_dir, out_csv)
    else:
        print("ℹ  --images_dir not provided (skipping image-size CSV)")

    # ── Console summary ──────────────────────────────────────────────────────
    total_boxes = sum(freqs)
    print("\nSummary")
    print("-------")
    print(f"Images parsed : {len(boxes_per_img)}")
    print(f"Total boxes   : {total_boxes}")
    for cls, n in zip(classes, freqs):
        pct = 100 * n / total_boxes
        print(f"  class {cls:>2}: {n:>4}  ({pct:5.1f} %)")

if __name__ == "__main__":
    main()

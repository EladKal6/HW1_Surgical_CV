
#!/usr/bin/env python3
"""
preprocess.py

Create additional training examples by applying simple augmentations
to images and their YOLO‐format labels (x_center, y_center, w, h, class)
using Albumentations.

Directory structure assumed:
  data/
    images/        <-- original images (e.g. .jpg, .png)
    labels/        <-- original YOLO .txt label files (same basename as images)
  augmented/
    images/        <-- will be created, holds augmented images
    labels/        <-- will be created, holds augmented YOLO .txt labels
"""

import os
import cv2
import glob
import random
import shutil
from pathlib import Path

import albumentations as A

# ─── CONFIG ────────────────────────────────────────────────────────────────────

# Paths (adjust as needed)
ORIG_IMG_DIR   = "/home/student/runs/data/images/train"
ORIG_LABEL_DIR = "/home/student/runs/data/labels/train"
AUG_IMG_DIR    = "/home/student/runs/augmented/images/train"
AUG_LABEL_DIR  = "/home/student/runs/augmented/labels/train"

# How many augmented copies to create per original image
COPIES_PER_IMAGE = 3

# Define a set of augmentations to apply randomly
# using Albumentations. We choose a few simple ones: flips, rotations,
# brightness/contrast jitter, and random scaling.
AUGMENTATION_PIPELINE = A.Compose(
    [
        A.OneOf([
            A.HorizontalFlip(p=1.0),
            A.VerticalFlip(p=1.0),
            A.RandomRotate90(p=1.0),
        ], p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.05, scale_limit=0.1, rotate_limit=15, border_mode=cv2.BORDER_CONSTANT, p=0.5
        ),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
    ],
    bbox_params=A.BboxParams(
        format="yolo",           # input/output format: normalized [x_center, y_center, w, h]
        label_fields=["class_ids"],
        min_visibility=0.3,      # drop boxes that become too small / invisible
    ),
)

# ─── FUNCTIONS ─────────────────────────────────────────────────────────────────

def load_yolo_labels(label_path):
    """
    Read a YOLO‐style .txt file. Each line: class_id x_center y_center w h (all normalized).
    Returns two lists: bboxes (each [x_center, y_center, w, h]) and class_ids (int).
    """
    bboxes = []
    class_ids = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, xc, yc, w, h = parts
            class_ids.append(int(cls))
            bboxes.append([float(xc), float(yc), float(w), float(h)])
    return bboxes, class_ids

def save_yolo_labels(label_path, bboxes, class_ids):
    """
    Write out YOLO‐format labels to a .txt file.
    bboxes: list of [x_center, y_center, w, h] (all normalized floats)
    class_ids: list of ints
    """
    with open(label_path, "w") as f:
        for cls, box in zip(class_ids, bboxes):
            xc, yc, w, h = box
            f.write(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

def create_output_dirs():
    """Ensure the augmented directories exist, wiping old contents if present."""
    if os.path.isdir("augmented"):
        shutil.rmtree("augmented")
    os.makedirs(AUG_IMG_DIR, exist_ok=True)
    os.makedirs(AUG_LABEL_DIR, exist_ok=True)

def augment_one_image(img_path, label_path, out_prefix, copy_idx):
    """
    Load one image + its YOLO labels, apply augmentation pipeline,
    and write out augmented image + updated label file.
    """
    # Read image
    img = cv2.imread(img_path)
    height, width = img.shape[:2]

    # Load original YOLO labels
    bboxes, class_ids = load_yolo_labels(label_path)

    if len(bboxes) == 0:
        # If no box in original, skip augmentation
        return

    # Prepare Albumentations format: list of bboxes = [ [xc, yc, w, h] normalized ],
    # and class_ids must be passed as list of ints
    transformed = AUGMENTATION_PIPELINE(
        image=img,
        bboxes=bboxes,
        class_ids=class_ids,
    )
    img_aug = transformed["image"]
    bboxes_aug = transformed["bboxes"]
    class_ids_aug = transformed["class_ids"]

    # If augmentation dropped all bboxes (rare), skip saving this copy
    if len(bboxes_aug) == 0:
        return

    # Build new filenames
    base_fn = Path(out_prefix).stem
    new_img_name = f"{base_fn}_aug{copy_idx}.jpg"
    new_label_name = f"{base_fn}_aug{copy_idx}.txt"

    # Save augmented image
    cv2.imwrite(os.path.join(AUG_IMG_DIR, new_img_name), img_aug)

    # Save augmented labels (already normalized)
    save_yolo_labels(
        os.path.join(AUG_LABEL_DIR, new_label_name),
        bboxes_aug,
        class_ids_aug,
    )

def main():
    create_output_dirs()

    image_paths = sorted(glob.glob(os.path.join(ORIG_IMG_DIR, "*.*")))
    for img_path in image_paths:
        base = Path(img_path).stem
        label_path = os.path.join(ORIG_LABEL_DIR, f"{base}.txt")
        if not os.path.isfile(label_path):
            # Skip if there's no corresponding label
            continue

        # Copy original image/label into augmented folder as well
        shutil.copy(img_path, os.path.join(AUG_IMG_DIR, f"{base}.jpg"))
        shutil.copy(label_path, os.path.join(AUG_LABEL_DIR, f"{base}.txt"))

        # Create a few augmented variants
        for i in range(COPIES_PER_IMAGE):
            augment_one_image(img_path, label_path, base, i + 1)

    print("Augmentation complete.")
    print(f"Augmented images in: {AUG_IMG_DIR}")
    print(f"Augmented labels in: {AUG_LABEL_DIR}")

if __name__ == "__main__":
    main()

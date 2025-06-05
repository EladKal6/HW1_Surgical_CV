#!/usr/bin/env python3
"""
augment_split_train.py
────────────────────────────────────────────────────────────────────────────
* Assumes original data live in:   data/images/train   + data/labels/train
* Creates:                         augmented/images/{train,val}
                                   augmented/labels/{train,val}
* Uses Albumentations to produce 3 synthetic copies per original.
* Randomly puts 15 % of (orig + aug) into validation.
* Writes augmented/student_data.yaml with absolute paths.
* Fine-tunes YOLOv8-n on the dataset.
"""

import os
import cv2
import glob
import random
import shutil
from pathlib import Path

import albumentations as A
import yaml
from ultralytics import YOLO

# ─── CONFIG ──────────────────────────────────────────────────────────────────
# Original dataset -----------------------------------------------------------
ORIG_IMG_DIR   = Path("data/images/train")
ORIG_LABEL_DIR = Path("data/labels/train")

# Where to place the new augmented dataset -----------------------------------
AUG_ROOT       = Path("augmented").resolve()         # <--- main root
AUG_IMG_TRAIN  = AUG_ROOT / "images" / "train"
AUG_IMG_VAL    = AUG_ROOT / "images" / "val"
AUG_LBL_TRAIN  = AUG_ROOT / "labels" / "train"
AUG_LBL_VAL    = AUG_ROOT / "labels" / "val"

COPIES_PER_IMAGE = 3           # how many synthetic variants per original
VAL_RATIO        = 0.15        # fraction of (orig+aug) that becomes validation
RAND_SEED        = 0           # to reproduce split

# YOLO meta ------------------------------------------------------------------
NC    = 3
NAMES = ["class0", "class1", "class2"]               # <- EDIT to real class names
EPOCHS       = 50
BATCH_SIZE   = 16
IMG_SIZE     = 640
LR0          = 1e-3
BASE_WEIGHTS = "yolov8n.pt"

# Augmentation pipeline ------------------------------------------------------
AUG_PIPE = A.Compose(
    [
        A.OneOf([
            A.HorizontalFlip(p=1.0),
            A.VerticalFlip(p=1.0),
            A.RandomRotate90(p=1.0),
        ], p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.1,
            rotate_limit=15,
            border_mode=cv2.BORDER_CONSTANT,
            p=0.5,
        ),
        A.RandomBrightnessContrast(0.2, 0.2, p=0.5),
        A.ColorJitter(0.2, 0.2, 0.2, 0.1, p=0.3),
    ],
    bbox_params=A.BboxParams(format="yolo",
                             label_fields=["class_ids"],
                             min_visibility=0.3),
)

# ─── HELPERS ─────────────────────────────────────────────────────────────────
def load_yolo_labels(fp):
    bboxes, class_ids = [], []
    with open(fp, "r") as f:
        for line in f:
            cls, xc, yc, w, h = line.strip().split()
            bboxes.append([float(xc), float(yc), float(w), float(h)])
            class_ids.append(int(cls))
    return bboxes, class_ids

def save_yolo_labels(fp, bboxes, class_ids):
    with open(fp, "w") as f:
        for cls, box in zip(class_ids, bboxes):
            xc, yc, w, h = box
            f.write(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

def make_dirs():
    for p in [AUG_IMG_TRAIN, AUG_IMG_VAL, AUG_LBL_TRAIN, AUG_LBL_VAL]:
        p.mkdir(parents=True, exist_ok=True)

def augment_dataset():
    image_paths = sorted(glob.glob(str(ORIG_IMG_DIR / "*.*")))
    for img_path in image_paths:
        base = Path(img_path).stem
        label_path = ORIG_LABEL_DIR / f"{base}.txt"
        if not label_path.exists():
            continue

        # Copy original to augmented/train first
        shutil.copy(img_path, AUG_IMG_TRAIN / f"{base}.jpg")
        shutil.copy(label_path, AUG_LBL_TRAIN / f"{base}.txt")

        # Generate synthetic variants
        img = cv2.imread(img_path)
        bboxes, class_ids = load_yolo_labels(label_path)

        for i in range(1, COPIES_PER_IMAGE + 1):
            transformed = AUG_PIPE(image=img, bboxes=bboxes,
                                   class_ids=class_ids)
            if len(transformed["bboxes"]) == 0:
                continue
            new_img_name  = f"{base}_aug{i}.jpg"
            new_lbl_name  = f"{base}_aug{i}.txt"

            cv2.imwrite(str(AUG_IMG_TRAIN / new_img_name), transformed["image"])
            save_yolo_labels(AUG_LBL_TRAIN / new_lbl_name,
                             transformed["bboxes"], transformed["class_ids"])
    print("✓ Augmentation stage finished.")


def train_val_split(root=AUG_ROOT, val_ratio=0.15, seed=0):
    """
    Move a val_ratio subset of images (and matching labels) from
    augmented/images/train → augmented/images/val.

    Handles corner cases gracefully:
      • zero images              → skip split
      • very small dataset       → take exactly 1 image for val
    """
    random.seed(seed)

    train_img_dir = root / "images" / "train"
    train_lbl_dir = root / "labels" / "train"
    val_img_dir   = root / "images" / "val"
    val_lbl_dir   = root / "labels" / "val"

    all_imgs = list(train_img_dir.glob("*.*"))
    n_total  = len(all_imgs)
    if n_total == 0:
        print("⚠️  No images found in augmented/images/train – skipping split.")
        return

    k = max(1, int(n_total * val_ratio))
    k = min(k, n_total)        # never larger than the population

    val_subset = random.sample(all_imgs, k)

    for img_path in val_subset:
        lbl_path = train_lbl_dir / f"{img_path.stem}.txt"
        if lbl_path.exists():
            shutil.move(str(lbl_path), val_lbl_dir / lbl_path.name)
        shutil.move(str(img_path), val_img_dir / img_path.name)

    print(f"✓ Moved {len(val_subset)} of {n_total} images to validation set.")


def write_yaml():
    data = {
        "train": str(AUG_IMG_TRAIN),
        "val":   str(AUG_IMG_VAL),
        "nc":    NC,
        "names": NAMES,
    }
    yaml_path = AUG_ROOT / "student_data.yaml"
    yaml_path.write_text(yaml.dump(data, sort_keys=False))
    print(f"✓ Wrote YAML → {yaml_path}")
    return yaml_path

def train_yolo(yaml_path):
    project_dir = AUG_ROOT / "yolo_runs"
    model = YOLO(BASE_WEIGHTS)
    model.train(
        data     = str(yaml_path),
        epochs   = EPOCHS,
        imgsz    = IMG_SIZE,
        batch    = BATCH_SIZE,
        lr0      = LR0,
        project  = str(project_dir),
        name     = "finetune",
        cache    = True,
        exist_ok = True,
        device   = "cuda",
    )
    print("✅ Training done.")
# ─── MAIN ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    make_dirs()
    augment_dataset()
    train_val_split()
    yaml_fp = write_yaml()
    train_yolo(yaml_fp)

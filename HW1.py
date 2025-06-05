# train_write_tmp_yaml.py

import os
import yaml
from ultralytics import YOLO

if __name__ == "__main__":
    # ─────────────────────────────────────────────────────────────────────────────
    # 1) Root of your “augmented” folder (where images/ and labels/ live)
    ROOT = "/home/student/runs/augmented"

    VAL_IMAGES = "/home/student/runs/data/images/val"
    VAL_LABELS = "/home/student/runs/data/labels/val"  # not strictly needed in YAML; YOLO assumes label path from image path

    # 2) Paths to your train/val image folders
    TRAIN_IMAGES = os.path.join(ROOT, "images", "train")

    # 3) Number of classes and (placeholder) class names.
    #    Adjust 'names' to your actual label strings if you know them.
    NC    = 3
    NAMES = ["class0", "class1", "class2"]

    # ─────────────────────────────────────────────────────────────────────────────
    # 4) Build a Python dict exactly like a data.yaml:
    data_dict = {
        "train": TRAIN_IMAGES,
        "val": VAL_IMAGES,
        "nc": NC,
        "names": NAMES
    }

    # 5) Write that dict to a temporary data.yaml in the same folder:
    yaml_path = os.path.join(ROOT, "tmp_data.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(data_dict, f, sort_keys=False)

    # ─────────────────────────────────────────────────────────────────────────────
    # 6) Which pretrained YOLOv8 checkpoint to start from:
    BASE_WEIGHTS = "yolov8n.pt"   # or "yolov8s.pt", etc.

    # 7) Training hyperparameters
    EPOCHS     = 50
    BATCH_SIZE = 16
    IMG_SIZE   = 640
    LR0        = 1e-3

    # ─────────────────────────────────────────────────────────────────────────────
    # 8) Where to save run outputs
    PROJECT_DIR = os.path.join(ROOT, "runs", "detect")
    RUN_NAME    = "train_with_tmp_yaml"
    os.makedirs(os.path.join(PROJECT_DIR, RUN_NAME), exist_ok=True)

    # ─────────────────────────────────────────────────────────────────────────────
    # 9) Instantiate the model from the pretrained checkpoint
    model = YOLO(BASE_WEIGHTS)

    # ─────────────────────────────────────────────────────────────────────────────
    # 10) Launch training, pointing `data` at the newly written YAML file
    model.train(
        data       = yaml_path,      # now a real “data.yaml” file
        epochs     = EPOCHS,
        imgsz      = IMG_SIZE,
        batch      = BATCH_SIZE,
        lr0        = LR0,            # initial learning rate
        project    = PROJECT_DIR,
        name       = RUN_NAME,
        exist_ok   = True,
        cache      = True,
        device     = "cuda",         # or "cpu" if needed
    )

    # ─────────────────────────────────────────────────────────────────────────────
    # 11) After training finishes, best.pt lives here:
    best_path = os.path.join(PROJECT_DIR, RUN_NAME, "weights", "best.pt")
    print(f"\n✅ Training complete. Best weights saved to:\n    {best_path}\n")

    # ─────────────────────────────────────────────────────────────────────────────
    # 12) (Optional) Remove the temporary YAML if you like:
    # os.remove(yaml_path)

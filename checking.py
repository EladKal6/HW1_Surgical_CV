import os

# ─────────────────────────────────────────────────────────────────────────────
# 1. Change this to wherever your “labels/train” actually lives.
#    In your case it should be:
LABEL_DIR = "/home/student/runs/augmented"

# ─────────────────────────────────────────────────────────────────────────────
# 2. Gather all .txt files in that folder
txt_files = [
    f for f in os.listdir(LABEL_DIR)
    if f.lower().endswith(".txt")
]

unique_classes = set()

# ─────────────────────────────────────────────────────────────────────────────
# 3. Parse each .txt file, pulling out the first token on each non‐empty line
for fname in txt_files:
    path = os.path.join(LABEL_DIR, fname)
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # The format is: "<class_id> <x_center> <y_center> <w> <h>"
            cls_id = float(line.split()[0])
            unique_classes.add(cls_id)

# ─────────────────────────────────────────────────────────────────────────────
# 4. Print out the summary
if unique_classes:
    print("✔ Found the following class IDs:", sorted(unique_classes))
    print("  • Number of distinct classes:", len(unique_classes))
    print("  • Maximum class ID       :", max(unique_classes))
else:
    print("⚠ No label files or no class IDs found under:", LABEL_DIR)

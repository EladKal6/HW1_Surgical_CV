import os
import pandas as pd
import matplotlib.pyplot as plt

# Path to your metrics file
CSV_PATH = "/home/student/runs/augmented/runs/detect/train_with_tmp_yaml/results.csv"
SAVE_DIR = "/home/student/runs/plots"
os.makedirs(SAVE_DIR, exist_ok=True)

# Load the CSV
df = pd.read_csv(CSV_PATH)

def plot_metric(metric_name, ylabel, filename):
    if metric_name not in df.columns:
        print(f"⚠️ Skipping: {metric_name} not found in CSV.")
        return

    last_val = df[metric_name].iloc[-1]
    print(f"📊 {metric_name} final value at last epoch: {last_val:.4f}")

    plt.figure()
    plt.plot(df.index, df[metric_name], label=f"{metric_name} (last: {last_val:.4f})", marker='o')
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(metric_name)
    plt.grid(True)
    plt.legend()
    save_path = os.path.join(SAVE_DIR, filename)
    plt.savefig(save_path)
    print(f"✅ Saved: {save_path}")
    plt.close()


# Plot all desired metrics
plot_metric("train/box_loss", "Train Box Loss", "train_box_loss.png")
plot_metric("val/box_loss",   "Val Box Loss",   "val_box_loss.png")
plot_metric("metrics/mAP50(B)", "mAP@0.5", "mAP_0.5.png")
plot_metric("metrics/mAP50-95(B)", "mAP@[.5:.95]", "mAP_0.5_0.95.png")


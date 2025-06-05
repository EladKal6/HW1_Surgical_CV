# live_inference_headless.py

import os
import cv2
from ultralytics import YOLO

def main():
    # ─────────────────────────────────────────────────────────────────────────
    # 1) Path to your trained YOLOv8 weights
    weights_path = "/tmp/pycharm_project_401/augmented/runs/detect/train_with_tmp_yaml/weights/best.pt"
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"Could not find weights at {weights_path}")

    # 2) Path to the input video
    video_path = "/tmp/pycharm_project_401/data/surg_1.mp4"
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Could not find video at {video_path}")

    # 3) Load the YOLOv8 model (on GPU if available)
    model = YOLO(weights_path)

    # 4) OpenCV video capture & writer
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 5) Prepare output file to save annotated video
    output_dir  = os.path.join("runs", "detect", "live_video")
    os.makedirs(output_dir, exist_ok=True)
    print(output_dir)
    output_file = os.path.join(output_dir, "surg_1_annotated.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    print("▶️  Processing video (headless, writing to disk)…")

    # 6) Loop over frames
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 7) Run YOLO inference on this single frame
        results = model.predict(
            source = frame,
            conf   = 0.25,
            iou    = 0.45,
            device = "cuda"  # or "cpu" if no GPU available
        )

        # 8) Draw detections onto the frame
        annotated_frame = results[0].plot()

        # 9) Write annotated frame to output video
        writer.write(annotated_frame)

        frame_idx += 1
        if frame_idx % 50 == 0:
            # Print progress every 50 frames
            print(f"   • processed {frame_idx} frames…")

    # 10) Cleanup
    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"\n✅ Saved annotated video to:\n    {output_file}\n")

if __name__ == "__main__":
    main()

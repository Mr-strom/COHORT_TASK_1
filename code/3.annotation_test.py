import cv2
import os
from ultralytics import YOLO

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(base_dir, 'output', 'best_ncnn_model')
if not os.path.exists(model_path):
    model_path = os.path.join(base_dir, 'output', 'best.pt')

video_path = os.path.join(base_dir, 'code', 'test_video.mp4')
output_dir = os.path.join(base_dir, 'output')

model = YOLO(model_path, task='detect')
cap = cv2.VideoCapture(video_path)

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0

out_path = os.path.join(output_dir, 'annotated_full_video.mp4')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

print("Processing and annotating entire video...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    try:
        results = model.predict(frame, conf=0.10, verbose=False, device="vulkan:0")
    except Exception:
        results = model.predict(frame, conf=0.10, verbose=False)

    annotated_frame = results[0].plot()
    out.write(annotated_frame)

cap.release()
out.release()
print(f"Annotated video saved to: {out_path}")
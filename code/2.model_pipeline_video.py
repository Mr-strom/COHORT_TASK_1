import cv2
import os
from collections import deque
from ultralytics import YOLO

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(base_dir, 'output', 'best.pt')
ncnn_path = os.path.join(base_dir, 'output', 'best_ncnn_model')
video_path = os.path.join(base_dir, 'code', 'test_video.mp4')
output_dir = os.path.join(base_dir, 'output')

VIOLATION_CLASSES = ['no_helmet', 'no_boots', 'no_gloves', 'no_goggle']

def load_pipeline_model(weights_path=None):
    if weights_path is None:
        if os.path.exists(ncnn_path):
            try:
                return YOLO(ncnn_path, task='detect')
            except Exception:
                pass
        weights_path = model_path
    return YOLO(weights_path, task='detect')

def get_saved_violations(output_directory=output_dir):
    violations_dir = os.path.join(output_directory, 'violations')
    if not os.path.exists(violations_dir):
        return {}

    categorized = {}
    for item in os.listdir(violations_dir):
        item_path = os.path.join(violations_dir, item)
        if os.path.isdir(item_path):
            images = [
                os.path.join(item_path, f) for f in os.listdir(item_path)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ]
            if images:
                categorized[item] = sorted(images, key=os.path.getmtime, reverse=True)
    return categorized

def run_video_pipeline(video_source=video_path, model=None, conf_threshold=0.5, output_directory=output_dir, frame_callback=None, max_frames=None):
    if model is None:
        model = load_pipeline_model()

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Error: Could not open video source {video_source}")
        return {"total_frames": 0, "violations_detected": 0, "evidence_path": None, "snapshots": []}

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_skip = int(original_fps / 5) if original_fps and original_fps > 0 else 6

    ring_buffer = deque(maxlen=25)
    frame_count = 0
    violation_triggered = False
    post_violation_frames = 0
    recording_frames = []
    evidence_saved_path = None
    violation_events_count = 0
    captured_snapshots = []

    violations_dir = os.path.join(output_directory, 'violations')
    os.makedirs(violations_dir, exist_ok=True)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if max_frames and frame_count > max_frames:
            break

        if frame_count % frame_skip != 0:
            continue

        try:
            results = model.predict(frame, conf=conf_threshold, verbose=False)
        except Exception:
            results = model.predict(frame, conf=conf_threshold, verbose=False, device='cpu')

        annotated_frame = results[0].plot()
        ring_buffer.append(annotated_frame)

        current_frame_violations = []
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            if class_name in VIOLATION_CLASSES:
                current_frame_violations.append(class_name)

        violation_found_in_frame = len(current_frame_violations) > 0

        if violation_found_in_frame and not violation_triggered:
            print(f"Violation detected at frame {frame_count}! Triggering evidence capture.")
            violation_triggered = True
            violation_events_count += 1
            recording_frames = list(ring_buffer)
            post_violation_frames = 0

            # Capture snapshot for each removed/missing safety class
            for v_class in set(current_frame_violations):
                class_folder = os.path.join(violations_dir, v_class)
                os.makedirs(class_folder, exist_ok=True)
                snapshot_name = f"violation_frame_{frame_count}_{v_class}.jpg"
                snapshot_path = os.path.join(class_folder, snapshot_name)
                cv2.imwrite(snapshot_path, annotated_frame)
                captured_snapshots.append({
                    "class": v_class,
                    "frame": frame_count,
                    "path": snapshot_path,
                    "timestamp": round(frame_count / original_fps, 2) if original_fps else 0
                })

        if violation_triggered:
            if post_violation_frames < 25:
                if len(recording_frames) == 0 or not (annotated_frame is recording_frames[-1]):
                    recording_frames.append(annotated_frame)
                post_violation_frames += 1
            else:
                print("Saving 10-second evidence clip with bounding boxes...")
                out_path = os.path.join(output_directory, 'violation_evidence.mp4')
                height, width, _ = annotated_frame.shape
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(out_path, fourcc, 5.0, (width, height))
                for rec_frame in recording_frames:
                    out.write(rec_frame)
                out.release()
                print(f"Evidence saved to {out_path}")
                evidence_saved_path = out_path

                violation_triggered = False
                recording_frames = []
                post_violation_frames = 0

        if frame_callback is not None:
            stop_requested = frame_callback(annotated_frame, frame_count, current_frame_violations, violation_triggered, evidence_saved_path)
            if stop_requested:
                break

    cap.release()
    print("Video processing complete!")
    return {
        "total_frames": frame_count,
        "violations_detected": violation_events_count,
        "evidence_path": evidence_saved_path,
        "snapshots": captured_snapshots
    }

if __name__ == '__main__':
    run_video_pipeline(video_path)
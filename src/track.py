import os
import cv2
import json
from deep_sort_realtime.deepsort_tracker import DeepSort
from src.utils import load_model


def run_tracking(video_path, weights_path, visualize=True):
    # Load YOLOv11 model
    model = load_model(weights_path)

    # Load input video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None:
        print("FPS is not detected!")
        fps = 25

    # Video writer
    os.makedirs("outputs", exist_ok=True)
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    tracked_video_path = os.path.join("outputs", f"{base_name}_tracked.mp4")
    writer = cv2.VideoWriter(tracked_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height)) if visualize else None

    # Initialize DeepSORT
    deepsort = DeepSort(max_age=30)

    frame_id = 0
    all_tracks = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        print(f"[INFO] Processing frame {frame_id}")
        results = model.predict(source=frame, verbose=False)[0]
        detections = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]

            # Only track 'player'
            if cls_name != "player":
                continue

            conf = float(box.conf[0])
            if conf < 0.5:
                continue  # filter weak detections

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            # Skip detections too high in the frame 
            if y1 < 200:
                continue

            w, h = x2 - x1, y2 - y1
            detections.append(([x1, y1, w, h], conf, None))  # (x, y, w, h)

        # DeepSORT
        tracks = deepsort.update_tracks(detections, frame=frame)
        frame_tracks = []

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            l, t, r, b = map(int, track.to_ltrb())
            frame_tracks.append({
                "frame": frame_id,
                "track_id": track_id,
                "bbox": [l, t, r, b]
            })

            if visualize:
                cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)
                cv2.putText(frame, f"ID {track_id}", (l, t - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        all_tracks.append({
            "frame": frame_id,
            "tracks": frame_tracks
        })

        if visualize:
            writer.write(frame)
            cv2.imshow("Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_id += 1

    cap.release()
    if visualize:
        writer.release()
        cv2.destroyAllWindows()

    # Save
    log_path = os.path.join("outputs", f"{base_name}_track_logs.json")
    with open(log_path, "w") as f:
        json.dump(all_tracks, f, indent=2)

    print(f"Saved tracked video to {tracked_video_path}")
    print(f"Saved tracking logs to {log_path}")


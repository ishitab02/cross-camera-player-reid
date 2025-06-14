import json
import cv2
from ultralytics import YOLO

# Load YOLO model
def load_model(weights_path):
    return YOLO(weights_path)

# Save detections as JSON
def save_detections_json(detections, save_path):
    with open(save_path, 'w') as f:
        json.dump(detections, f, indent=2)

# Load video frames using OpenCV
def load_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames
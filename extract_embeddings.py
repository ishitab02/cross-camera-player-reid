import cv2
import torch
import os
import json
import numpy as np
from torchvision import models, transforms
from src.utils import load_model

# Load pretrained feature extractor (ResNet50)
resnet = models.resnet50(pretrained=True)
resnet.fc = torch.nn.Identity() 
resnet.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def extract_track_embeddings(video_path, track_log_path, output_path):
    cap = cv2.VideoCapture(video_path)
    with open(track_log_path, 'r') as f:
        logs = json.load(f)

    all_embeddings = {}

    for frame_data in logs:
        frame_id = frame_data['frame']
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        if not ret:
            continue

        for track in frame_data['tracks']:
            tid = track['track_id']
            l, t, r, b = track['bbox']
            crop = frame[t:b, l:r]

            if crop.size == 0:
                continue

            input_tensor = transform(crop).unsqueeze(0)
            with torch.no_grad():
                embedding = resnet(input_tensor).squeeze(0).numpy()

            all_embeddings.setdefault(tid, []).append(embedding)

    averaged = {
        tid: np.mean(embeds, axis=0).tolist()
        for tid, embeds in all_embeddings.items()}

    with open(output_path, 'w') as f:
        json.dump(averaged, f, indent=2)

    print(f"Saved track embeddings to {output_path}")

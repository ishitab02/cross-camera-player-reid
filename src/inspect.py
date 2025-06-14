import pandas as pd
import cv2
import json
import os

def load_tracks(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def get_crop(frame_path, bbox):
    img = cv2.imread(frame_path)
    x1, y1, x2, y2 = map(int, bbox)
    return img[y1:y2, x1:x2]

def visual_inspect_matches(broadcast_json, tacticam_json, video_dir, match_csv, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    
    matches = pd.read_csv(match_csv)
    broadcast_tracks = load_tracks(broadcast_json)
    tacticam_tracks = load_tracks(tacticam_json)
    for _, row in matches.iterrows():
        gid = row['global_id']
        b_id = str(row['broadcast_id'])
        t_id = str(row['tacticam_id'])
        sim = row['similarity']

        # First appearance of each track
        b_crop, t_crop = None, None
        for frame in broadcast_tracks:
            for t in frame['tracks']:
                if str(t['track_id']) == b_id:
                    b_crop = get_crop(os.path.join(video_dir, f"broadcast_frames/frame_{frame['frame_idx']:04d}.jpg"), t['bbox'])
                    break
            if b_crop is not None:
                break

        for frame in tacticam_tracks:
            for t in frame['tracks']:
                if str(t['track_id']) == t_id:
                    t_crop = get_crop(os.path.join(video_dir, f"tacticam_frames/frame_{frame['frame_idx']:04d}.jpg"), t['bbox'])
                    break
            if t_crop is not None:
                break

        if b_crop is not None and t_crop is not None:
            combined = cv2.hconcat([b_crop, t_crop])
            cv2.putText(combined, f"Global ID: {gid}, Sim: {sim:.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imwrite(os.path.join(save_dir, f"match_{gid}.jpg"), combined)

    print(f"Saved visual comparisons to {save_dir}")

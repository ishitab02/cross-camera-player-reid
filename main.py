import yaml
from src.detect import run_detection
from src.track import run_tracking
from extract_embeddings import extract_track_embeddings
from src.match import match_players
from src.assign_id import assign_global_ids

def load_config(path="config/config.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main():
    cfg = load_config()

    # Run detection on videos
    run_detection(
        video_path=cfg['videos']['broadcast'],
        weights_path=cfg['model']['weights_path'],
        output_json_path=cfg['output']['broadcast_detections'],
        visualize=True
    )

    run_detection(
        video_path=cfg['videos']['tacticam'],
        weights_path=cfg['model']['weights_path'],
        output_json_path=cfg['output']['tacticam_detections'],
        visualize=True
    )

    # Run tracking on both videos
    run_tracking(
        video_path=cfg['videos']['broadcast'],
        weights_path=cfg['model']['weights_path'],
        visualize=True
    )

    run_tracking(
        video_path=cfg['videos']['tacticam'],
        weights_path=cfg['model']['weights_path'],
        visualize=True
    )

    # Extract embeddings and match players 
    extract_track_embeddings(
        "data/broadcast.mp4", 
        "outputs/broadcast_track_logs.json", 
        "outputs/broadcast_embeddings.json")
    
    extract_track_embeddings("data/tacticam.mp4", 
                             "outputs/tacticam_track_logs.json", 
                             "outputs/tacticam_embeddings.json")
    
    match_players(
        broadcast_emb_path="outputs/broadcast_embeddings.json",
        tacticam_emb_path="outputs/tacticam_embeddings.json",
        threshold=0.91) 

    assign_global_ids(
        broadcast_tracks_path="outputs/broadcast_track_logs.json",
        tacticam_tracks_path="outputs/tacticam_track_logs.json",
        match_path="outputs/matched_players.json",
        out_broadcast_path="outputs/broadcast_track_global.json",
        out_tacticam_path="outputs/tacticam_track_global.json")

if __name__ == "__main__":
    main()

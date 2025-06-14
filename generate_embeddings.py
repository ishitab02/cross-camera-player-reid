from extract_embeddings import extract_track_embeddings

extract_track_embeddings(
    video_path="data/broadcast.mp4",
    track_log_path="outputs/broadcast_track_logs.json",
    output_path="outputs/broadcast_embeddings.json"
)

extract_track_embeddings(
    video_path="data/tacticam.mp4",
    track_log_path="outputs/tacticam_track_logs.json",
    output_path="outputs/tacticam_embeddings.json"
)
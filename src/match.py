import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def match_players(broadcast_emb_path, tacticam_emb_path, threshold=0.7):
    with open(broadcast_emb_path, 'r') as f1, open(tacticam_emb_path, 'r') as f2:
        broadcast_embs = json.load(f1)
        tacticam_embs = json.load(f2)

    broadcast_ids = list(broadcast_embs.keys())
    tacticam_ids = list(tacticam_embs.keys())

    B = np.array([broadcast_embs[bid] for bid in broadcast_ids])
    T = np.array([tacticam_embs[tid] for tid in tacticam_ids])

    sim_matrix = cosine_similarity(B, T)

    matches = []

    for i, b_id in enumerate(broadcast_ids):
        # Tacticam_id with highest similarity to broadcast_id
        best_j = np.argmax(sim_matrix[i])
        best_score = sim_matrix[i, best_j]
        if best_score >= threshold:
            matches.append({
                "broadcast_id": b_id,
                "tacticam_id": tacticam_ids[best_j],
                "similarity": float(best_score)
            })

    with open("outputs/matched_players.json", 'w') as f:
        json.dump(matches, f, indent=2)

    print("Saved top matched players to outputs/matched_players.json")


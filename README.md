# Cross-Camera Player Mapping

This project focuses on mapping players consistently across two different camera feeds of the same sports gameplay—broadcast.mp4 and tacticam.mp4. The goal is to assign consistent player IDs across both views using a provided object detection model, fine-tuned YOLOv11.

## How to Run

1. Clone the repository

```
git clone https://github.com/your-username/cross-camera-player-reid.git
cd cross-camera-player-reid
```

2. Install dependencies

```
pip install -r requirements.txt
```

3. Place your videos
   
  Make sure broadcast.mp4 and tacticam.mp4 videos are in the data/ folder.

4. Run the pipeline

```
python main.py
```

## Dependencies

- PyTorch, Torchvision – for model and embeddings
- OpenCV-Python – for video I/O
- Deep SORT Real-time – for tracking purposes
- NumPy, Pandas, Matplotlib, Scikit-learn – for processing & plotting
- Ultralytics – for trained and fine-tuned YOLOv11 model

## Features

- Detection with fine-tuned YOLOv11 model (trained for player and ball detection) on both camera views.
- Tracking using DeepSORT for persistent local IDs.
- Embeddings extraction with ResNet50 for each track.
- Cross-view matching using Cosine Similarity. 
- Global IDs assigned for consistent identity across views.
- CSV files of matches and similarity distributions.
- Annotated and tracked videos and JSON logs for visualization purposes.




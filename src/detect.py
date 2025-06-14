import cv2
from src.utils import load_model, save_detections_json
import os

def is_inside_field(x1, y1, x2, y2, width=1920, height=1080):
   # Define margins for the field
    margin_x = 100  
    margin_y = 80    
    return (
        x1 >= margin_x and x2 <= (width - margin_x) and
        y1 >= margin_y and y2 <= (height - margin_y))

def run_detection(video_path, weights_path, output_json_path, target_class_name="player", visualize=False):
    model = load_model(weights_path)
    cap = cv2.VideoCapture(video_path)
    frame_id = 0
    all_detections = []

    out_writer = None
    if visualize:
        os.makedirs("outputs", exist_ok=True)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        video_name = os.path.basename(video_path).replace(".mp4", "_annotated.mp4")
        output_video_path = os.path.join("outputs", video_name)
        out_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(source=frame, verbose=False)[0]
        frame_detections = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]

            if cls_name != target_class_name:
                continue

            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()  
            x1, y1, x2, y2 = map(int, xyxy)

            if not is_inside_field(x1, y1, x2, y2, width, height):
                if visualize:
                    # Rejected boxes in red
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, f"OUT", (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                continue  

            frame_detections.append({
                "frame": frame_id,
                "bbox": [x1, y1, x2, y2],
                "conf": conf})

            if visualize:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{cls_name} {conf:.2f}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        all_detections.append({
            "frame": frame_id,
            "detections": frame_detections})

        if visualize:
            out_writer.write(frame)
            cv2.imshow("Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_id += 1

    cap.release()
    if visualize:
        out_writer.release()
        cv2.destroyAllWindows()

    save_detections_json(all_detections, output_json_path)
    print(f"Saved detections to {output_json_path}")
    if visualize:
        print(f"Saved annotated video to {output_video_path}")
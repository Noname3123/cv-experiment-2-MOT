import os
import cv2
import numpy as np
from ultralytics import YOLO

# --- Configuration ---
MOT17_PATH = "MOT17"
MODEL_WEIGHTS = "yolo11x.pt"
INFERENCE_SIZE = 1280
CONF_THRESHOLD = 0.001  # Low threshold to allow flexibility in optimization

def process_sequence(seq_name, subset, model, mot_path):
    print(f"Processing sequence: {seq_name} ({subset})")
    
    # Handle path if running from MR/ subdirectory
    if not os.path.exists(mot_path) and os.path.exists(os.path.join("..", mot_path)):
        mot_path = os.path.join("..", mot_path)
        
    seq_dir = os.path.join(mot_path, subset, seq_name, "img1")
    if not os.path.exists(seq_dir):
        print(f"Sequence directory not found: {seq_dir}")
        return

    image_files = sorted([f for f in os.listdir(seq_dir) if f.endswith('.jpg')])
    
    all_detections = []

    for frame_idx, img_file in enumerate(image_files, start=1):
        img_path = os.path.join(seq_dir, img_file)
        frame = cv2.imread(img_path)
        if frame is None: continue

        # Run YOLO Detection
        # classes=[0] restricts to Person class
        results = model(frame, verbose=False, imgsz=INFERENCE_SIZE, conf=CONF_THRESHOLD, classes=[0])
        
        for result in results:
            boxes = result.boxes.data.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2, conf, cls = box
                # Store: frame_idx, x1, y1, x2, y2, conf, cls
                all_detections.append([frame_idx, x1, y1, x2, y2, conf, cls])
        
        if frame_idx % 100 == 0:
            print(f"  Processed {frame_idx}/{len(image_files)} frames")

    all_detections = np.array(all_detections)
    
    # Save in the sequence directory
    save_dir = os.path.join(mot_path, subset, seq_name)
    # Filename labeled with video name
    save_path = os.path.join(save_dir, f"{seq_name}_yolo_detections.npy")
    
    np.save(save_path, all_detections)
    print(f"Saved {len(all_detections)} detections to {save_path}")

def main():
    print(f"Loading model: {MODEL_WEIGHTS}...")
    model = YOLO(MODEL_WEIGHTS)
    
    # Locate MOT17
    mot_path = MOT17_PATH
    if not os.path.exists(mot_path) and os.path.exists(os.path.join("..", MOT17_PATH)):
        mot_path = os.path.join("..", MOT17_PATH)
    
    subsets = ["train", "test"]

    for subset in subsets:
        subset_path = os.path.join(mot_path, subset)
        if not os.path.exists(subset_path):
            print(f"MOT17 {subset} directory not found at {subset_path}")
            continue

        sequences = sorted([d for d in os.listdir(subset_path) if os.path.isdir(os.path.join(subset_path, d))])
        print(f"Found {len(sequences)} sequences in {subset}: {sequences}")

        for seq in sequences:
            process_sequence(seq, subset, model, mot_path)

if __name__ == "__main__":
    main()

import os
import shutil
import yaml
import datetime
import cv2
import numpy as np
import motmetrics as mm
from ultralytics import YOLO
from norfair import Detection, Tracker, draw_tracked_objects

# --- Configuration ---
MOT17_PATH = "MOT17"
YOLO_DATASET_DIR = "mot17_yolo_data_multiclass"  # Where to store converted training data
EXPERIMENT_DIR = "experiments"
OUTPUT_DIR = "tracking_results_trained"

# Training Config
BASE_MODEL = "yolo11l.pt"  # Start with nano model for speed
EPOCHS = 10                # Number of training epochs
IMG_SIZE = 640
BATCH_SIZE = 4

# Tracking Config
CONFIDENCE_THRESHOLD = 0.4
DISTANCE_THRESHOLD = 30
HIT_INERTIA_MIN = 3
HIT_INERTIA_MAX = 6

# --- Part 1: Data Preparation (MOT17 GT -> YOLO Labels) ---

def convert_mot_to_yolo():
    print("Preparing YOLO training data from MOT17...")
    
    # Create dataset structure
    images_train_dir = os.path.join(YOLO_DATASET_DIR, "images", "train")
    labels_train_dir = os.path.join(YOLO_DATASET_DIR, "labels", "train")
    
    if os.path.exists(YOLO_DATASET_DIR):
        print(f"Dataset directory {YOLO_DATASET_DIR} exists. Skipping conversion (delete folder to force regenerate).")
        return
    
    os.makedirs(images_train_dir, exist_ok=True)
    os.makedirs(labels_train_dir, exist_ok=True)

    train_path = os.path.join(MOT17_PATH, "train")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"MOT17 train directory not found at {train_path}")

    # Get sequences. MOT17 has duplicates (02-SDP, 02-DPM, etc.). We only need one copy for training images.
    # We prefer SDP as it usually has slightly cleaner crops, but it doesn't matter much for images.
    sequences = sorted(os.listdir(train_path))
    processed_ids = set()

    for seq in sequences:
        seq_path = os.path.join(train_path, seq)
        if not os.path.isdir(seq_path): continue
        
        # Extract sequence ID (e.g., "MOT17-04") to avoid duplicates
        seq_id = "-".join(seq.split("-")[:2])
        if seq_id in processed_ids:
            continue
        processed_ids.add(seq_id)

        print(f"Processing sequence for training: {seq}")
        
        img_dir = os.path.join(seq_path, "img1")
        gt_file = os.path.join(seq_path, "gt", "gt.txt")
        
        if not os.path.exists(gt_file):
            print(f"  No GT found for {seq}, skipping.")
            continue

        # Read GT
        # Format: frame, id, left, top, width, height, conf, class, visibility
        gt_data = np.loadtxt(gt_file, delimiter=',')
        
        # Get image dimensions from the first image
        first_img_name = sorted(os.listdir(img_dir))[0]
        first_img = cv2.imread(os.path.join(img_dir, first_img_name))
        img_h, img_w = first_img.shape[:2]

        # Group by frame
        frames = np.unique(gt_data[:, 0])
        
        for frame_idx in frames:
            frame_rows = gt_data[gt_data[:, 0] == frame_idx]
            
            # YOLO Label content
            yolo_lines = []
            for row in frame_rows:
                # MOT17 Classes: 1=Pedestrian, 2=Person on vehicle, 7=Static person, etc.
                cls_id = int(row[7])
                
                # Visibility check (optional, but good for training stability)
                vis = row[8]
                if vis < 0.2: # Skip heavily occluded
                    continue

                # Convert xywh (top-left) to xywh (center, normalized)
                x1, y1, w, h = row[2], row[3], row[4], row[5]
                
                cx = (x1 + w / 2) / img_w
                cy = (y1 + h / 2) / img_h
                nw = w / img_w
                nh = h / img_h
                
                # YOLO class 0-11 (MOT class ID - 1)
                yolo_lines.append(f"{cls_id - 1} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

            if not yolo_lines:
                continue

            # Copy Image
            src_img = os.path.join(img_dir, f"{int(frame_idx):06d}.jpg")
            dst_img_name = f"{seq}_{int(frame_idx):06d}.jpg"
            dst_img = os.path.join(images_train_dir, dst_img_name)
            shutil.copy(src_img, dst_img)
            
            # Write Label
            dst_label = os.path.join(labels_train_dir, dst_img_name.replace('.jpg', '.txt'))
            with open(dst_label, 'w') as f:
                f.write("\n".join(yolo_lines))

    # Create YAML file for Ultralytics
    yaml_content = {
        'path': os.path.abspath(YOLO_DATASET_DIR),
        'train': 'images/train',
        'val': 'images/train', # Use train as val for simplicity in this script
        'names': {
            0: 'Pedestrian', 1: 'Person_on_vehicle', 2: 'Car', 3: 'Bicycle',
            4: 'Motorbike', 5: 'Non_motorized_vehicle', 6: 'Static_person',
            7: 'Distractor', 8: 'Occluder', 9: 'Occluder_on_ground',
            10: 'Occluder_full', 11: 'Reflection'
        }
    }
    
    with open(os.path.join(YOLO_DATASET_DIR, "mot17.yaml"), 'w') as f:
        yaml.dump(yaml_content, f)
        
    print("Data preparation complete.")

# --- Part 2: Training ---

def train_yolo():
    print(f"Starting training with base model {BASE_MODEL}...")
    model = YOLO(BASE_MODEL)
    
    yaml_path = os.path.join(YOLO_DATASET_DIR, "mot17.yaml")
    
    # Train
    results = model.train(
        data=yaml_path,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        project=EXPERIMENT_DIR,
        name="yolo_mot17_finetune",
        exist_ok=True, # Overwrite existing experiment folder
        verbose=True
    )
    
    # Return path to best weights
    best_weight = os.path.join(EXPERIMENT_DIR, "yolo_mot17_finetune", "weights", "best.pt")
    print(f"Training finished. Best weights at: {best_weight}")
    return best_weight

# --- Part 3: Tracking & Evaluation (Reused from yolo_norfair.py) ---

def euclidean_distance(detection, tracked_object):
    return np.linalg.norm(detection.points - tracked_object.estimate)

def yolo_to_norfair(yolo_results, conf_threshold):
    norfair_detections = []
    for result in yolo_results:
        boxes = result.boxes.data.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2, conf, cls = box
            if conf >= conf_threshold:
                centroid = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
                norfair_detections.append(Detection(points=centroid, data=np.array([x1, y1, x2, y2, conf])))
    return norfair_detections

def process_sequence(seq_name, model, output_path):
    print(f"Tracking sequence: {seq_name}")
    
    # Find sequence in test dir
    seq_dir = None
    for subdir in ["test"]:
        d = os.path.join(MOT17_PATH, subdir, seq_name, "img1")
        if os.path.exists(d):
            seq_dir = d
            break
    
    if not seq_dir:
        print(f"Sequence {seq_name} not found in test folder.")
        return None, None

    image_files = sorted([f for f in os.listdir(seq_dir) if f.endswith('.jpg')])
    
    tracker = Tracker(
        distance_function=euclidean_distance,
        distance_threshold=DISTANCE_THRESHOLD,
        initialization_delay=HIT_INERTIA_MIN,
        hit_counter_max=HIT_INERTIA_MAX
    )

    results_file = os.path.join(output_path, f"{seq_name}.txt")
    
    with open(results_file, 'w') as f_out:
        for frame_idx, img_file in enumerate(image_files, start=1):
            img_path = os.path.join(seq_dir, img_file)
            frame = cv2.imread(img_path)
            if frame is None: continue

            yolo_results = model(frame, verbose=False)
            detections = yolo_to_norfair(yolo_results, CONFIDENCE_THRESHOLD)
            tracked_objects = tracker.update(detections=detections)

            for obj in tracked_objects:
                if obj.age >= HIT_INERTIA_MIN:
                    if obj.last_detection is not None:
                        d_data = obj.last_detection.data
                        x1, y1, x2, y2, conf = d_data
                    else:
                        last_data = obj.last_detection.data if obj.last_detection else [0,0,0,0,0]
                        w, h = last_data[2] - last_data[0], last_data[3] - last_data[1]
                        cx, cy = obj.estimate[0]
                        x1, y1, x2, y2 = cx - w/2, cy - h/2, cx + w/2, cy + h/2
                        conf = 1.0

                    w, h = x2 - x1, y2 - y1
                    f_out.write(f"{frame_idx},{obj.id},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{conf:.2f},-1,-1,-1\n")

    gt_file = os.path.join(seq_dir, '..', 'gt', 'gt.txt')
    return results_file, gt_file

def main():
    # 1. Prepare Data
    convert_mot_to_yolo()
    
    # 2. Train Model
    best_model_path = train_yolo()
    
    # 3. Evaluate
    print(f"Loading trained model from {best_model_path} for evaluation...")
    model = YOLO(best_model_path)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    sequences = []
    test_path = os.path.join(MOT17_PATH, "test")
    if os.path.exists(test_path):
        sequences = sorted([d for d in os.listdir(test_path) if os.path.isdir(os.path.join(test_path, d))])

    accs, names = [], []
    for seq in sequences:
        res_file, gt_file = process_sequence(seq, model, OUTPUT_DIR)
        if gt_file and os.path.exists(gt_file):
            gt = mm.io.loadtxt(gt_file, fmt="mot15-2D", min_confidence=1)
            ts = mm.io.loadtxt(res_file, fmt="mot15-2D")
            acc = mm.utils.compare_to_groundtruth(gt, ts, 'iou', distth=0.5)
            accs.append(acc)
            names.append(seq)

    if accs:
        print("\nComputing Metrics on Test Set...")
        mh = mm.metrics.create()
        summary = mh.compute_many(accs, metrics=mm.metrics.motchallenge_metrics, names=names, generate_overall=True)
        str_summary = mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names)
        print(str_summary)
        
        # Logging
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(EXPERIMENT_DIR, f"experiment_trained_{timestamp}.txt")
        with open(log_file, "w") as f:
            f.write(f"Experiment Log - {timestamp}\n")
            f.write("========================================\n")
            f.write(f"Model: {BASE_MODEL} (Finetuned on MOT17 all classes)\n")
            f.write(f"Tracker: Norfair\n")
            f.write("Hyperparameters:\n")
            f.write(f"  EPOCHS: {EPOCHS}\n")
            f.write(f"  IMG_SIZE: {IMG_SIZE}\n")
            f.write(f"  BATCH_SIZE: {BATCH_SIZE}\n")
            f.write(f"  CONFIDENCE_THRESHOLD: {CONFIDENCE_THRESHOLD}\n")
            f.write(f"  DISTANCE_THRESHOLD: {DISTANCE_THRESHOLD}\n")
            f.write(f"  HIT_INERTIA_MIN: {HIT_INERTIA_MIN}\n")
            f.write(f"  HIT_INERTIA_MAX: {HIT_INERTIA_MAX}\n")
            f.write("\nResults:\n")
            f.write(str_summary)
        print(f"\nExperiment results saved to {log_file}")
    else:
        print("No Ground Truth found in test folder for evaluation.")

if __name__ == "__main__":
    main()
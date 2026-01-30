import os
import datetime
import cv2
import numpy as np
from ultralytics import YOLO
from norfair import Detection, Tracker, Video, draw_tracked_objects
import motmetrics as mm
import pandas as pd

# Configuration
MOT17_PATH = "MOT17"  # Update this path
EXPERIMENT_DIR = "experiments"
OUTPUT_DIR = "tracking_results"
MODEL_WEIGHTS = "yolo11l.pt"  # Can be yolo11n, yolo11s, yolo11m, yolo11l, yolo11x
CONFIDENCE_THRESHOLD = 0.4
DISTANCE_THRESHOLD = 30  # Norfair parameter: max pixel distance to match
HIT_INERTIA_MIN = 3      # Norfair parameter: min hits to establish a track
HIT_INERTIA_MAX = 6      # Norfair parameter: max frames to keep a lost track

def euclidean_distance(detection, tracked_object):
    """
    Computes Euclidean distance between a detection centroid and a tracked object centroid.
    """
    return np.linalg.norm(detection.points - tracked_object.estimate)

def yolo_to_norfair(yolo_results, conf_threshold):
    """
    Converts YOLO detections to Norfair Detection objects.
    YOLO format: [x1, y1, x2, y2, conf, cls]
    """
    norfair_detections = []
    
    # yolo_results is a list of Result objects
    for result in yolo_results:
        boxes = result.boxes.data.cpu().numpy()
        
        for box in boxes:
            x1, y1, x2, y2, conf, cls = box
            
            # Filter by confidence and class (0 is usually 'person' in COCO)
            if conf >= conf_threshold and int(cls) == 0:
                centroid = np.array([
                    (x1 + x2) / 2,
                    (y1 + y2) / 2
                ])
                
                # We store the raw box coordinates in 'data' to retrieve them later
                norfair_detections.append(
                    Detection(points=centroid, data=np.array([x1, y1, x2, y2, conf]))
                )
                
    return norfair_detections

def process_sequence(seq_name, model, output_path, visualize=False):
    print(f"Processing sequence: {seq_name}")
    
    # Construct paths
    seq_dir = None
    for subdir in ["test"]:
        d = os.path.join(MOT17_PATH, subdir, seq_name, "img1")
        if os.path.exists(d):
            seq_dir = d
            break

    if seq_dir is None:
        print(f"Sequence directory not found: {seq_name}")
        return None, None

    # Get sorted list of images
    image_files = sorted([f for f in os.listdir(seq_dir) if f.endswith('.jpg')])
    
    # Initialize Norfair Tracker
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
            
            if frame is None:
                continue

            # 1. Run YOLO Detection
            # verbose=False suppresses the per-frame print logs
            yolo_results = model(frame, verbose=False)

            # 2. Convert to Norfair format
            detections = yolo_to_norfair(yolo_results, CONFIDENCE_THRESHOLD)

            # 3. Update Tracker
            tracked_objects = tracker.update(detections=detections)

            # 4. Process Results
            for obj in tracked_objects:
                # Only save objects that are established (hit_inertia_min passed)
                # unless you want to report everything immediately.
                if obj.age >= HIT_INERTIA_MIN: # Check if object is stable
                    
                    # Retrieve the bounding box. 
                    # If the object is currently being detected, use the detection box.
                    # If it is a prediction (occluded), use the estimate or last known box.
                    if obj.last_detection is not None:
                        # Use the actual detection box
                        d_data = obj.last_detection.data
                        x1, y1, x2, y2, conf = d_data
                    else:
                        # If lost, Norfair tracks the point (centroid). 
                        # We need to estimate the box size based on history.
                        # For simplicity here, we use the last known detection's size
                        # centered around the new estimated centroid.
                        last_data = obj.last_detection.data if obj.last_detection else [0,0,0,0,0]
                        w = last_data[2] - last_data[0]
                        h = last_data[3] - last_data[1]
                        
                        cx, cy = obj.estimate[0]
                        x1 = cx - w/2
                        y1 = cy - h/2
                        x2 = cx + w/2
                        y2 = cy + h/2
                        conf = 1.0 # Confidence is arbitrary for predicted tracks

                    # MOT Format: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
                    w = x2 - x1
                    h = y2 - y1
                    
                    line = f"{frame_idx},{obj.id},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{conf:.2f},-1,-1,-1\n"
                    f_out.write(line)

            # Optional: Visualization
            if visualize:
                draw_tracked_objects(frame, tracked_objects)
                cv2.imshow("Tracking", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    
    if visualize:
        cv2.destroyAllWindows()
    print(f"Finished {seq_name}. Results saved to {results_file}")
    
    # Return paths for evaluation
    gt_file = os.path.join(seq_dir, '..', 'gt', 'gt.txt')
    return results_file, gt_file

def compute_metrics(accs, names):
    """
    Compute and print MOT metrics from accumulators.
    """
    print("\nComputing Metrics...")
    mh = mm.metrics.create()
    
    summary = mh.compute_many(
        accs, 
        metrics=mm.metrics.motchallenge_metrics, 
        names=names,
        generate_overall=True
    )
    
    str_summary = mm.io.render_summary(
        summary, 
        formatters=mh.formatters, 
        namemap=mm.io.motchallenge_metric_names
    )
    print(str_summary)
    print("\nNote: HOTA metric requires the official TrackEval library and is not included in motmetrics.")
    return str_summary

def main():
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load YOLO Model
    # Ultralytics will automatically download the model if not found
    print(f"Loading model: {MODEL_WEIGHTS}...")
    model = YOLO(MODEL_WEIGHTS)

    # List of sequences to process
    # Auto-discover sequences in train and test folders
    sequences = []
    for subdir in ["test"]:
        subdir_path = os.path.join(MOT17_PATH, subdir)
        if os.path.exists(subdir_path):
            for seq in os.listdir(subdir_path):
                # Check if it's a directory and has img1
                if os.path.isdir(os.path.join(subdir_path, seq)):
                    sequences.append(seq)
    sequences = sorted(sequences)
    print(f"Found {len(sequences)} sequences: {sequences}")

    accs = []
    names = []
    
    for seq in sequences:
        res_file, gt_file = process_sequence(seq, model, OUTPUT_DIR, visualize=False)
        
        # If GT exists, accumulate for evaluation
        if gt_file and os.path.exists(gt_file):
            # Load GT and Results
            # motmetrics expects standard MOT format
            gt = mm.io.loadtxt(gt_file, fmt="mot15-2D", min_confidence=1)
            ts = mm.io.loadtxt(res_file, fmt="mot15-2D")
            
            # Compare
            acc = mm.utils.compare_to_groundtruth(gt, ts, 'iou', distth=0.5)
            accs.append(acc)
            names.append(seq)

    if accs:
        metrics_summary = compute_metrics(accs, names)
        
        # Document results
        os.makedirs(EXPERIMENT_DIR, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(EXPERIMENT_DIR, f"experiment_{timestamp}.txt")
        
        with open(log_file, "w") as f:
            f.write(f"Experiment Log - {timestamp}\n")
            f.write("========================================\n")
            f.write(f"Model: {MODEL_WEIGHTS}\n")
            f.write(f"Tracker: Norfair\n")
            f.write("Hyperparameters:\n")
            f.write(f"  CONFIDENCE_THRESHOLD: {CONFIDENCE_THRESHOLD}\n")
            f.write(f"  DISTANCE_THRESHOLD: {DISTANCE_THRESHOLD}\n")
            f.write(f"  HIT_INERTIA_MIN: {HIT_INERTIA_MIN}\n")
            f.write(f"  HIT_INERTIA_MAX: {HIT_INERTIA_MAX}\n")
            f.write("\nResults:\n")
            f.write(metrics_summary)
            f.write("\n\nNote: HOTA metric requires the official TrackEval library.\n")
            
        print(f"\nExperiment results saved to {log_file}")
    else:
        print("No Ground Truth found for evaluation.")

if __name__ == "__main__":
    main()

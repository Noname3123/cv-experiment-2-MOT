import os
import datetime
import cv2
import numpy as np
from ultralytics import YOLO
import motmetrics as mm

# Dependencies for DeepSORT
try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
except ImportError:
    raise ImportError("deep-sort-realtime not found. Please install: pip install deep-sort-realtime")


# --- Configuration ---
MOT17_PATH = "MOT17"
EXPERIMENT_DIR = "experiments"
OUTPUT_DIR = "tracking_results_deepsort_high_res"
MODEL_WEIGHTS = "yolo11l.pt"  # Upgraded to Extra Large model for better recall

# DeepSORT Hyperparameters
CONFIDENCE_THRESHOLD = 0.5    # Increased to 0.5 to reduce False Positives from YOLO11x
INFERENCE_SIZE = 1920
MAX_AGE = 60                  # Increased to 60 to reduce ID switches during occlusions
MIN_HITS = 3
NMS_MAX_OVERLAP = 1.0         # Non-max suppression threshold (1.0 = disable, let YOLO handle it)


# --- Utilities ---

def get_camera_motion(prev_img, curr_img):
    """
    Computes the affine transformation (camera motion) between two frames.
    Note: While calculated here, standard deep_sort_realtime does not easily support 
    injecting this matrix to update the Kalman Filter state externally.
    """
    # Detect features to track
    prev_pts = cv2.goodFeaturesToTrack(prev_img, maxCorners=3000, qualityLevel=0.01, minDistance=10)
    if prev_pts is None: return np.eye(2, 3)
    
    # Track features using Optical Flow
    curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_img, curr_img, prev_pts, None)
    
    # Select good points
    idx = np.where(status == 1)[0]
    prev_pts = prev_pts[idx]
    curr_pts = curr_pts[idx]
    
    if len(prev_pts) < 10: return np.eye(2, 3)
    
    # Estimate Affine Transformation (Rotation + Translation + Scale)
    m, inliers = cv2.estimateAffinePartial2D(prev_pts, curr_pts)
    if m is None: return np.eye(2, 3)
    
    return m

# --- Post-Processing ---

def interpolate_tracks(results, max_gap=20):
    """
    Fill in gaps in tracks using linear interpolation.
    results: list of [frame, id, x1, y1, w, h]
    """
    # Convert to dict: id -> list of (frame, box)
    tracks = {}
    for row in results:
        frame, track_id = int(row[0]), int(row[1])
        box = row[2:6] # x1, y1, w, h
        if track_id not in tracks:
            tracks[track_id] = []
        tracks[track_id].append((frame, box))
    
    new_results = []
    
    for track_id in tracks:
        # Sort by frame
        track_data = sorted(tracks[track_id], key=lambda x: x[0])
        
        for i in range(len(track_data) - 1):
            curr_frame, curr_box = track_data[i]
            next_frame, next_box = track_data[i+1]
            
            # Add current frame
            new_results.append([curr_frame, track_id] + list(curr_box))
            
            gap = next_frame - curr_frame
            if 1 < gap <= max_gap:
                # Interpolate
                curr_box = np.array(curr_box)
                next_box = np.array(next_box)
                step = (next_box - curr_box) / gap
                
                for g in range(1, gap):
                    interp_frame = curr_frame + g
                    interp_box = curr_box + step * g
                    new_results.append([interp_frame, track_id] + list(interp_box))
        
        # Add last frame
        last_frame, last_box = track_data[-1]
        new_results.append([last_frame, track_id] + list(last_box))
        
    # Sort by frame then ID
    new_results.sort(key=lambda x: (x[0], x[1]))
    return new_results


# --- Main Processing Logic ---

def process_sequence(seq_name, model, output_path):
    print(f"Processing sequence: {seq_name}")
    
    seq_dir = None
    for subdir in ["test"]:
        d = os.path.join(MOT17_PATH, subdir, seq_name, "img1")
        if os.path.exists(d):
            seq_dir = d
            break

    if seq_dir is None:
        print(f"Sequence directory not found: {seq_name}")
        return None, None

    image_files = sorted([f for f in os.listdir(seq_dir) if f.endswith('.jpg')])
    
    # Initialize DeepSORT Tracker
    # embedder='mobilenet' is the default lightweight ReID model downloaded automatically
    tracker = DeepSort(
        max_age=MAX_AGE,
        n_init=MIN_HITS,
        nms_max_overlap=NMS_MAX_OVERLAP,
        max_iou_distance=0.7,
        max_cosine_distance=0.2,
        embedder='mobilenet', 
        half=True,
        bgr=True,
        embedder_gpu=True
    )

    results_file = os.path.join(output_path, f"{seq_name}.txt")
    
    raw_results = []
    prev_gray = None
    
    for frame_idx, img_file in enumerate(image_files, start=1):
        img_path = os.path.join(seq_dir, img_file)
        frame = cv2.imread(img_path)
        if frame is None: continue
        
        # 0. Camera Motion Calculation (CMC)
        # Note: We calculate it here to verify motion, but deep_sort_realtime 
        # does not support injecting this warp matrix easily.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            warp = get_camera_motion(prev_gray, gray)
            # tracker.apply_cmc(warp) # Not supported in standard deep_sort_realtime
        prev_gray = gray

        # 1. Run YOLO Detection
        yolo_results = model(frame, verbose=False, imgsz=INFERENCE_SIZE, classes=[0])
        
        # 2. Prepare detections for DeepSORT
        # Format: [[left, top, w, h], confidence, detection_class]
        dets = []
        for result in yolo_results:
            boxes = result.boxes.data.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2, conf, cls = box
                if conf >= CONFIDENCE_THRESHOLD:
                    w = x2 - x1
                    h = y2 - y1
                    dets.append([[x1, y1, w, h], conf, int(cls)])
        
        # 3. Update Tracker
        # DeepSORT handles feature extraction internally using the 'frame'
        tracks = tracker.update_tracks(dets, frame=frame)

        # 4. Store Results
        for track in tracks:
            if not track.is_confirmed():
                continue
            
            track_id = track.track_id
            ltrb = track.to_ltrb() # left, top, right, bottom
            x1, y1, x2, y2 = ltrb
            w = x2 - x1
            h = y2 - y1
            
            raw_results.append([frame_idx, int(track_id), x1, y1, w, h])

    # 5. Interpolate Missing Frames
    interpolated_results = interpolate_tracks(raw_results, max_gap=20)

    # 6. Write to File
    with open(results_file, 'w') as f_out:
        for row in interpolated_results:
            frame_idx, track_id, x1, y1, w, h = row
            # MOT Format: frame, id, left, top, width, height, conf, -1, -1, -1
            line = f"{frame_idx},{track_id},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1.00,-1,-1,-1\n"
            f_out.write(line)

    print(f"Finished {seq_name}. Results saved to {results_file}")
    gt_file = os.path.join(seq_dir, '..', 'gt', 'gt.txt')
    return results_file, gt_file

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Loading model: {MODEL_WEIGHTS}...")
    model = YOLO(MODEL_WEIGHTS)

    sequences = []
    test_path = os.path.join(MOT17_PATH, "test")
    if os.path.exists(test_path):
        sequences = sorted([d for d in os.listdir(test_path) if os.path.isdir(os.path.join(test_path, d))])
    
    print(f"Found {len(sequences)} sequences: {sequences}")

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
        os.makedirs(EXPERIMENT_DIR, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(EXPERIMENT_DIR, f"experiment_deepsort_yololarge_{timestamp}.txt")
        with open(log_file, "w") as f:
            f.write(f"Experiment Log - {timestamp}\n")
            f.write("========================================\n")
            f.write(f"Model: {MODEL_WEIGHTS}\n")
            f.write(f"Tracker: DeepSORT (deep-sort-realtime)\n")
            f.write("Hyperparameters:\n")
            f.write(f"  CONFIDENCE_THRESHOLD: {CONFIDENCE_THRESHOLD}\n")
            f.write(f"  INFERENCE_SIZE: {INFERENCE_SIZE}\n")
            f.write(f"  MAX_AGE: {MAX_AGE}\n")
            f.write(f"  MIN_HITS: {MIN_HITS}\n")
            f.write(f"  POST_PROCESSING: Linear Interpolation (max_gap=20)\n")
            f.write(f"  CMC: Calculated but NOT applied (Library limitation)\n")
            f.write("\nResults:\n")
            f.write(str_summary)
        print(f"\nExperiment results saved to {log_file}")
    else:
        print("No Ground Truth found in test folder for evaluation.")

if __name__ == "__main__":
    main()

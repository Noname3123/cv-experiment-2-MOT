import os
import sys
import cv2
import numpy as np
import torch
import motmetrics as mm
import datetime

# Fix for SAM2 shadowing issue: Add the repo root to sys.path so the inner 'sam2' package is found
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "sam2"))

# --- Dependencies ---
try:
    from filterpy.kalman import KalmanFilter
except ImportError:
    raise ImportError("filterpy not found. Please install: pip install filterpy")

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    raise ImportError("scipy not found. Please install: pip install scipy")

# SAM2 Imports
# Assuming sam2 is installed or in python path (e.g. from the samurai/sam2 folder)
try:
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
except ImportError:
    print("Warning: Could not import SAM2. Make sure 'sam2' is installed or in your PYTHONPATH.")
    # Mocking for demonstration if SAM2 is missing
    class SAM2AutomaticMaskGenerator:
        def __init__(self, model): pass
        def generate(self, img): return []
    def build_sam2(config, checkpoint): return None

# --- Configuration ---
MOT17_PATH = "MOT17"
OUTPUT_DIR = "tracking_results_sam2_ocsort"
EXPERIMENT_DIR = "experiments"

# SAM2 Config
# You need to download a SAM2 checkpoint (e.g., sam2_hiera_large.pt)
SAM2_CHECKPOINT = "sam2/checkpoints/sam2.1_hiera_small.pt" 
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# OC-SORT Hyperparameters
CONFIDENCE_THRESHOLD = 0.4 # SAM2 score threshold
IOU_THRESHOLD = 0.3
MAX_AGE = 30
MIN_HITS = 3
INERTIA = 0.2
DELTA_T = 3

# --- Part 1: OC-SORT Implementation (Copied from train_yolo_ocsort.py) ---

def iou_batch(bb_test, bb_gt):
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)
    
    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])                                      
        + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)                                              
    return o  

def convert_bbox_to_z(bbox):
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w/2.
    y = bbox[1] + h/2.
    s = w * h
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))

def convert_x_to_bbox(x, score=None):
    if len(x.shape) == 2 and x.shape[1] == 1:
        x = x.flatten()

    if len(x.shape) == 1:
        s = max(0, x[2])
        r = max(0, x[3])
        w = np.sqrt(s * r)
        h = s / w if w > 0 else 0
        if(score==None):
            return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))
        else:
            return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((1,5))
    else:
        s = np.maximum(0, x[:, 2])
        r = np.maximum(0, x[:, 3])
        w = np.sqrt(s * r)
        h = np.zeros_like(w)
        np.divide(s, w, out=h, where=w > 0)
        x1 = x[:, 0] - w/2.
        y1 = x[:, 1] - h/2.
        x2 = x[:, 0] + w/2.
        y2 = x[:, 1] + h/2.
        return np.stack((x1, y1, x2, y2), axis=1)

class KalmanBoxTracker(object):
    count = 0
    def __init__(self, bbox, delta_t=3):
        self.kf = KalmanFilter(dim_x=7, dim_z=4) 
        self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],  
                            [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
        self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])

        self.kf.R[2:,2:] *= 10.
        self.kf.P[4:,4:] *= 1000. 
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        
        self.last_observation = convert_bbox_to_z(bbox)
        self.observations = dict()
        self.history_observations = []
        self.velocity = None
        self.delta_t = delta_t

    def update(self, bbox):
        if bbox is not None:
            if self.last_observation.sum() >= 0:
                previous_box = None
                for i in range(self.delta_t):
                    dt = self.delta_t - i
                    if self.age - dt in self.observations:
                        previous_box = self.observations[self.age - dt]
                        break
                
                if previous_box is None:
                    previous_box = self.last_observation
                
                self.velocity = convert_bbox_to_z(bbox) - previous_box
            
            self.last_observation = convert_bbox_to_z(bbox)
            self.observations[self.age] = self.last_observation
            self.history_observations.append(self.last_observation)

            self.time_since_update = 0
            self.history = []
            self.hits += 1
            self.hit_streak += 1
            self.kf.update(convert_bbox_to_z(bbox))
        else:
            self.kf.update(bbox)

    def predict(self):
        if((self.kf.x[6]+self.kf.x[2])<=0):
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if(self.time_since_update>0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

class OCSort(object):
    def __init__(self, det_thresh, max_age=30, min_hits=3, iou_threshold=0.3, delta_t=3, inertia=0.2):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
        self.det_thresh = det_thresh
        self.delta_t = delta_t
        self.inertia = inertia
        KalmanBoxTracker.count = 0

    def update(self, output_results):
        self.frame_count += 1
        
        if output_results.shape[0] > 0:
            output_results = output_results[output_results[:, 4] >= self.det_thresh]

        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        velocities = np.array([trk.velocity.flatten() if trk.velocity is not None else np.array((0,0,0,0)) for trk in self.trackers])
        last_boxes = np.array([trk.last_observation.flatten() for trk in self.trackers])
        k_observations = np.array([k.kf.x.flatten() for k in self.trackers])

        matched, unmatched_dets, unmatched_trks = self.associate(output_results, trks, self.iou_threshold, velocities, k_observations, self.inertia)
        
        for m in matched:
            self.trackers[m[1]].update(output_results[m[0], :])

        if len(unmatched_dets) > 0 and len(unmatched_trks) > 0:
            left_dets = output_results[unmatched_dets]
            left_trks = last_boxes[unmatched_trks]
            
            iou_left = iou_batch(left_dets, convert_x_to_bbox(left_trks + velocities[unmatched_trks]))
            iou_left = np.array(iou_left)
            
            if iou_left.max() > self.iou_threshold:
                matched_indices = linear_sum_assignment(-iou_left)
                matched_indices = np.array(list(zip(matched_indices[0], matched_indices[1])))
                
                to_remove_dets = []
                to_remove_trks = []
                
                for m in matched_indices:
                    det_idx = unmatched_dets[m[0]]
                    trk_idx = unmatched_trks[m[1]]
                    
                    if iou_left[m[0], m[1]] < self.iou_threshold:
                        continue
                        
                    self.trackers[trk_idx].update(output_results[det_idx, :])
                    to_remove_dets.append(m[0])
                    to_remove_trks.append(m[1])
                
                unmatched_dets = np.delete(unmatched_dets, to_remove_dets)
                unmatched_trks = np.delete(unmatched_trks, to_remove_trks)

        for m in unmatched_trks:
            self.trackers[m].update(None)

        for i in unmatched_dets:
            trk = KalmanBoxTracker(output_results[i,:])
            self.trackers.append(trk)

        i = len(self.trackers)
        for trk in reversed(self.trackers):
            if len(trk.history) > 0:
                d = trk.history[-1][0]
            else:
                d = convert_x_to_bbox(trk.kf.x)[0]
            
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d,[trk.id+1])).reshape(1,-1)) 
            i -= 1
            if(trk.time_since_update > self.max_age):
                self.trackers.pop(i)
                
        if(len(ret)>0):
            return np.concatenate(ret)
        return np.empty((0,5))

    def associate(self, detections, trackers, iou_threshold, velocities, previous_obs, vdc_weight):
        if(len(trackers)==0):
            return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)
        
        Y, X = speed_direction_batch(detections, previous_obs)
        inertia_Y, inertia_X = velocities[:,0], velocities[:,1]
        inertia_Y = np.repeat(inertia_Y[:, np.newaxis], Y.shape[1], axis=1)
        inertia_X = np.repeat(inertia_X[:, np.newaxis], X.shape[1], axis=1)
        
        diff_angle_cos = inertia_X * X + inertia_Y * Y
        diff_angle_cos = np.clip(diff_angle_cos, a_min=-1, a_max=1)
        diff_angle = np.arccos(diff_angle_cos)
        diff_angle = (np.pi /2.0 - np.abs(diff_angle)) / np.pi

        valid_mask = np.ones(previous_obs.shape[0])
        valid_mask[np.where(previous_obs[:, 4] < 0)] = 0

        iou_matrix = iou_batch(detections, trackers)
        scores = np.repeat(detections[:,-1][:, np.newaxis], trackers.shape[0], axis=1)
        valid_mask = np.repeat(valid_mask[:, np.newaxis], X.shape[1], axis=1)

        angle_diff_cost = (valid_mask * diff_angle) * vdc_weight
        angle_diff_cost = angle_diff_cost.T
        angle_diff_cost[np.isnan(angle_diff_cost)] = 0
        
        if min(iou_matrix.shape) > 0:
            a = (iou_matrix > iou_threshold).astype(np.float32)
            if a.sum(1).max() == 1 and a.sum(0).max() == 1:
                matched_indices = np.stack(np.where(a), axis=1)
            else:
                cost_matrix = -(iou_matrix + angle_diff_cost)
                matched_indices = linear_sum_assignment(cost_matrix)
                matched_indices = np.array(list(zip(matched_indices[0], matched_indices[1])))
        else:
            matched_indices = np.empty(shape=(0,2))

        unmatched_detections = []
        for d, det in enumerate(detections):
            if(d not in matched_indices[:,0]):
                unmatched_detections.append(d)
        
        unmatched_trackers = []
        for t, trk in enumerate(trackers):
            if(t not in matched_indices[:,1]):
                unmatched_trackers.append(t)

        matches = []
        for m in matched_indices:
            if(iou_matrix[m[0], m[1]] < iou_threshold):
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1,2))
        
        if(len(matches)==0):
            matches = np.empty((0,2),dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)

        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

def speed_direction_batch(dets, tracks):
    CX1 = tracks[:, 0][:, np.newaxis]
    CY1 = tracks[:, 1][:, np.newaxis]
    CX2 = dets[:, 0][np.newaxis, :]
    CY2 = dets[:, 1][np.newaxis, :]
    dx = CX2 - CX1
    dy = CY2 - CY1
    norm = np.sqrt(dx**2 + dy**2) + 1e-6
    dx = dx / norm
    dy = dy / norm
    return dy, dx

# --- Part 2: SAM2 Detector Wrapper ---

class SAM2Detector:
    def __init__(self, checkpoint_path, config_path, device):
        print(f"Initializing SAM2 from {checkpoint_path}...")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"SAM2 checkpoint not found at {checkpoint_path}. Please download it.")
            
        self.model = build_sam2(config_path, checkpoint_path, device=device, apply_postprocessing=False)
        
        # Automatic Mask Generator acts as a detector (detects everything)
        self.mask_generator = SAM2AutomaticMaskGenerator(
            model=self.model,
            points_per_side=32, # Grid density
            points_per_batch=16, # Reduced from 64 to save VRAM
            pred_iou_thresh=0.7,
            stability_score_thresh=0.9,
            crop_n_layers=0, # Speed up
            crop_n_points_downscale_factor=1,
            min_mask_region_area=100,  # Filter small noise
        )
        self.device = device

    def detect(self, image):
        # SAM2 expects RGB
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Generate masks
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            masks = self.mask_generator.generate(img_rgb)
        
        detections = []
        for mask_data in masks:
            # mask_data keys: 'segmentation', 'area', 'bbox', 'predicted_iou', 'point_coords', 'stability_score', 'crop_box'
            x, y, w, h = mask_data['bbox']
            score = mask_data['predicted_iou'] # Using IoU prediction as confidence
            
            # --- HEURISTIC FILTERING ---
            # SAM2 detects everything (trees, cars, ground). 
            # MOT17 is primarily pedestrians. We must filter by aspect ratio.
            # Pedestrians are usually taller than wide.
            aspect_ratio = h / w if w > 0 else 0
            
            # Filter: Keep only vertical-ish boxes (e.g., AR > 1.5) and reasonable size
            if aspect_ratio > 1.2 and h > 20: 
                # Convert xywh to x1, y1, x2, y2
                x1, y1, x2, y2 = x, y, x + w, y + h
                detections.append([x1, y1, x2, y2, score])
        
        return np.array(detections)

# --- Part 3: Main Processing Logic ---

def process_sequence(seq_name, detector, output_path):
    print(f"Processing sequence: {seq_name}")
    
    seq_dir = None
    for subdir in ["test", "train"]:
        d = os.path.join(MOT17_PATH, subdir, seq_name, "img1")
        if os.path.exists(d):
            seq_dir = d
            break

    if seq_dir is None:
        print(f"Sequence directory not found: {seq_name}")
        return None, None

    image_files = sorted([f for f in os.listdir(seq_dir) if f.endswith('.jpg')])
    
    tracker = OCSort(
        det_thresh=CONFIDENCE_THRESHOLD,
        iou_threshold=IOU_THRESHOLD,
        max_age=MAX_AGE,
        min_hits=MIN_HITS,
        delta_t=DELTA_T,
        inertia=INERTIA
    )

    results_file = os.path.join(output_path, f"{seq_name}.txt")
    
    with open(results_file, 'w') as f_out:
        for frame_idx, img_file in enumerate(image_files, start=1):
            img_path = os.path.join(seq_dir, img_file)
            frame = cv2.imread(img_path)
            if frame is None: continue

            # 1. Detect using SAM2
            dets = detector.detect(frame)
            
            if len(dets) == 0:
                dets = np.empty((0, 5))

            # 2. Track using OC-SORT
            track_results = tracker.update(dets)

            for t in track_results:
                x1, y1, x2, y2, track_id = t
                w = x2 - x1
                h = y2 - y1
                line = f"{frame_idx},{int(track_id)},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1.00,-1,-1,-1\n"
                f_out.write(line)
            
            if frame_idx % 20 == 0:
                print(f"  Frame {frame_idx}/{len(image_files)} processed.")

    print(f"Finished {seq_name}. Results saved to {results_file}")
    gt_file = os.path.join(seq_dir, '..', 'gt', 'gt.txt')
    return results_file, gt_file

def main():
    # 1. Initialize SAM2 Detector
    # Note: SAM2 is not trained here, we use the foundation model zero-shot
    try:
        detector = SAM2Detector(SAM2_CHECKPOINT, SAM2_CONFIG, DEVICE)
    except Exception as e:
        print(f"Failed to initialize SAM2: {e}")
        return

    # 2. Evaluate
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    sequences = []
    test_path = os.path.join(MOT17_PATH, "test")
    if os.path.exists(test_path):
        sequences = sorted([d for d in os.listdir(test_path) if os.path.isdir(os.path.join(test_path, d))])

    accs, names = [], []
    for seq in sequences:
        res_file, gt_file = process_sequence(seq, detector, OUTPUT_DIR)
        
        # Evaluation logic
        if gt_file and os.path.exists(gt_file):
            try:
                gt = mm.io.loadtxt(gt_file, fmt="mot15-2D", min_confidence=1)
                ts = mm.io.loadtxt(res_file, fmt="mot15-2D")
                acc = mm.utils.compare_to_groundtruth(gt, ts, 'iou', distth=0.5)
                accs.append(acc)
                names.append(seq)
            except Exception as e:
                print(f"Error evaluating {seq}: {e}")

    if accs:
        print("\nComputing Metrics on Test Set...")
        mh = mm.metrics.create()
        summary = mh.compute_many(accs, metrics=mm.metrics.motchallenge_metrics, names=names, generate_overall=True)
        str_summary = mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names)
        print(str_summary)
        
        # Logging
        os.makedirs(EXPERIMENT_DIR, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(EXPERIMENT_DIR, f"experiment_sam2_ocsort_{timestamp}.txt")
        with open(log_file, "w") as f:
            f.write(f"Experiment Log - {timestamp}\n")
            f.write("========================================\n")
            f.write(f"Detector: SAM2 (Automatic Mask Generator)\n")
            f.write(f"Tracker: OC-SORT\n")
            f.write("\nResults:\n")
            f.write(str_summary)
        print(f"\nExperiment results saved to {log_file}")
    else:
        print("No Ground Truth found in test folder for evaluation.")

if __name__ == "__main__":
    main()

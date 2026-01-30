import os
import shutil
import yaml
import datetime
import cv2
import numpy as np
import motmetrics as mm
from ultralytics import YOLO

# Dependencies for OC-SORT
try:
    from filterpy.kalman import KalmanFilter
except ImportError:
    raise ImportError("filterpy not found. Please install: pip install filterpy")

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    raise ImportError("scipy not found. Please install: pip install scipy")

# --- Configuration ---
MOT17_PATH = "MOT17"
YOLO_DATASET_DIR = "mot17_yolo_data_multiclass"  # Shared with train_yolo_norfair
EXPERIMENT_DIR = "experiments"
OUTPUT_DIR = "tracking_results_trained_ocsort"

# Training Config
BASE_MODEL = "yolo11l.pt"
EPOCHS = 10
IMG_SIZE = 640
BATCH_SIZE = 4

# OC-SORT Hyperparameters
CONFIDENCE_THRESHOLD = 0.4
IOU_THRESHOLD = 0.3
MAX_AGE = 30
MIN_HITS = 3
INERTIA = 0.2
DELTA_T = 3

# --- Part 1: Data Preparation (MOT17 GT -> YOLO Labels) ---

def convert_mot_to_yolo():
    print("Preparing YOLO training data from MOT17...")
    
    # Create dataset structure
    images_train_dir = os.path.join(YOLO_DATASET_DIR, "images", "train")
    labels_train_dir = os.path.join(YOLO_DATASET_DIR, "labels", "train")
    
    if os.path.exists(YOLO_DATASET_DIR):
        print(f"Dataset directory {YOLO_DATASET_DIR} exists. Skipping conversion.")
        return
    
    os.makedirs(images_train_dir, exist_ok=True)
    os.makedirs(labels_train_dir, exist_ok=True)

    train_path = os.path.join(MOT17_PATH, "train")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"MOT17 train directory not found at {train_path}")

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
        gt_data = np.loadtxt(gt_file, delimiter=',')
        
        # Get image dimensions
        first_img_name = sorted(os.listdir(img_dir))[0]
        first_img = cv2.imread(os.path.join(img_dir, first_img_name))
        img_h, img_w = first_img.shape[:2]

        frames = np.unique(gt_data[:, 0])
        
        for frame_idx in frames:
            frame_rows = gt_data[gt_data[:, 0] == frame_idx]
            
            yolo_lines = []
            for row in frame_rows:
                cls_id = int(row[7])
                vis = row[8]
                if vis < 0.2: continue

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

    # Create YAML
    yaml_content = {
        'path': os.path.abspath(YOLO_DATASET_DIR),
        'train': 'images/train',
        'val': 'images/train',
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
    
    results = model.train(
        data=yaml_path,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        project=EXPERIMENT_DIR,
        name="yolo_mot17_finetune",
        exist_ok=True,
        verbose=True
    )
    
    best_weight = os.path.join(EXPERIMENT_DIR, "yolo_mot17_finetune", "weights", "best.pt")
    print(f"Training finished. Best weights at: {best_weight}")
    return best_weight

# --- Part 3: OC-SORT Implementation ---

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

            yolo_results = model(frame, verbose=False)
            
            dets = []
            for result in yolo_results:
                boxes = result.boxes.data.cpu().numpy()
                for box in boxes:
                    x1, y1, x2, y2, conf, cls = box
                    # Filter for Pedestrian (Class 0 in trained model)
                    if int(cls) == 0: 
                        dets.append([x1, y1, x2, y2, conf])
            
            dets = np.array(dets)
            if len(dets) == 0:
                dets = np.empty((0, 5))

            track_results = tracker.update(dets)

            for t in track_results:
                x1, y1, x2, y2, track_id = t
                w = x2 - x1
                h = y2 - y1
                line = f"{frame_idx},{int(track_id)},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1.00,-1,-1,-1\n"
                f_out.write(line)

    print(f"Finished {seq_name}. Results saved to {results_file}")
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
        os.makedirs(EXPERIMENT_DIR, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(EXPERIMENT_DIR, f"experiment_trained_ocsort_{timestamp}.txt")
        with open(log_file, "w") as f:
            f.write(f"Experiment Log - {timestamp}\n")
            f.write("========================================\n")
            f.write(f"Model: {BASE_MODEL} (Finetuned on MOT17 all classes)\n")
            f.write(f"Tracker: OC-SORT (Custom Implementation)\n")
            f.write(f"Hyperparameters:\n")
            f.write(f"  EPOCHS: {EPOCHS}\n")
            f.write(f"  IMG_SIZE: {IMG_SIZE}\n")
            f.write(f"  BATCH_SIZE: {BATCH_SIZE}\n")
            f.write(f"  IOU_THRESH: {IOU_THRESHOLD}\n")
            f.write(f"  MAX_AGE: {MAX_AGE}\n")
            f.write(f"  MIN_HITS: {MIN_HITS}\n")
            f.write(f"  INERTIA: {INERTIA}\n")
            f.write("\nResults:\n")
            f.write(str_summary)
        print(f"\nExperiment results saved to {log_file}")
    else:
        print("No Ground Truth found in test folder for evaluation.")

if __name__ == "__main__":
    main()

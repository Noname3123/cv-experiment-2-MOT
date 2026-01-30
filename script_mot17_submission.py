import os
import datetime
import zipfile
import glob
import cv2
import time
import numpy as np

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
MOT17_PATH = "MOT17_submission_dataset"
LOG_DIR = "experiments"
SUBMISSION_DIR = "mot17_submission"

# OC-SORT Hyperparameters
CONFIDENCE_THRESHOLD = 0.5    # Initial detection confidence threshold
CONFIDENCE_LOW = 0.05         # Low threshold for maintaining tracks (ByteTrack)
INFERENCE_SIZE = 1280         # This is no longer used for detection, but kept for reference
IOU_THRESHOLD = 0.3
MAX_AGE = 60                  # Tracking age
MIN_HITS = 3
INERTIA = 0.2
DELTA_T = 3


# --- OC-SORT Implementation ---

def get_camera_motion(prev_img, curr_img):
    """
    Computes the affine transformation (camera motion) between two frames.
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

def iou_batch(bb_test, bb_gt):
    """
    Computes IOU between two bboxes in the form [x1,y1,x2,y2]
    """
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
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
    [cx,cy,s,r] where cx,cy is the center of the box and s is the scale/area and r is
    the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w/2.
    y = bbox[1] + h/2.
    s = w * h
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))

def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the center form [cx,cy,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
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
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0
    def __init__(self, bbox, delta_t=3):
        # define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4) 
        self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],  
                            [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
        self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])

        self.kf.R[2:,2:] *= 10.
        self.kf.P[4:,4:] *= 1000. # give high uncertainty to the unobservable initial velocities
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
        
        # OC-SORT specific
        self.last_observation = convert_bbox_to_z(bbox)
        self.observations = dict()
        self.history_observations = []
        self.velocity = None
        self.delta_t = delta_t

    def update(self, bbox):
        """
        Updates the state vector with observed bbox.
        """
        if bbox is not None:
            if self.last_observation.sum() >= 0: # if previous observation exists
                previous_box = None
                for i in range(self.delta_t):
                    dt = self.delta_t - i
                    if self.age - dt in self.observations:
                        previous_box = self.observations[self.age - dt]
                        break
                
                if previous_box is None:
                    previous_box = self.last_observation
                
                # Estimate velocity
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
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if((self.kf.x[6]+self.kf.x[2])<=0):
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if(self.time_since_update>0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def apply_affine_correction(self, warp):
        """
        Applies camera motion correction to the Kalman Filter state.
        warp: 2x3 affine transformation matrix
        """
        # 1. Transform Position (cx, cy)
        p = np.array([self.kf.x[0, 0], self.kf.x[1, 0], 1.0]).reshape(3, 1)
        p_new = warp @ p
        self.kf.x[0, 0] = p_new[0, 0]
        self.kf.x[1, 0] = p_new[1, 0]
        
        # 2. Transform Velocity (vx, vy) - only rotation/scale affects velocity
        v = np.array([self.kf.x[4, 0], self.kf.x[5, 0]]).reshape(2, 1)
        v_new = warp[:, :2] @ v
        self.kf.x[4, 0] = v_new[0, 0]
        self.kf.x[5, 0] = v_new[1, 0]
        
        # 3. Transform Scale (s) - Area scales by square of linear scale
        scale = np.sqrt(warp[0, 0]**2 + warp[1, 0]**2)
        self.kf.x[2, 0] *= (scale ** 2)


class OCSort(object):
    def __init__(self, det_thresh, det_thresh_low, max_age=30, min_hits=3, iou_threshold=0.3, delta_t=3, inertia=0.2):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
        self.det_thresh = det_thresh
        self.det_thresh_low = det_thresh_low
        self.delta_t = delta_t
        self.inertia = inertia
        KalmanBoxTracker.count = 0

    def apply_cmc(self, warp):
        """
        Apply Camera Motion Compensation to all active trackers.
        """
        for trk in self.trackers:
            trk.apply_affine_correction(warp)

    def update(self, output_results):
        """
        Params:
          output_results - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Returns:
          a numpy array of tracked objects in the format [[x1,y1,x2,y2,ID],[x1,y1,x2,y2,ID],...]
        """
        self.frame_count += 1
        
        # Split detections into High and Low confidence (ByteTrack strategy)
        if output_results.shape[0] == 0:
            dets_high = np.empty((0, 5))
            dets_low = np.empty((0, 5))
        else:
            dets_high = output_results[output_results[:, 4] >= self.det_thresh]
            dets_low = output_results[(output_results[:, 4] >= self.det_thresh_low) & (output_results[:, 4] < self.det_thresh)]

        # Get predicted locations from existing trackers.
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

        # Associate high confidence detections
        matched, unmatched_dets, unmatched_trks = self.associate(dets_high, trks, self.iou_threshold, velocities, k_observations, self.inertia)
        
        for m in matched:
            self.trackers[m[1]].update(dets_high[m[0], :])

        # Associate low confidence detections (ByteTrack logic)
        
        if len(dets_low) > 0 and len(unmatched_trks) > 0:
            
            u_trks = trks[unmatched_trks]
            iou_matrix = iou_batch(dets_low, u_trks)
            
            # Hungarian matching on IOU
            if iou_matrix.max() > 0.1: # Loose threshold for low conf
                matched_indices = linear_sum_assignment(-iou_matrix)
                matched_indices = np.array(list(zip(matched_indices[0], matched_indices[1])))
                
                to_remove_trks = []
                for m in matched_indices:
                    det_idx = m[0]
                    trk_idx_in_unmatched = m[1]
                    real_trk_idx = unmatched_trks[trk_idx_in_unmatched]
                    
                    if iou_matrix[det_idx, trk_idx_in_unmatched] < self.iou_threshold: # Use global IOU threshold
                        continue
                        
                    self.trackers[real_trk_idx].update(dets_low[det_idx, :])
                    to_remove_trks.append(trk_idx_in_unmatched)
                
                unmatched_trks = np.delete(unmatched_trks, to_remove_trks)

        # OCM (Observation Centric Momentum) Recovery
        # recover tracks that were lost but consistent with motion
        if len(unmatched_dets) > 0 and len(unmatched_trks) > 0:
            left_dets = dets_high[unmatched_dets]
            left_trks = last_boxes[unmatched_trks]
            
            # propagate last observation using velocity
            iou_left = iou_batch(left_dets, convert_x_to_bbox(left_trks + velocities[unmatched_trks]))
            iou_left = np.array(iou_left)
            
            if iou_left.max() > self.iou_threshold:
                # Re-associate
                matched_indices = linear_sum_assignment(-iou_left)
                matched_indices = np.array(list(zip(matched_indices[0], matched_indices[1])))
                
                to_remove_dets = []
                to_remove_trks = []
                
                for m in matched_indices:
                    det_idx = unmatched_dets[m[0]]
                    trk_idx = unmatched_trks[m[1]]
                    
                    if iou_left[m[0], m[1]] < self.iou_threshold:
                        continue
                        
                    self.trackers[trk_idx].update(dets_high[det_idx, :])
                    to_remove_dets.append(m[0])
                    to_remove_trks.append(m[1])
                
                unmatched_dets = np.delete(unmatched_dets, to_remove_dets)
                unmatched_trks = np.delete(unmatched_trks, to_remove_trks)

        for m in unmatched_trks:
            self.trackers[m].update(None)

        # new trackers only from unmatched high confidence detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets_high[i,:])
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
            # remove dead tracklet
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
                # Add consistency cost
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

        # Filter matches with low IOU
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


# --- Post-Processing corrections ---

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


# --- Main Logic ---

def process_sequence(seq_name, output_path):
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

    # Load all detections for this sequence from the provided det.txt
    det_file = os.path.join(seq_dir, '..', 'det', 'det.txt')
    all_dets = {}
    if os.path.exists(det_file):
        data = np.loadtxt(det_file, delimiter=',')
        for row in data:
            frame_idx = int(row[0])
            if frame_idx not in all_dets:
                all_dets[frame_idx] = []
            # Format: [x1, y1, w, h, conf] -> [x1, y1, x2, y2, conf]
            x1, y1, w, h, conf = row[2:7]
            all_dets[frame_idx].append([x1, y1, x1 + w, y1 + h, conf])
    for frame_idx in all_dets:
        all_dets[frame_idx] = np.array(all_dets[frame_idx])
    
    # Initialize OC-SORT 
    tracker = OCSort(
        det_thresh=CONFIDENCE_THRESHOLD,
        det_thresh_low=CONFIDENCE_LOW,
        iou_threshold=IOU_THRESHOLD,
        max_age=MAX_AGE,
        min_hits=MIN_HITS,
        delta_t=DELTA_T,
        inertia=INERTIA
    )

    results_file = os.path.join(output_path, f"{seq_name}.txt")
    
    raw_results = []
    prev_gray = None
    
    start_time = time.time()
    
    for frame_idx, img_file in enumerate(image_files, start=1):
        img_path = os.path.join(seq_dir, img_file)
        frame = cv2.imread(img_path)
        if frame is None: continue
        
        # Camera Motion Compensation
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            warp = get_camera_motion(prev_gray, gray)
            tracker.apply_cmc(warp)
        prev_gray = gray

        # Get pre-loaded detections for the current frame
        dets = all_dets.get(frame_idx, np.empty((0, 5)))

        # Update Tracker
        track_results = tracker.update(dets)

        # Store Results 
        for t in track_results:
            x1, y1, x2, y2, track_id = t
            w = x2 - x1
            h = y2 - y1
            raw_results.append([frame_idx, int(track_id), x1, y1, w, h])

    end_time = time.time()
    total_time = end_time - start_time
    num_frames = len(image_files)
    fps = num_frames / total_time if total_time > 0 else 0

    # Interpolate Missing Frames
    interpolated_results = interpolate_tracks(raw_results, max_gap=20)

    # Write detects to File
    with open(results_file, 'w') as f_out:
        for row in interpolated_results:
            frame_idx, track_id, x1, y1, w, h = row
            # MOT Format: frame, id, left, top, width, height, conf, -1, -1, -1
            # Using -1 for confidence to match the example and ensure compatibility
            line = f"{frame_idx}, {track_id}, {x1:.2f}, {y1:.2f}, {w:.2f}, {h:.2f}, -1, -1, -1, -1\n"
            f_out.write(line)

    print(f"Finished {seq_name}. Results saved to {results_file}")
    print(f"  -> Processing speed: {fps:.2f} FPS (Hz)")
    return results_file, fps

def main():
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    print("Starting MOT17 Test Set Processing for Benchmark Submission")

    sequences = []
    test_path = os.path.join(MOT17_PATH, "test")
    if os.path.exists(test_path):
        sequences = sorted([d for d in os.listdir(test_path) if os.path.isdir(os.path.join(test_path, d))])
    
    if not sequences:
        print(f"Error: No sequences found in '{test_path}'. Please ensure the MOT17 test set is structured correctly.")
        return

    print(f"Found {len(sequences)} sequences: {sequences}")

    result_files = []
    all_fps = []
    for seq in sequences:
        res_file, fps = process_sequence(seq, SUBMISSION_DIR)
        if res_file:
            result_files.append(res_file)
            if fps is not None:
                all_fps.append(fps)

    # Create submission zip file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if result_files:
        submission_zip_file = f"submission_{timestamp}.zip"
        print(f"\nCreating submission file: {submission_zip_file}")
        with zipfile.ZipFile(submission_zip_file, 'w') as zf:
            for res_file in result_files:
                # Add file to the root of the zip archive
                zf.write(res_file, os.path.basename(res_file))
        print(f"Submission zip created successfully: {submission_zip_file}")

    avg_fps = 0
    if all_fps:
        avg_fps = np.mean(all_fps)
        print(f"\nAverage processing speed: {avg_fps:.2f} FPS (Hz)")

    # Log experiment details
    os.makedirs(LOG_DIR, exist_ok=True)
    log_file = os.path.join(LOG_DIR, f"submission_log_{timestamp}.txt")
    with open(log_file, "w") as f:
        f.write(f"MOT17 Submission Log - {timestamp}\n")
        f.write("========================================\n")
        f.write(f"Average Speed (Hz): {avg_fps:.2f}\n")
        f.write(f"Tracker: OC-SORT + ByteTrack + Interpolation + CMC\n")
        f.write("Hyperparameters:\n")
        f.write(f"  CONFIDENCE_THRESHOLD (High): {CONFIDENCE_THRESHOLD}\n")
        f.write(f"  CONFIDENCE_LOW: {CONFIDENCE_LOW}\n")
        f.write(f"  IOU_THRESHOLD: {IOU_THRESHOLD}\n")
        f.write(f"  MAX_AGE: {MAX_AGE}\n")
        f.write(f"  MIN_HITS: {MIN_HITS}\n")
        f.write(f"  INERTIA: {INERTIA}\n")
        f.write(f"  POST_PROCESSING: Linear Interpolation (max_gap=20)\n")
        f.write(f"  CMC: Optical Flow (Affine)\n")
    print(f"Experiment log saved to {log_file}")

if __name__ == "__main__":
    main()

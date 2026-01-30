import os
import datetime
import cv2
import sys
import numpy as np
import traceback
import torch
from ultralytics import YOLO
import motmetrics as mm
from scipy.optimize import linear_sum_assignment

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "samurai", "sam2"))
from sam2.build_sam import build_sam2_video_predictor

NO_OBJ_SCORE = -1024.0

# --- Configuration ---
MOT17_PATH = "MOT17"
EXPERIMENT_DIR = "experiments"
OUTPUT_DIR = "tracking_results_samurai"
MODEL_WEIGHTS = "yolo11l.pt"

# Tracking Hyperparameters
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.3
MAX_LOST_FRAMES = 10
SAMURAI_CHECKPOINT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samurai/sam2/checkpoints/sam2.1_hiera_large.pt")
SAMURAI_CONFIG = "configs/samurai/sam2.1_hiera_l.yaml"

# --- SAMURAI Wrapper ---
class SamuraiWrapper:
    """
    Lightweight wrapper to handle a single object track ID.
    The heavy lifting is done by the shared predictor in MultiObjectManager.
    """
    def __init__(self, obj_id, bbox):
        self.id = obj_id
        self.bbox = bbox # [x1, y1, x2, y2]
        self.lost = 0

    def update(self, bbox):
        """
        Update the tracker state with a new bounding box.
        """
        self.bbox = bbox

# --- Utilities ---

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

def pad_inference_state(inference_state, target_obj_num):
    """
    Pad the output_dict tensors in inference_state to match the new target_obj_num.
    This is necessary because SAM2 expects consistent batch size across memory frames.
    """
    for storage_key in ["cond_frame_outputs", "non_cond_frame_outputs"]:
        for frame_idx, out in inference_state["output_dict"][storage_key].items():
            current_b = out["obj_ptr"].shape[0]
            if current_b < target_obj_num:
                diff = target_obj_num - current_b
                device = out["obj_ptr"].device
                
                # Pad obj_ptr with NO_OBJ_SCORE
                pad_ptr = torch.full((diff, out["obj_ptr"].shape[1]), NO_OBJ_SCORE, device=device, dtype=out["obj_ptr"].dtype).to(device)
                out["obj_ptr"] = torch.cat([out["obj_ptr"], pad_ptr], dim=0).to(device)
                
                # Pad pred_masks
                if "pred_masks" in out:
                    pad_mask = torch.full((diff, *out["pred_masks"].shape[1:]), NO_OBJ_SCORE, device=device, dtype=out["pred_masks"].dtype).to(device)
                    out["pred_masks"] = torch.cat([out["pred_masks"], pad_mask], dim=0).to(device)

                # Pad pred_masks_high_res
                if "pred_masks_high_res" in out:
                    pad_mask_hr = torch.full((diff, *out["pred_masks_high_res"].shape[1:]), NO_OBJ_SCORE, device=device, dtype=out["pred_masks_high_res"].dtype).to(device)
                    out["pred_masks_high_res"] = torch.cat([out["pred_masks_high_res"], pad_mask_hr], dim=0).to(device)

                # Pad object_score_logits
                if "object_score_logits" in out:
                    pad_logits = torch.full((diff, 1), -10.0, device=device, dtype=out["object_score_logits"].dtype).to(device)
                    out["object_score_logits"] = torch.cat([out["object_score_logits"], pad_logits], dim=0).to(device)
                
                # Pad maskmem_features
                if out.get("maskmem_features") is not None:
                    pad_mem = torch.zeros((diff, *out["maskmem_features"].shape[1:]), device=device, dtype=out["maskmem_features"].dtype).to(device)
                    out["maskmem_features"] = torch.cat([out["maskmem_features"], pad_mem], dim=0).to(device)
                
                # Pad maskmem_pos_enc (list of tensors)
                if out.get("maskmem_pos_enc") is not None:
                    # Expand the first element to create padding (assuming identical pos enc across batch)
                    out["maskmem_pos_enc"] = [torch.cat([pe, pe[0:1].expand(diff, -1, -1, -1)], dim=0).to(device) for pe in out["maskmem_pos_enc"]]

class MultiObjectManager:
    def __init__(self, predictor, state):
        self.predictor = predictor
        self.state = state
        self.trackers = {} # Dict of obj_id -> SamuraiWrapper
        self.next_id = 1

    def update(self, frame_idx, detections):
        """
        1. Predict all existing tracks using SAMURAI.
        2. Match with YOLO detections to confirm tracks and find new objects.
        3. Spawn new SAMURAI trackers for new objects.
        """
        
        # 1. Predict using SAMURAI (Propagate 1 frame)
        # We use propagate_in_video starting from current frame to get predictions
        # Note: frame_idx is 0-based for SAM2
        pred_boxes_map = {}
        
        if len(self.trackers) > 0:
            # Run inference for just this frame
            try:
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                    for _, object_ids, masks in self.predictor.propagate_in_video(
                        self.state, 
                        start_frame_idx=frame_idx, 
                        max_frame_num_to_track=1
                    ):
                        for obj_id, mask in zip(object_ids, masks):
                            # Convert mask to bbox
                            mask = mask[0].cpu().numpy() > 0.0
                            if mask.any():
                                y_min, x_min = np.argwhere(mask).min(axis=0)
                                y_max, x_max = np.argwhere(mask).max(axis=0)
                                bbox = [x_min, y_min, x_max, y_max] # x1, y1, x2, y2
                                pred_boxes_map[obj_id] = bbox
                                
                                # Update internal tracker state
                                if obj_id in self.trackers:
                                    self.trackers[obj_id].update(bbox)
                        break # Stop after 1 frame
            except Exception as e:
                print(f"Error during SAMURAI propagation at frame {frame_idx}: {e}")
                traceback.print_exc()
                # Continue to allow YOLO to potentially recover or re-init

        # Collect predicted boxes for association
        predicted_boxes = []
        active_track_ids = []
        
        for obj_id, trk in self.trackers.items():
            if obj_id in pred_boxes_map:
                predicted_boxes.append(pred_boxes_map[obj_id])
                active_track_ids.append(obj_id)
        
        predicted_boxes = np.array(predicted_boxes) # [N, 4]
        if len(predicted_boxes) == 0:
            predicted_boxes = np.empty((0, 4))

        # 2. Associate with Detections (Hungarian Algorithm)
        if len(detections) > 0 and len(predicted_boxes) > 0:
            iou_matrix = iou_batch(detections[:, :4], predicted_boxes)
            row_ind, col_ind = linear_sum_assignment(-iou_matrix)
            
            matches = []
            for r, c in zip(row_ind, col_ind):
                if iou_matrix[r, c] >= IOU_THRESHOLD:
                    matches.append((r, c))
            
            matched_det_indices = {m[0] for m in matches}
            matched_trk_indices = {m[1] for m in matches}
            
            unmatched_dets = [d for d in range(len(detections)) if d not in matched_det_indices]
            unmatched_trks = [t for t in range(len(self.trackers)) if t not in matched_trk_indices]
        else:
            # If no predictions (first frame or lost), everything is unmatched
            matches = []
            unmatched_dets = list(range(len(detections)))
            # All active tracks that didn't get a match are unmatched
            # But wait, we need indices into the 'predicted_boxes' array, not obj_ids
            unmatched_trks = list(range(len(predicted_boxes)))

        # 3. Handle Matches
        # Reset lost counter for matched tracks
        for det_idx, trk_idx in matches:
            obj_id = active_track_ids[trk_idx]
            self.trackers[obj_id].lost = 0
            # Optional: We could refine SAM2 state with YOLO box here if API supported it easily

        # 4. Handle Unmatched Tracks (Lost)
        for trk_idx in unmatched_trks:
            obj_id = active_track_ids[trk_idx]
            self.trackers[obj_id].lost += 1

        # 5. Handle Unmatched Detections (New Tracks)
        for det_idx in unmatched_dets:
            bbox = detections[det_idx, :4]
            obj_id = self.next_id
            self.next_id += 1
            
            # Initialize new track in SAMURAI
            # SAM2 expects [x1, y1, x2, y2]
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                # Hack: Temporarily reset tracking_has_started to allow adding new objects
                # This bypasses the check in SAM2VideoPredictor.add_new_points_or_box
                was_tracking = self.state["tracking_has_started"]
                self.state["tracking_has_started"] = False
                
                # Hack 2: Temporarily remove frame from frames_already_tracked so it's treated as init
                # This ensures we don't try to condition on memory for a brand new object
                was_tracked_meta = self.state["frames_already_tracked"].pop(frame_idx, None)

                self.predictor.add_new_points_or_box(
                    self.state, 
                    box=bbox, 
                    frame_idx=frame_idx, 
                    obj_id=obj_id
                )
                self.state["tracking_has_started"] = was_tracking
                if was_tracked_meta is not None:
                    self.state["frames_already_tracked"][frame_idx] = was_tracked_meta
                
                # Pad previous frames' outputs to match the new total object count
                # This prevents "stack expects each tensor to be equal size" error
                pad_inference_state(self.state, len(self.state["obj_ids"]))
            
            self.trackers[obj_id] = SamuraiWrapper(obj_id, bbox)

        # 6. Remove Dead Tracks
        self.trackers = {oid: t for oid, t in self.trackers.items() if t.lost <= MAX_LOST_FRAMES}

        # Return results [x1, y1, x2, y2, id]
        results = []
        for obj_id, trk in self.trackers.items():
            if trk.lost == 0:
                bbox = trk.bbox
                results.append([bbox[0], bbox[1], bbox[2], bbox[3], obj_id])
        
        return np.array(results)

# --- Main Processing ---

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

    # Initialize SAMURAI Predictor for this sequence
    print(f"Initializing SAMURAI for {seq_name}...")
    predictor = build_sam2_video_predictor(SAMURAI_CONFIG, SAMURAI_CHECKPOINT, device="cuda:0")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        state = predictor.init_state(seq_dir, offload_video_to_cpu=True, offload_state_to_cpu=True)

    image_files = sorted([f for f in os.listdir(seq_dir) if f.endswith('.jpg')])
    
    # Initialize Manager
    manager = MultiObjectManager(predictor, state)

    results_file = os.path.join(output_path, f"{seq_name}.txt")
    
    with open(results_file, 'w') as f_out:
        for frame_idx, img_file in enumerate(image_files, start=1):
            img_path = os.path.join(seq_dir, img_file)
            frame = cv2.imread(img_path)
            if frame is None: continue

            # 1. Run YOLO Detection
            yolo_results = model(frame, verbose=False)
            
            dets = []
            for result in yolo_results:
                boxes = result.boxes.data.cpu().numpy()
                for box in boxes:
                    x1, y1, x2, y2, conf, cls = box
                    if int(cls) == 0 and conf >= CONFIDENCE_THRESHOLD: 
                        dets.append([x1, y1, x2, y2, conf])
            dets = np.array(dets)

            # 2. Update SAMURAI Trackers
            # Pass 0-based frame index
            track_results = manager.update(frame_idx - 1, dets)

            # 3. Save Results
            for t in track_results:
                x1, y1, x2, y2, track_id = t
                w = x2 - x1
                h = y2 - y1
                line = f"{frame_idx},{int(track_id)},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1.00,-1,-1,-1\n"
                f_out.write(line)

    # Cleanup
    del predictor
    del state
    torch.cuda.empty_cache()

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
        try:
            res_file, gt_file = process_sequence(seq, model, OUTPUT_DIR)
            if gt_file and os.path.exists(gt_file):
                gt = mm.io.loadtxt(gt_file, fmt="mot15-2D", min_confidence=1)
                ts = mm.io.loadtxt(res_file, fmt="mot15-2D")
                acc = mm.utils.compare_to_groundtruth(gt, ts, 'iou', distth=0.5)
                accs.append(acc)
                names.append(seq)
        except Exception as e:
            print(f"Error processing {seq}: {e}")
            traceback.print_exc()
            print("Ensure SAMURAI is installed and the SamuraiWrapper class is implemented correctly.")

    if accs:
        print("\nComputing Metrics on Test Set...")
        mh = mm.metrics.create()
        summary = mh.compute_many(accs, metrics=mm.metrics.motchallenge_metrics, names=names, generate_overall=True)
        str_summary = mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names)
        print(str_summary)
        
        # Logging
        os.makedirs(EXPERIMENT_DIR, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(EXPERIMENT_DIR, f"experiment_samurai_{timestamp}.txt")
        with open(log_file, "w") as f:
            f.write(f"Experiment Log - {timestamp}\n")
            f.write("========================================\n")
            f.write(f"Model: {MODEL_WEIGHTS}\n")
            f.write(f"Tracker: SAMURAI (SOT-based MOT)\n")
            f.write("\nResults:\n")
            f.write(str_summary)
        print(f"\nExperiment results saved to {log_file}")
    else:
        print("No Ground Truth found in test folder for evaluation.")

if __name__ == "__main__":
    main()

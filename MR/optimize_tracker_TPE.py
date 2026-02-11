import os
import sys
import logging
import numpy as np
import motmetrics as mm
import optuna
from datetime import datetime

# Import configuration and tracker logic
# Ensure we can import from the same directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from optimization_config import SEARCH_SPACE
import evaluate_tracker_cached as tracker_module

# --- Configuration ---
MOT17_PATH = "MOT17"
STUDY_NAME = "ocsort_optimization_tpe"
STORAGE_DIR = "optimization_results"
N_TRIALS = 300  # Number of optimization trials

# Setup Logging
os.makedirs(STORAGE_DIR, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(STORAGE_DIR, f"optimization_{timestamp}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def get_validation_sequences(mot_path):
    """
    Finds sequences that have both cached detections and Ground Truth.
    """
    sequences = []
    # Check both train and test folders
    for subdir in ["train", "test"]:
        path = os.path.join(mot_path, subdir)
        if not os.path.exists(path): continue
        
        for seq in os.listdir(path):
            seq_path = os.path.join(path, seq)
            if not os.path.isdir(seq_path): continue
            
            # Check for cached detections
            det_path = os.path.join(seq_path, f"{seq}_yolo_detections.npy")
            # Check for Ground Truth
            gt_path = os.path.join(seq_path, "gt", "gt.txt")
            
            if os.path.exists(det_path) and os.path.exists(gt_path):
                sequences.append((seq, det_path, gt_path))
    
    return sorted(sequences)

def objective(trial):
    # 1. Sample Hyperparameters based on config
    params = {}
    for key, config in SEARCH_SPACE.items():
        if config["type"] == "float":
            params[key] = trial.suggest_float(key, config["low"], config["high"])
        elif config["type"] == "int":
            params[key] = trial.suggest_int(key, config["low"], config["high"])
        elif config["type"] == "categorical":
            params[key] = trial.suggest_categorical(key, config["choices"])

    # Constraint: Low threshold must be lower than High threshold
    if params["CONFIDENCE_LOW"] >= params["CONFIDENCE_THRESHOLD"]:
        # Prune invalid trials early
        raise optuna.TrialPruned("CONFIDENCE_LOW >= CONFIDENCE_THRESHOLD")

    # 2. Run Tracker on all validation sequences
    accs = []
    names = []
    
    # Locate MOT17
    mot_path = MOT17_PATH
    if not os.path.exists(mot_path) and os.path.exists(os.path.join("..", MOT17_PATH)):
        mot_path = os.path.join("..", MOT17_PATH)
        
    val_sequences = get_validation_sequences(mot_path)
    
    if not val_sequences:
        logger.error("No validation sequences found (need .npy detections AND gt.txt).")
        return 0.0

    # Create a temp dir for this trial's outputs to avoid race conditions/clutter
    trial_output_dir = os.path.join(STORAGE_DIR, f"trial_{trial.number}")
    os.makedirs(trial_output_dir, exist_ok=True)

    for seq_name, det_path, gt_path in val_sequences:
        # Load cached detections
        all_detections = np.load(det_path)
        
        # Run tracker with injected params
        res_file, _ = tracker_module.process_sequence(
            seq_name, 
            all_detections, 
            trial_output_dir, 
            mot_path, 
            params=params
        )
        
        # Evaluate
        gt = mm.io.loadtxt(gt_path, fmt="mot15-2D", min_confidence=1)
        ts = mm.io.loadtxt(res_file, fmt="mot15-2D")
        acc = mm.utils.compare_to_groundtruth(gt, ts, 'iou', distth=0.5)
        accs.append(acc)
        names.append(seq_name)

    # 3. Compute Aggregate Metrics
    mh = mm.metrics.create()
    summary = mh.compute_many(accs, metrics=['mota', 'idf1', 'num_switches'], names=names, generate_overall=True)
    
    overall_mota = summary.loc['OVERALL']['mota']
    overall_idf1 = summary.loc['OVERALL']['idf1']
    overall_idsw = summary.loc['OVERALL']['num_switches']
    
    # 4. Define Optimization Goal (Scalar)
    # We want to maximize MOTA and IDF1. 
    # Simple weighted sum: 50% MOTA + 50% IDF1
    score = (overall_mota + overall_idf1) / 2.0
    
    logger.info(f"Trial {trial.number}: Score={score:.4f} | MOTA={overall_mota:.4f} | IDF1={overall_idf1:.4f} | IDsw={overall_idsw} | Params={params}")
    
    return score

def main():
    logger.info("Starting Hyperparameter Optimization (TPE)...")
    
    # Check if we have data
    mot_path = MOT17_PATH
    if not os.path.exists(mot_path) and os.path.exists(os.path.join("..", MOT17_PATH)):
        mot_path = os.path.join("..", MOT17_PATH)
    seqs = get_validation_sequences(mot_path)
    logger.info(f"Found {len(seqs)} sequences for validation: {[s[0] for s in seqs]}")
    
    if not seqs:
        logger.error("Please run generate_yolo_detections.py first to create .npy files for sequences that have Ground Truth.")
        return

    # Create Study
    study = optuna.create_study(direction="maximize", study_name=STUDY_NAME)
    
    # Run Optimization
    study.optimize(objective, n_trials=N_TRIALS)
    
    logger.info("Optimization Finished.")
    logger.info(f"Best Trial: {study.best_trial.number}")
    logger.info(f"Best Score: {study.best_value}")
    logger.info(f"Best Params: {study.best_params}")

if __name__ == "__main__":
    main()

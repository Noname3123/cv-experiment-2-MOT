import os
import sys
import logging
import numpy as np
import motmetrics as mm
import optuna
import random
import math
from datetime import datetime

# Import configuration and tracker logic
# Ensure we can import from the same directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from optimization_config import SEARCH_SPACE
import evaluate_tracker_cached as tracker_module

# --- Configuration ---
MOT17_PATH = "MOT17"
STUDY_NAME = "ocsort_optimization_sa"
STORAGE_DIR = "optimization_results"
N_TRIALS = 500  # Number of optimization trials

# Setup Logging
os.makedirs(STORAGE_DIR, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(STORAGE_DIR, f"optimization_sa_{timestamp}.log")

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

def get_random_params(search_space):
    """Generates a random valid set of parameters."""
    params = {}
    for key, config in search_space.items():
        if config["type"] == "float":
            params[key] = random.uniform(config["low"], config["high"])
        elif config["type"] == "int":
            params[key] = random.randint(config["low"], config["high"])
        elif config["type"] == "categorical":
            params[key] = random.choice(config["choices"])
    
    # Enforce constraints
    if params.get("CONFIDENCE_LOW") is not None and params.get("CONFIDENCE_THRESHOLD") is not None:
        if params["CONFIDENCE_LOW"] >= params["CONFIDENCE_THRESHOLD"]:
            params["CONFIDENCE_LOW"] = params["CONFIDENCE_THRESHOLD"] * 0.5
    return params

def perturb_params(current_params, search_space, temperature):
    """Perturbs parameters to find a neighbor state."""
    new_params = current_params.copy()
    
    # Select one parameter to change
    key_to_change = random.choice(list(search_space.keys()))
    config = search_space[key_to_change]
    
    # Perturbation magnitude scales with temperature
    if config["type"] == "float":
        span = config["high"] - config["low"]
        sigma = span * 0.2 * max(temperature, 0.01)
        delta = random.gauss(0, sigma)
        new_val = new_params[key_to_change] + delta
        new_params[key_to_change] = max(config["low"], min(config["high"], new_val))
        
    elif config["type"] == "int":
        span = config["high"] - config["low"]
        max_step = max(1, int(span * 0.2 * max(temperature, 0.01)))
        delta = random.randint(-max_step, max_step)
        new_val = new_params[key_to_change] + delta
        new_params[key_to_change] = max(config["low"], min(config["high"], new_val))
        
    elif config["type"] == "categorical":
        choices = config["choices"]
        if len(choices) > 1:
            others = [c for c in choices if c != new_params[key_to_change]]
            new_params[key_to_change] = random.choice(others)

    # Enforce constraints
    if new_params.get("CONFIDENCE_LOW") is not None and new_params.get("CONFIDENCE_THRESHOLD") is not None:
        if new_params["CONFIDENCE_LOW"] >= new_params["CONFIDENCE_THRESHOLD"]:
            # Fix by lowering low threshold
            new_params["CONFIDENCE_LOW"] = new_params["CONFIDENCE_THRESHOLD"] - 0.01
            if new_params["CONFIDENCE_LOW"] < search_space["CONFIDENCE_LOW"]["low"]:
                new_params["CONFIDENCE_LOW"] = search_space["CONFIDENCE_LOW"]["low"]

    return new_params

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
    trial_output_dir = os.path.join(STORAGE_DIR, f"trial_sa_{trial.number}")
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
    logger.info("Starting Hyperparameter Optimization (Manual Simulated Annealing)...")
    
    # Check if we have data
    mot_path = MOT17_PATH
    if not os.path.exists(mot_path) and os.path.exists(os.path.join("..", MOT17_PATH)):
        mot_path = os.path.join("..", MOT17_PATH)
    seqs = get_validation_sequences(mot_path)
    logger.info(f"Found {len(seqs)} sequences for validation: {[s[0] for s in seqs]}")
    
    if not seqs:
        logger.error("Please run generate_yolo_detections.py first to create .npy files for sequences that have Ground Truth.")
        return

    # Create Study (Sampler doesn't matter as we manually enqueue trials)
    study = optuna.create_study(direction="maximize", study_name=STUDY_NAME)
    
    # --- Simulated Annealing Loop ---
    current_params = get_random_params(SEARCH_SPACE)
    current_score = -1.0
    best_score = -1.0
    best_params = current_params

    # Cooling Schedule
    T_initial = 1.0
    T_min = 0.001
    # Calculate alpha to reach T_min at the end of N_TRIALS
    alpha = math.exp(math.log(T_min / T_initial) / N_TRIALS)
    T = T_initial

    for i in range(N_TRIALS):
        # 1. Enqueue the parameters we want to test
        study.enqueue_trial(current_params)
        
        # 2. Run the trial
        trial = study.ask()
        try:
            new_score = objective(trial)
            study.tell(trial, new_score)
        except optuna.TrialPruned:
            study.tell(trial, state=optuna.trial.TrialState.PRUNED)
            new_score = -1.0
        except Exception as e:
            logger.error(f"Trial {i} failed: {e}")
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
            new_score = -1.0

        # 3. Acceptance Probability (Metropolis Criterion)
        delta = new_score - current_score
        
        if delta > 0:
            accept = True
            best_score = max(best_score, new_score)
            best_params = current_params
        else:
            # Scale delta for probability calculation (scores are roughly 0.0-1.0)
            probability = math.exp(delta / (T * 0.1))
            accept = random.random() < probability
        
        # 4. Update State & Perturb for next iteration
        if accept and new_score != -1.0:
            current_score = new_score
            
        current_params = perturb_params(current_params, SEARCH_SPACE, T)
        T *= alpha
    
    logger.info("Optimization Finished.")
    logger.info(f"Best Trial: {study.best_trial.number}")
    logger.info(f"Best Score: {study.best_value}")
    logger.info(f"Best Params: {study.best_params}")

if __name__ == "__main__":
    main()

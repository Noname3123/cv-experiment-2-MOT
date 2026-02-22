import os
import sys
import logging
import numpy as np
import motmetrics as mm
from datetime import datetime

# Bayesian Optimization (Gaussian Process)
try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer, Categorical
    from skopt.utils import use_named_args
except ImportError as e:
    raise ImportError(
        "scikit-optimize not found. Install with: pip install scikit-optimize"
    ) from e

# Ensure we can import from the same directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from optimization_config import SEARCH_SPACE
import evaluate_tracker_cached as tracker_module

# --- Configuration ---
MOT17_PATH = "MOT17"
STORAGE_DIR = "optimization_results"
N_CALLS = 120          # Total BO evaluations
N_RANDOM_STARTS = 25   # Random warmup points

# Setup Logging
os.makedirs(STORAGE_DIR, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(STORAGE_DIR, f"optimization_bayesopt_{timestamp}.log")

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
    for subdir in ["train", "test"]:
        path = os.path.join(mot_path, subdir)
        if not os.path.exists(path):
            continue

        for seq in os.listdir(path):
            seq_path = os.path.join(path, seq)
            if not os.path.isdir(seq_path):
                continue

            det_path = os.path.join(seq_path, f"{seq}_yolo_detections.npy")
            gt_path = os.path.join(seq_path, "gt", "gt.txt")

            if os.path.exists(det_path) and os.path.exists(gt_path):
                sequences.append((seq, det_path, gt_path))

    return sorted(sequences)


def build_skopt_space(search_space):
    """
    Converts SEARCH_SPACE dict into skopt dimensions + ordered param names.
    """
    dims = []
    names = []
    for key, config in search_space.items():
        names.append(key)
        if config["type"] == "float":
            dims.append(Real(config["low"], config["high"], name=key))
        elif config["type"] == "int":
            dims.append(Integer(config["low"], config["high"], name=key))
        elif config["type"] == "categorical":
            dims.append(Categorical(config["choices"], name=key))
        else:
            raise ValueError(f"Unsupported type in SEARCH_SPACE for '{key}': {config}")
    return dims, names


def locate_mot17():
    mot_path = MOT17_PATH
    if not os.path.exists(mot_path) and os.path.exists(os.path.join("..", MOT17_PATH)):
        mot_path = os.path.join("..", MOT17_PATH)
    return mot_path


# Global cached list (same as your other scripts’ pattern)
MOT_PATH = locate_mot17()
VAL_SEQS = get_validation_sequences(MOT_PATH)


def evaluate_params(params, trial_idx):
    """
    Runs tracker for all validation sequences and returns scalar score (higher is better).
    """
    trial_output_dir = os.path.join(STORAGE_DIR, f"trial_bayes_{trial_idx}")
    os.makedirs(trial_output_dir, exist_ok=True)

    accs, names = [], []
    for seq_name, det_path, gt_path in VAL_SEQS:
        all_detections = np.load(det_path)

        res_file, _ = tracker_module.process_sequence(
            seq_name,
            all_detections,
            trial_output_dir,
            MOT_PATH,
            params=params
        )

        gt = mm.io.loadtxt(gt_path, fmt="mot15-2D", min_confidence=1)
        ts = mm.io.loadtxt(res_file, fmt="mot15-2D")
        acc = mm.utils.compare_to_groundtruth(gt, ts, 'iou', distth=0.5)
        accs.append(acc)
        names.append(seq_name)

    mh = mm.metrics.create()
    summary = mh.compute_many(
        accs,
        metrics=['mota', 'idf1', 'num_switches'],
        names=names,
        generate_overall=True
    )

    overall_mota = float(summary.loc['OVERALL']['mota'])
    overall_idf1 = float(summary.loc['OVERALL']['idf1'])
    overall_idsw = int(summary.loc['OVERALL']['num_switches'])

    score = (overall_mota + overall_idf1) / 2.0

    logger.info(
        f"Eval {trial_idx}: Score={score:.4f} | MOTA={overall_mota:.4f} | IDF1={overall_idf1:.4f} | "
        f"IDsw={overall_idsw} | Params={params}"
    )

    return score


def main():
    logger.info("Starting Hyperparameter Optimization (Bayesian Optimization / Gaussian Process)...")

    if not VAL_SEQS:
        logger.error("No validation sequences found (need .npy detections AND gt.txt).")
        logger.error("Please run generate_yolo_detections.py first to create .npy files for sequences that have Ground Truth.")
        return

    logger.info(f"Found {len(VAL_SEQS)} sequences for validation: {[s[0] for s in VAL_SEQS]}")
    dims, ordered_names = build_skopt_space(SEARCH_SPACE)

    # We minimize in gp_minimize, so objective returns -score
    eval_counter = {"i": 0}

    @use_named_args(dims)
    def objective(**kwargs):
        eval_counter["i"] += 1
        i = eval_counter["i"]

        params = dict(kwargs)

        # Constraint handling: if invalid, return large penalty (remember: minimize)
        if params["CONFIDENCE_LOW"] >= params["CONFIDENCE_THRESHOLD"]:
            logger.info(f"Eval {i}: INVALID (CONFIDENCE_LOW >= CONFIDENCE_THRESHOLD) -> penalty")
            return 1e6

        try:
            score = evaluate_params(params, trial_idx=i)
            return -score
        except Exception as e:
            logger.error(f"Eval {i} failed: {e}")
            return 1e6

    result = gp_minimize(
        func=objective,
        dimensions=dims,
        n_calls=N_CALLS,
        n_random_starts=N_RANDOM_STARTS,
        acq_func="EI",       # Expected Improvement
        random_state=42
    )

    best_x = result.x
    best_params = {name: val for name, val in zip(ordered_names, best_x)}
    best_score = -result.fun

    logger.info("Optimization Finished.")
    logger.info(f"Best Score: {best_score:.4f}")
    logger.info(f"Best Params: {best_params}")


if __name__ == "__main__":
    main()

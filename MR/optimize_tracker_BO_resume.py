import os
import sys
import re
import json
import logging
import numpy as np
import motmetrics as mm
from datetime import datetime

# Bayesian Optimization (skopt Optimizer for "ask/tell" resume)
try:
    from skopt import Optimizer
    from skopt.space import Real, Integer, Categorical
except ImportError as e:
    raise ImportError("scikit-optimize not found. Install with: pip install scikit-optimize") from e

# Ensure we can import from the same directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from optimization_config import SEARCH_SPACE
import evaluate_tracker_cached as tracker_module

# --- Configuration (match your other scripts) ---
MOT17_PATH = "MOT17"
STORAGE_DIR = "optimization_results"

# --- RESUME SETTINGS ---
RESUME_LOG_PATH = os.path.join(STORAGE_DIR, "optimization_bayesopt_20260212_171706.log")  # <-- your uploaded log name
TARGET_EVALS = 300                     # go from 1..300 total
RANDOM_STATE = 42
ACQ_FUNC = "EI"                        # same intent as gp_minimize(acq_func="EI")

# Setup Logging
os.makedirs(STORAGE_DIR, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(STORAGE_DIR, f"optimization_bayesopt_resume_{timestamp}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Optional: also write a machine-readable progress file (so next time resume is trivial)
PROGRESS_JSONL = os.path.join(STORAGE_DIR, "bayesopt_progress.jsonl")


def get_validation_sequences(mot_path):
    """
    Finds sequences that have both cached detections and Ground Truth.
    (Your dataset is in MOT17/test; train is not required.)
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
    dims, names = [], []
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


# Global cached list
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
        acc = mm.utils.compare_to_groundtruth(gt, ts, "iou", distth=0.5)
        accs.append(acc)
        names.append(seq_name)

    mh = mm.metrics.create()
    summary = mh.compute_many(
        accs,
        metrics=["mota", "idf1", "num_switches"],
        names=names,
        generate_overall=True
    )

    overall_mota = float(summary.loc["OVERALL"]["mota"])
    overall_idf1 = float(summary.loc["OVERALL"]["idf1"])
    overall_idsw = int(summary.loc["OVERALL"]["num_switches"])

    score = (overall_mota + overall_idf1) / 2.0

    logger.info(
        f"Eval {trial_idx}: Score={score:.4f} | MOTA={overall_mota:.4f} | IDF1={overall_idf1:.4f} | "
        f"IDsw={overall_idsw} | Params={params}"
    )

    # Write machine-readable progress line
    try:
        with open(PROGRESS_JSONL, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "eval": trial_idx,
                "score": score,
                "mota": overall_mota,
                "idf1": overall_idf1,
                "idsw": overall_idsw,
                "params": params
            }) + "\n")
    except Exception:
        pass

    return score


def parse_existing_evals_from_log(log_path):
    """
    Parses lines like:
    Eval 12: Score=0.1234 | ... | Params={...}
    Returns dict: eval_idx -> {"score": float, "params": dict}
    """
    if not os.path.exists(log_path):
        logger.warning(f"Resume log not found: {log_path}")
        return {}

    eval_re = re.compile(
        r"Eval\s+(\d+):\s+Score=([0-9.]+)\s+\|\s+MOTA=([0-9.]+)\s+\|\s+IDF1=([0-9.]+)\s+\|\s+IDsw=(\d+)\s+\|\s+Params=(\{.*\})"
    )

    data = {}
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = eval_re.search(line)
            if not m:
                continue
            idx = int(m.group(1))
            score = float(m.group(2))
            # Params is a Python dict literal in the log
            params = eval(m.group(6), {"__builtins__": {}})
            data[idx] = {"score": score, "params": params}

    return data


def params_to_x(params, ordered_names):
    return [params[name] for name in ordered_names]


def main():
    logger.info("Starting Hyperparameter Optimization (BayesOpt RESUME via skopt Optimizer ask/tell)...")

    if not VAL_SEQS:
        logger.error("No validation sequences found (need .npy detections AND gt.txt).")
        return

    logger.info(f"Found {len(VAL_SEQS)} sequences for validation: {[s[0] for s in VAL_SEQS]}")

    dims, ordered_names = build_skopt_space(SEARCH_SPACE)

    # Load existing evaluations from the previous log
    existing = parse_existing_evals_from_log(RESUME_LOG_PATH)
    if not existing:
        logger.error("No existing evals found in resume log -> this would start from scratch.")
        logger.error(f"Check RESUME_LOG_PATH: {RESUME_LOG_PATH}")
        return

    done_evals = sorted(existing.keys())
    last_done = max(done_evals)
    logger.info(f"Loaded {len(done_evals)} existing evals from log. Last eval = {last_done}.")
    if last_done >= TARGET_EVALS:
        logger.info(f"Nothing to do: last_done ({last_done}) >= TARGET_EVALS ({TARGET_EVALS}).")
        return

    # Build optimizer and "tell" it the old data
    opt = Optimizer(
        dimensions=dims,
        base_estimator="GP",
        acq_func=ACQ_FUNC,
        random_state=RANDOM_STATE
    )

    best_score = -1.0
    best_params = None

    # Feed previous observations
    # objective for optimizer is MINIMIZE => y = -score
    for idx in done_evals:
        params = existing[idx]["params"]
        score = existing[idx]["score"]

        x = params_to_x(params, ordered_names)
        y = -score
        opt.tell(x, y)

        if score > best_score:
            best_score = score
            best_params = params

    logger.info(f"Best from loaded history: Score={best_score:.4f} | Params={best_params}")

    # Continue evaluations
    next_eval = last_done + 1
    while next_eval <= TARGET_EVALS:
        x = opt.ask()
        params = {name: val for name, val in zip(ordered_names, x)}

        # Constraint: CONFIDENCE_LOW must be < CONFIDENCE_THRESHOLD
        if params["CONFIDENCE_LOW"] >= params["CONFIDENCE_THRESHOLD"]:
            logger.info(f"Eval {next_eval}: INVALID (CONFIDENCE_LOW >= CONFIDENCE_THRESHOLD) -> penalty")
            opt.tell(x, 1e6)  # large penalty (minimize)
            next_eval += 1
            continue

        try:
            score = evaluate_params(params, trial_idx=next_eval)
            y = -score
        except Exception as e:
            logger.error(f"Eval {next_eval} failed: {e}")
            y = 1e6

        opt.tell(x, y)

        if y != 1e6 and score > best_score:
            best_score = score
            best_params = params
            logger.info(f"New BEST at Eval {next_eval}: Score={best_score:.4f}")

        next_eval += 1

    logger.info("Resume Optimization Finished.")
    logger.info(f"Best Score (overall): {best_score:.4f}")
    logger.info(f"Best Params (overall): {best_params}")


if __name__ == "__main__":
    main()

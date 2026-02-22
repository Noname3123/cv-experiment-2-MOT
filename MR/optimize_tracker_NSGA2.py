import os
import sys
import logging
import numpy as np
import motmetrics as mm
import optuna
from optuna.samplers import NSGAIISampler
from datetime import datetime

# Ensure we can import from the same directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from optimization_config import SEARCH_SPACE
import evaluate_tracker_cached as tracker_module

# --- Configuration ---
MOT17_PATH = "MOT17"
STUDY_NAME = "ocsort_optimization_nsga2"
STORAGE_DIR = "optimization_results"
N_TRIALS = 300  # Number of optimization trials

# Setup Logging
os.makedirs(STORAGE_DIR, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(STORAGE_DIR, f"optimization_nsga2_{timestamp}.log")

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


def sample_params(trial):
    params = {}
    for key, config in SEARCH_SPACE.items():
        if config["type"] == "float":
            params[key] = trial.suggest_float(key, config["low"], config["high"])
        elif config["type"] == "int":
            params[key] = trial.suggest_int(key, config["low"], config["high"])
        elif config["type"] == "categorical":
            params[key] = trial.suggest_categorical(key, config["choices"])
    return params


def objective(trial):
    # 1) Sample hyperparams
    params = sample_params(trial)

    # Constraint
    if params["CONFIDENCE_LOW"] >= params["CONFIDENCE_THRESHOLD"]:
        raise optuna.TrialPruned("CONFIDENCE_LOW >= CONFIDENCE_THRESHOLD")

    # 2) Locate MOT17
    mot_path = MOT17_PATH
    if not os.path.exists(mot_path) and os.path.exists(os.path.join("..", MOT17_PATH)):
        mot_path = os.path.join("..", MOT17_PATH)

    val_sequences = get_validation_sequences(mot_path)
    if not val_sequences:
        logger.error("No validation sequences found (need .npy detections AND gt.txt).")
        # Return very bad objectives
        return 0.0, 0.0, 10**9

    # 3) Trial output dir
    trial_output_dir = os.path.join(STORAGE_DIR, f"trial_nsga2_{trial.number}")
    os.makedirs(trial_output_dir, exist_ok=True)

    accs, names = [], []

    for seq_name, det_path, gt_path in val_sequences:
        all_detections = np.load(det_path)

        res_file, _ = tracker_module.process_sequence(
            seq_name,
            all_detections,
            trial_output_dir,
            mot_path,
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

    # NSGA-II is multi-objective:
    # - maximize MOTA
    # - maximize IDF1
    # - minimize ID switches
    logger.info(
        f"Trial {trial.number}: MOTA={overall_mota:.4f} | IDF1={overall_idf1:.4f} | "
        f"IDsw={overall_idsw} | Params={params}"
    )

    return overall_mota, overall_idf1, overall_idsw


def pick_representative_solution(pareto_trials):
    """
    From Pareto front choose a single "representative" solution for printing:
    maximize (MOTA + IDF1)/2 and tie-break by lower IDsw.
    """
    def key_fn(t):
        mota, idf1, idsw = t.values
        return ((mota + idf1) / 2.0, -idsw)

    return max(pareto_trials, key=key_fn)


def main():
    logger.info("Starting Hyperparameter Optimization (NSGA-II Genetic Algorithm)...")

    mot_path = MOT17_PATH
    if not os.path.exists(mot_path) and os.path.exists(os.path.join("..", MOT17_PATH)):
        mot_path = os.path.join("..", MOT17_PATH)

    seqs = get_validation_sequences(mot_path)
    logger.info(f"Found {len(seqs)} sequences for validation: {[s[0] for s in seqs]}")

    if not seqs:
        logger.error("Please run generate_yolo_detections.py first to create .npy files for sequences that have Ground Truth.")
        return

    sampler = NSGAIISampler()
    study = optuna.create_study(
        directions=["maximize", "maximize", "minimize"],
        study_name=STUDY_NAME,
        sampler=sampler
    )

    study.optimize(objective, n_trials=N_TRIALS)

    logger.info("Optimization Finished.")
    pareto = study.best_trials  # Pareto front
    logger.info(f"Pareto front size: {len(pareto)}")

    # Print a representative "best" (scalarized) solution for convenience
    rep = pick_representative_solution(pareto)
    mota, idf1, idsw = rep.values
    score = (mota + idf1) / 2.0
    logger.info("Representative solution (max avg(MOTA, IDF1), tie-break min IDsw):")
    logger.info(f"  Score(avg)={score:.4f} | MOTA={mota:.4f} | IDF1={idf1:.4f} | IDsw={int(idsw)}")
    logger.info(f"  Params={rep.params}")

    # Also dump all Pareto trials (short)
    for t in pareto:
        mota, idf1, idsw = t.values
        logger.info(f"[Pareto] Trial {t.number}: MOTA={mota:.4f} IDF1={idf1:.4f} IDsw={int(idsw)} Params={t.params}")


if __name__ == "__main__":
    main()

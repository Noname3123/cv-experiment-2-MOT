import os
import shutil
import yaml
import cv2
import datetime
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt

# --- Configuration ---
MOT17_PATH = "MOT17"
YOLO_DATASET_DIR = "mot17_yolo_data_singleclass"
EXPERIMENT_DIR = "experiments"
MODEL_WEIGHTS = "yolo11x.pt"
IMG_SIZE = 1280  # High res setting from your tracking experiments

# --- Data Preparation (Adapted from train_yolo_ocsort.py) ---

def convert_mot_to_yolo():
    print("Preparing YOLO data from MOT17...")
    
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
        
        # Extract sequence ID to avoid duplicates (e.g. MOT17-02-SDP vs MOT17-02-DPM)
        seq_id = "-".join(seq.split("-")[:2])
        if seq_id in processed_ids:
            continue
        processed_ids.add(seq_id)

        print(f"Processing sequence: {seq}")
        
        img_dir = os.path.join(seq_path, "img1")
        gt_file = os.path.join(seq_path, "gt", "gt.txt")
        
        if not os.path.exists(gt_file):
            continue

        gt_data = np.loadtxt(gt_file, delimiter=',')
        
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

                if cls_id != 1:  # Only Pedestrian
                    continue

                x1, y1, w, h = row[2], row[3], row[4], row[5]
                cx = (x1 + w / 2) / img_w
                cy = (y1 + h / 2) / img_h
                nw = w / img_w
                nh = h / img_h
                
                # YOLO class 0
                yolo_lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

            if not yolo_lines:
                continue

            src_img = os.path.join(img_dir, f"{int(frame_idx):06d}.jpg")
            dst_img_name = f"{seq}_{int(frame_idx):06d}.jpg"
            dst_img = os.path.join(images_train_dir, dst_img_name)
            shutil.copy(src_img, dst_img)
            
            dst_label = os.path.join(labels_train_dir, dst_img_name.replace('.jpg', '.txt'))
            with open(dst_label, 'w') as f:
                f.write("\n".join(yolo_lines))

    yaml_content = {
        'path': os.path.abspath(YOLO_DATASET_DIR),
        'train': 'images/train',
        'val': 'images/train', # Use train set for validation since it has GT
        'names': {
            0: 'person'
        }
    }
    
    with open(os.path.join(YOLO_DATASET_DIR, "mot17.yaml"), 'w') as f:
        yaml.dump(yaml_content, f)
        
    print("Data preparation complete.")

# --- Custom Plotting ---

def plot_custom_curves(metrics, save_dir):
    """
    Generates custom plots for P, R, F1, and PR curves for the 'person' class only.
    Legend is placed below the plot.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # metrics.box.p/r/f1 are shape (nc, 1000)
    # We assume single class evaluation (Person) or we take the first class
    cls_idx = 0
    cls_name = 'Person'
    
    # Attempt to retrieve curve data
    p_curve_raw, r_curve_raw, f1_curve_raw = None, None, None

    # Priority 1: metrics.box.curves
    raw_curves = getattr(metrics.box, 'curves', [])
    if raw_curves and len(raw_curves) >= 3:
        print("Using metrics.box.curves for plotting data.")
        p_curve_raw, r_curve_raw, f1_curve_raw = raw_curves[:3]
    
    # Priority 2: metrics.box.{p,r,f1}_curve attributes
    if p_curve_raw is None:
        p_curve_raw = getattr(metrics.box, 'p_curve', None)
        r_curve_raw = getattr(metrics.box, 'r_curve', None)
        f1_curve_raw = getattr(metrics.box, 'f1_curve', None)
        if p_curve_raw is not None:
            print("Using metrics.box.{p,r,f1}_curve attributes.")

    # Priority 3: Fallback to scalars (will likely result in skipped plots)
    if p_curve_raw is None:
        print("Warning: Curve data not found. Using scalar attributes.")
        p_curve_raw, r_curve_raw, f1_curve_raw = metrics.box.p, metrics.box.r, metrics.box.f1

    def get_metric_array(arr, c_idx):
        if not isinstance(arr, np.ndarray):
            arr = np.array(arr)
        if arr.size <= 1:
            return arr
        if arr.ndim == 1:
            return arr
        if arr.shape[0] > arr.shape[1]: 
            return arr[:, c_idx]
        else:
            return arr[c_idx]

    f1_curve = get_metric_array(f1_curve_raw, cls_idx)
    p_curve = get_metric_array(p_curve_raw, cls_idx)
    r_curve = get_metric_array(r_curve_raw, cls_idx)

    # Define the curves to plot: (Data Array, Y-Label, Title, Filename, Label Suffix)
    curves = [
        (f1_curve, 'F1', 'F1-Confidence Curve', 'Custom_BoxF1_curve.png', f'Max: {f1_curve.max():.2f}'),
        (p_curve, 'Precision', 'Precision-Confidence Curve', 'Custom_BoxP_curve.png', ''),
        (r_curve, 'Recall', 'Recall-Confidence Curve', 'Custom_BoxR_curve.png', '')
    ]

    for data, ylabel, title, filename, label_suffix in curves:
        if data.size < 2:
            print(f"Skipping {filename}: Data size too small ({data.shape})")
            continue
            
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        # Dynamic x-axis based on data length
        x = np.linspace(0, 1, len(data))
        label = f'{cls_name} {label_suffix}'.strip()
        ax.plot(x, data, linewidth=2, label=label, color='blue')
        ax.set_xlabel('Confidence')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.grid(True, linestyle='--', alpha=0.5)
        # Legend inside the plot
        ax.legend(loc='best', fancybox=True, shadow=False, ncol=1)
        plt.tight_layout()
        fig.savefig(os.path.join(save_dir, filename), dpi=300)
        plt.close(fig)

    # PR Curve (P vs R)
    if r_curve.size < 2 or p_curve.size < 2:
        print("Skipping PR curve: Data size too small")
        return
        
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ap50 = metrics.box.map50
    ax.plot(r_curve, p_curve, linewidth=2, label=f'{cls_name} mAP@0.5: {ap50:.3f}', color='blue')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='best', fancybox=True, shadow=False, ncol=1)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'Custom_BoxPR_curve.png'), dpi=300)
    plt.close(fig)
    print("Custom curves saved.")

# --- Evaluation ---

def main():
    # 1. Prepare Data
    convert_mot_to_yolo()
    
    # 2. Load Base Model
    print(f"Loading base model: {MODEL_WEIGHTS}...")
    model = YOLO(MODEL_WEIGHTS)
    
    # We use the generated yaml which points 'val' to the training images (since we have GT there)
    yaml_path = os.path.join(YOLO_DATASET_DIR, "mot17.yaml")
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"yolo11x_mot17_detection_singleclass_{timestamp}"

    # Run validation
    # classes=[0] restricts evaluation to the 'Pedestrian' class.
    # Note: yolo11x.pt is trained on COCO where class 0 is 'person'.
    # Our MOT17 dataset yaml defines class 0 as 'Pedestrian'. These align perfectly.
    metrics = model.val(
        data=yaml_path,
        imgsz=IMG_SIZE,
        batch=4,
        conf=0.001,       # Low confidence for calculating full mAP curves
        iou=0.6,          # NMS IoU threshold
        classes=[0],      # Only evaluate 'Pedestrian' class
        project=EXPERIMENT_DIR,
        name=run_name,
        exist_ok=True,
        verbose=True,
        plots=True
    )
    
    # --- Custom Confusion Matrix Calculation ---
    print("\n" + "="*40)
    print("Custom Single-Class Confusion Matrix (Person vs Background)")
    print("="*40)

    # The confusion matrix from ultralytics has shape (nc+1, nc+1)
    # where nc is the number of classes in the model (80 for COCO).
    # The last row represents False Positives (predictions with no matching GT).
    # The last column represents False Negatives (GT with no matching prediction).
    # Our GT dataset only contains class 0 (person).

    full_matrix = metrics.confusion_matrix.matrix
    person_class_index = 0

    # True Positives: GT is 'person' and prediction is 'person'.
    tp = full_matrix[person_class_index, person_class_index]

    # False Negatives: GT is 'person' but prediction is background (missed).
    fn = full_matrix[person_class_index, -1]

    # False Positives: GT is background but prediction is 'person'.
    fp = full_matrix[-1, person_class_index]

    print(f"True Positives (TP): {int(tp)}")
    print(f"False Positives (FP): {int(fp)}")
    print(f"False Negatives (FN): {int(fn)}")

    print("\nSimple Matrix View:")
    print("          | Pred Person | Pred Background (Missed)")
    print("----------------------------------------------------")
    print(f"GT Person | {int(tp):<11} | {int(fn):<11}")
    print(f"GT Bkgnd  | {int(fp):<11} | TN: N/A")
    print("="*40)
    print("Note: The full confusion matrix plot from Ultralytics is also saved in the run directory.")

    print("\n" + "="*40)
    print(f"Detection Results for {MODEL_WEIGHTS} on MOT17 (Pedestrian)")
    print("="*40)
    print(f"mAP@50:    {metrics.box.map50:.4f}")
    print(f"mAP@50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall:    {metrics.box.mr:.4f}")
    print("="*40)
    print(f"Detailed results saved to: {os.path.join(EXPERIMENT_DIR, run_name)}")
    
    # Generate custom plots
    plot_custom_curves(metrics, os.path.join(EXPERIMENT_DIR, run_name))

if __name__ == "__main__":
    main()

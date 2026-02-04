# EXPERIMENTAL RESULTS AND ANALYSIS

This section documents the experimental evaluation of the proposed tracking pipeline on the MOT17 benchmark. We focus on two primary configurations to assess the impact of input resolution and the effectiveness of the integrated tracking components.

## A. Experiment 10: Optimized OC-SORT with CMC

### 1) Experimental Setup and Implementation
The first configuration (Experiment 10) represents the proposed robust tracking pipeline. The system integrates a YOLO11x object detector with an enhanced OC-SORT tracker. The input image resolution was set to $1280 \times 1280$ pixels.

**Detector Implementation:**
We utilized the YOLO11x model, pre-trained on the COCO dataset and fine-tuned for the pedestrian class. To adapt the MOT17 dataset for YOLO training and validation, we converted the ground truth annotations from the standard MOT format `[frame, id, left, top, width, height]` to the YOLO format `[class, x_center, y_center, width, height]`, normalized by image dimensions. Only entries belonging to the 'Pedestrian' class (Class ID 1 in MOT17) were retained to ensure a single-class detection focus.

**Tracker Implementation:**
The tracking core is based on the Observation-Centric SORT (OC-SORT) algorithm, which uses a Kalman Filter to estimate object state. We augmented the baseline implementation with three critical components to address the challenges of dynamic urban scenes:

1.  **Camera Motion Compensation (CMC):** To decouple ego-motion from pedestrian motion, we implemented a CMC module using the Lucas-Kanade optical flow method. Features are tracked between consecutive frames to estimate an affine transformation matrix (translation, rotation, and scale). This transformation is applied to the Kalman Filter's state vector (position and velocity) before the prediction step, ensuring that the motion model accounts for the camera's movement.
2.  **ByteTrack Association Logic:** To mitigate fragmented trajectories caused by occlusion, we adopted a two-stage data association strategy. High-confidence detections ($score \ge 0.5$) are associated first using Intersection over Union (IoU). Remaining low-confidence detections ($0.05 \le score < 0.5$) are then matched against unmatched tracks. This allows the tracker to recover objects that are partially occluded or blurred, which typically exhibit lower detection scores.
3.  **Trajectory Interpolation:** A post-processing linear interpolation step was applied to fill gaps in trajectories up to 20 frames long, significantly improving the recall of the system.

### 2) Detector Performance
The performance of the YOLO11x detector was validated on the prepared MOT17 dataset. The confusion matrix and precision-recall curves are presented below.

!Confusion Matrix
*Figure 1: Confusion Matrix for the YOLO11x detector on the MOT17 pedestrian class.*

!Precision-Recall Curve
*Figure 2: Precision-Recall Curve demonstrating the detector's trade-off between false positives and false negatives.*

As indicated by the validation artifacts in Figure 1 and Figure 2, the detector achieves a robust balance between precision and recall. The high true positive rate is crucial for the subsequent tracking stage, as the OC-SORT algorithm relies heavily on the quality of observations to correct its momentum-based predictions.

### 3) Tracking Results
The tracking performance for Experiment 10 was evaluated using the standard MOTChallenge metrics. The results are summarized in Table I.

**Table I: Tracking Performance on MOT17 (Experiment 10)**

| Metric | Value | Description |
| :--- | :--- | :--- |
| **MOTA** | **40.4%** | Multiple Object Tracking Accuracy |
| **IDF1** | **51.7%** | ID F1 Score |
| **Recall** | 51.4% | Percentage of ground truth objects detected |
| **Precision** | 82.7% | Percentage of detected objects that are relevant |
| **ID Switches** | 276 | Number of identity jumps |
| **MT** | 138 | Mostly Tracked trajectories |
| **ML** | 141 | Mostly Lost trajectories |
| **FP** | 9,744 | False Positives |
| **FN** | 44,028 | False Negatives |

**Analysis:**
The configuration achieved a MOTA of 40.4% and an IDF1 of 51.7%. Notably, the inclusion of Camera Motion Compensation and ByteTrack logic resulted in a relatively low number of ID switches (276) compared to baseline methods evaluated in preliminary studies. The high precision (82.7%) suggests that the detector and the association logic effectively filter out background noise. The recall of 51.4% indicates that the interpolation and low-confidence matching successfully recovered a significant portion of pedestrian trajectories that would otherwise be lost due to occlusion.

## B. Experiment 11: High-Resolution Inference

### 1) Comparative Setup
Experiment 11 utilized the identical pipeline as Experiment 10 (YOLO11x + OC-SORT + ByteTrack + CMC + Interpolation) but increased the input inference resolution from $1280 \times 1280$ to $1920 \times 1920$ pixels. The objective was to determine if higher resolution input yields a proportional increase in tracking accuracy by resolving smaller or distant pedestrians.

### 2) Comparative Analysis
The results for Experiment 11 are presented in Table II, alongside the delta from Experiment 10.

**Table II: Comparison of Experiment 11 (1920px) vs. Experiment 10 (1280px)**

| Metric | Exp 11 (1920px) | Exp 10 (1280px) | $\Delta$ |
| :--- | :--- | :--- | :--- |
| **MOTA** | 40.7% | 40.4% | +0.3% |
| **IDF1** | 52.1% | 51.7% | +0.4% |
| **Recall** | 50.5% | 51.4% | -0.9% |
| **Precision** | 84.2% | 82.7% | +1.5% |
| **ID Switches** | 306 | 276 | +30 |
| **FP** | 8,610 | 9,744 | -1,134 |
| **FN** | 44,871 | 44,028 | +843 |

**Discussion:**
Increasing the resolution to 1920px resulted in marginal gains in the primary composite metrics, with MOTA increasing by only 0.3% and IDF1 by 0.4%.

Interestingly, the recall decreased slightly (from 51.4% to 50.5%), while precision improved (from 82.7% to 84.2%). This suggests that while the higher resolution allows the model to be more selective—reducing False Positives by over 1,000—it also leads to the suppression of some valid detections, possibly due to changes in the effective anchor scales or confidence calibration at higher resolutions. Furthermore, the number of ID switches increased from 276 to 306, indicating slightly less stable identity maintenance.

**Conclusion on Configuration:**
We conclude that the configuration from **Experiment 10 (1280px)** is superior for practical deployment. The marginal improvement in MOTA observed in Experiment 11 does not justify the significantly higher computational cost associated with processing $1920 \times 1920$ images (a $2.25\times$ increase in pixel count). The 1280px setup offers a better balance between recall and precision, maintains more stable identities (fewer ID switches), and operates with greater computational efficiency, making it the optimal choice for autonomous driving systems where real-time processing is a constraint.

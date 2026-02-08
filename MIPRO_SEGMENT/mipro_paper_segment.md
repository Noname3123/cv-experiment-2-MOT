# I. INTRODUCTION

Multiple Object Tracking (MOT) is a fundamental component of modern perception systems, particularly in the domain of autonomous driving and intelligent transporation systems, where reliable and temporally consistent tracking of pedestrians is essential for safe navigation and making decisions. Unlike single-frame object detection, MOT requires maintaining object identities across time under challenging real-world conditions such as frequent occlusions, illumination changes, viewpoint variations and camera motion. These challenges are especially pronounced in urban environments, where dense crowds and dynamic backgrounds significantly increase the complexity of the tracking task.

To support standardized evaluation and fair comparison of tracking methods, the MOTChallenge benchmark has become the de facto reference in the research community. In this work, we focus on the MOT17 dataset, which provides a diverse set of urban video sequences annotated with pedestrian trajectories and accompanied by detections from multiple detectors. MOT17 enables evaluation under realistic conditions and reports performance using a rich set of metrics that capture both detection quality and identity preservation.

Recent advances in MOT have shown that tracking-by-detection pipelines remain dominant, combining strong object detectors with increasingly sophisticated data association strategies. While deep learning has significantly improved detection accuracy, identity consistency remains a key challenge, particularly in scenarios involving partial occlusions and camera motion. As a result, modern trackers often integrate classical motion models with additional heuristics and post-processing steps to enhance robustness.

Among contemporary tracking methods, OC-SORT (Observation-Centric SORT) has emerged as a promising approach by rethinking the role of the Kalman filter and emphasizing observation-driven updates. Unlike traditional SORT-based trackers, OC-SORT reduces reliance on motion prediction during ambiguous associations, which has been shown to improve robustness in crowded scenes. However, as with most MOT frameworks, its performance strongly depends on detector quality, association logic, and hyperparameter tuning.

The goal of this paper is to conduct a systematic experimental evaluation of multiple tracking configurations on the MOT17 benchmark, with particular emphasis on OC-SORT-based pipelines. We analyze how different design choices—such as detector confidence thresholds, inference resolution, association strategies, interpolation, and camera motion compensation—affect overall tracking performance. In total, 18 experimental configurations were evaluated, ranging from baseline trackers to progressively enhanced systems.

Performance is assessed using standard MOTChallenge metrics, including MOTA, IDF1, precision, recall, and identity switches. While MOTA provides an overall accuracy measure, IDF1 is of particular importance in this study, as it directly reflects the ability of a tracker to preserve object identities over time—an essential requirement for downstream autonomous driving applications.

The experiments were designed in an incremental manner: starting from baseline configurations, individual components were introduced step by step to isolate their contribution to tracking performance. This approach enables a clear interpretation of results and provides practical insight into which mechanisms yield the most significant improvements under realistic constraints.

# II. EXPERIMENTAL RESULTS AND ANALYSIS

To determine the optimal tracking architecture, a total of 18 distinct experimental configurations were evaluated, exploring a spectrum of methodologies ranging from simple distance-based association to complex segmentation-driven tracking. Initial baselines utilizing Norfair and DeepSORT yielded modest MOTA scores of 21.4% and 12.3%, respectively, highlighting the limitations of standard re-identification embeddings and Euclidean matching in crowded scenes. Attempts to fine-tune the YOLO detector on the MOT17 training split resulted in performance degradation (MOTA < 20%), suggesting overfitting to the specific sequence characteristics. Similarly, the integration of the Segment Anything Model 2 (SAM2) proved ineffective for this task, suffering from excessive false positives. In contrast, the OC-SORT algorithm demonstrated superior potential; starting from a baseline MOTA of 22.3%, performance was progressively enhanced through hyperparameter tuning (35.8%), the incorporation of ByteTrack association logic (34.6%), and trajectory interpolation (38.9%). Building upon these findings, this section documents the experimental evaluation of the proposed tracking pipeline on the MOT17 benchmark, focusing on the two most effective configurations to assess the impact of input resolution and the effectiveness of the integrated tracking components.

## A. Experiment 10: Optimized OC-SORT with CMC

### 1) Experimental Setup and Implementation
The first configuration (Experiment 10) represents the proposed robust tracking pipeline. The system integrates a YOLO11x object detector with an enhanced OC-SORT tracker. The input image resolution was set to $1280 \times 1280$ pixels.

We utilized the YOLO11x model as a detector, pre-trained on the COCO dataset. To adapt the MOT17 dataset for YOLO training and validation, we converted the ground truth annotations from the standard MOT format `[frame, id, left, top, width, height]` to the YOLO format `[class, x_center, y_center, width, height]`, normalized by image dimensions. Only entries belonging to the 'Pedestrian' class (Class ID 1 in MOT17) were used to perform a single-class testing of the model.


The tracking core is based on the Observation-Centric SORT (OC-SORT) algorithm, which uses a Kalman Filter to estimate object state. We augmented the baseline implementation with three critical components to address the challenges of dynamic urban scenes:

1.  **Camera Motion Compensation (CMC):** To decouple ego-motion (eg. dashboard cameras) from pedestrian motion, we implemented a CMC module using the Lucas-Kanade optical flow method. Features are tracked between consecutive frames to estimate an affine transformation matrix (translation, rotation, and scale). This transformation is applied to the Kalman Filter's state vector (position and velocity) before the prediction step, ensuring that the motion model accounts for the camera's movement.
2.  **ByteTrack Association Logic:** To mitigate fragmented trajectories caused by occlusion, we adopted a two-stage data association strategy. High-confidence detections ($score \ge 0.5$) are associated first using Intersection over Union (IoU). Remaining low-confidence detections ($0.05 \le score < 0.5$) are then matched against unmatched tracks. This allows the tracker to recover objects that are partially occluded or blurred, which typically exhibit lower detection scores.
3.  **Trajectory Interpolation:** A post-processing linear interpolation step was applied to fill gaps in trajectories up to 20 frames long, significantly improving the recall of the system.

### 2) Detector Performance
The performance of the YOLO11x detector was validated on the prepared subset of the MOT17 dataset. The confusion matrix and precision-recall curves are presented below.



|    Ground truth labels      | Pred Person | Pred Background (Missed)|
|----------|-------------|--------------------------|
|GT Person | 23362       | 8940 |      
|GT Bkgnd  | 4837        | TN: N/A|

*Figure 1: Confusion Matrix for the YOLO11x detector on the MOT17 pedestrian class.*

![Precision-Recall Curve](./images/BoxPR_curve.png)
*Figure 2: Precision-Recall Curve demonstrating the detector's trade-off between false positives and false negatives.*

As indicated by the validation artifacts in Figure 1 and Figure 2, the detector achieves an acceptable balance between precision and recall. The high true positive rate is crucial for the subsequent tracking stage, as the OC-SORT algorithm relies heavily on the quality of observations to correct its momentum-based predictions. 

Additional details can be seen by analysing the model's F1, P and R curves.

Figure 3 depicts the F1-Confidence curve, illustrating the harmonic mean of precision and recall across varying confidence thresholds. The curve exhibits a stable plateau, indicating that the detector's performance is robust to threshold variations. The peak F1 score suggests an optimal trade-off point, which informs the selection of the high-confidence threshold ($0.5$) used in our tracking experiments. This ensures that the primary tracking associations are grounded in high-quality detections, while the subsequent association stages can exploit the higher recall available at lower confidence levels.

![F1 Curve](./images/BoxF1_curve.png)
*Figure 3: F1-Confidence Curve. The curve demonstrates the optimal confidence threshold for maximizing the harmonic mean of precision and recall.*

Figure 4 presents the Precision-Confidence curve, which characterizes the detector's reliability across the confidence spectrum. The curve demonstrates a monotonic increase in precision as the confidence threshold rises, indicating effective suppression of false positives at higher scores. At the operational threshold of $0.5$, the detector achieves a precision significantly above $0.8$, ensuring that the initial tracking candidates are of high quality. This high-precision regime is critical for the OC-SORT tracker, as it reduces the likelihood of initializing false tracks or incorrectly updating existing Kalman filters with background noise.

![PCurve](./images/BoxP_curve.png)

*Figure 4: Precision-Confidence Curve. The monotonic rise in precision validates the model's ability to filter false positives at higher confidence levels.*

Finally, Figure 5 illustrates the Recall-Confidence curve, showing the inverse relationship between confidence and recall. As the confidence threshold increases, the system naturally retrieves fewer ground truth objects. While a threshold of $0.5$ reduces the raw detection recall, this trade-off is mitigated by the ByteTrack algorithm, which specifically recovers low-confidence detections ($0.1$ to $0.5$) during the second association stage. This hierarchical approach allows the system to maintain high precision for track initialization while preserving recall for ongoing trajectories.

![RCurve](./images/BoxR_curve.png)
*Figure 5: Recall-Confidence Curve. The decline in recall at higher thresholds is compensated by the multi-stage association logic of ByteTrack.*

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



The configuration achieved a MOTA of 40.4% and an IDF1 of 51.7%. Notably, the inclusion of Camera Motion Compensation and ByteTrack logic resulted in a relatively low number of ID switches (276) compared to baseline methods evaluated in preliminary studies. The high precision (82.7%) suggests that the detector and the association logic effectively filter out background noise. The recall of 51.4% indicates that the interpolation and low-confidence matching successfully recovered a significant portion of pedestrian trajectories that would otherwise be lost due to occlusion.

More detailed table (per video) can be seen below:

*Table II: Tracking performance per video in subset of MOT17 dataset*

| Sequence | IDF1 | IDP | IDR | Rcll | Prcn | GT | MT | PT | ML | FP | FN | IDs | FM | MOTA | MOTP | IDt | IDa | IDm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MOT17-02-DPM | 43.6% | 58.2% | 34.8% | 45.7% | 76.4% | 62 | 10 | 28 | 24 | 2628 | 10086 | 65 | 118 | 31.2% | 0.199 | 38 | 33 | 9 |
| MOT17-02-FRCNN | 43.6% | 58.2% | 34.8% | 45.7% | 76.4% | 62 | 10 | 28 | 24 | 2628 | 10086 | 65 | 118 | 31.2% | 0.199 | 38 | 33 | 9 |
| MOT17-02-SDP | 43.6% | 58.2% | 34.8% | 45.7% | 76.4% | 62 | 10 | 28 | 24 | 2628 | 10086 | 65 | 118 | 31.2% | 0.199 | 38 | 33 | 9 |
| MOT17-13-DPM | 64.1% | 80.7% | 53.2% | 60.6% | 91.9% | 110 | 36 | 51 | 23 | 620 | 4590 | 27 | 81 | 55.0% | 0.223 | 30 | 18 | 23 |
| MOT17-13-FRCNN | 64.1% | 80.7% | 53.2% | 60.6% | 91.9% | 110 | 36 | 51 | 23 | 620 | 4590 | 27 | 81 | 55.0% | 0.223 | 30 | 18 | 23 |
| MOT17-13-SDP | 64.1% | 80.7% | 53.2% | 60.6% | 91.9% | 110 | 36 | 51 | 23 | 620 | 4590 | 27 | 81 | 55.0% | 0.223 | 30 | 18 | 23 |
| OVERALL | 51.7% | 67.4% | 41.9% | 51.4% | 82.7% | 516 | 138 | 237 | 141 | 9744 | 44028 | 276 | 597 | 40.4% | 0.210 | 204 | 153 | 96 |

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


Increasing the resolution to 1920px resulted in marginal gains in the primary composite metrics, with MOTA increasing by only 0.3% and IDF1 by 0.4%.

Interestingly, the recall decreased slightly (from 51.4% to 50.5%), while precision improved (from 82.7% to 84.2%). This suggests that while the higher resolution allows the model to be more selective—reducing False Positives by over 1,000—it also leads to the suppression of some valid detections, possibly due to changes in the effective anchor scales or confidence calibration at higher resolutions. Furthermore, the number of ID switches increased from 276 to 306, indicating slightly less stable identity maintenance.


We conclude that the configuration from **Experiment 10 (1280px)** is superior for practical deployment. The marginal improvement in MOTA observed in Experiment 11 does not justify the significantly higher computational cost associated with processing $1920 \times 1920$ images (a $2.25\times$ increase in pixel count). The 1280px setup offers a better balance between recall and precision, maintains more stable identities (fewer ID switches), and operates with greater computational efficiency, making it the better choice for autonomous driving systems where real-time processing is a constraint.



## C. Evaluation on Public Detections

### 1) Experimental Setup
To assess the tracker's performance within the constraints of the MOT17 "Public Detection" track, we decoupled the YOLO11x detector and utilized the standard detections provided by the benchmark. These detections originate from three different detectors: DPM, FRCNN, and SDP, and are provided for the test set sequences. The tracking pipeline utilized the identical configuration as Experiment 10 (OC-SORT + ByteTrack + CMC + Interpolation), ensuring a direct evaluation of the tracker's capability to handle diverse detection qualities. The system achieved an average processing speed of 31.08 Hz during this evaluation.

### 2) Quantitative Results
The results on the MOT17 test set using public detections are summarized in Table III. The metrics represent the combined performance across all sequences and detector types.

**Table III: Tracking Performance on MOT17 Test Set (Public Detections)**

| Metric | Value | Description |
| :--- | :--- | :--- |
| **MOTA** | **46.6%** | Multiple Object Tracking Accuracy |
| **IDF1** | **51.0%** | ID F1 Score |
| **HOTA** | 40.0% | Higher Order Tracking Accuracy |
| **Recall** | 50.2% | Percentage of ground truth objects detected |
| **Precision** | 93.8% | Percentage of detected objects that are relevant |
| **ID Switches** | 1,692 | Number of identity jumps |
| **MT** | 357 | Mostly Tracked trajectories |
| **ML** | 1,011 | Mostly Lost trajectories |
| **FP** | 18,638 | False Positives |
| **FN** | 281,106 | False Negatives |

### 3) Result Analysis
The proposed tracking pipeline demonstrates robust performance even when relying on external public detections, achieving a MOTA of 46.6% and an IDF1 of 51.0%.


Interestingly, the MOTA score on the test set with public detections (46.6%) is higher than that achieved on the training set with the YOLO11x detector (40.4% in Experiment 10). This difference is largely driven by the significantly higher Precision (93.8% vs. 82.7%), indicating that the provided public detections—while potentially missing more objects (Recall 50.2%)—contain fewer false positives than our fine-tuned YOLO model.


The tracker maintained a high average processing speed of 31.08 Hz, confirming its suitability for real-time applications. The IDF1 score of 51.0% remains consistent with the custom detector experiments (51.7%), suggesting that the tracker's ability to maintain identity is stable regardless of the detection source. However, the absolute number of ID switches (1,692) is higher, likely due to the larger size and diversity of the test set compared to the validation subset used in previous experiments, as well as the inherent noise in older detectors like DPM.

# III. CONCLUSION

In this paper, we presented a comprehensive experimental study of multiple object tracking pipelines evaluated on the MOT17 benchmark, with a particular focus on OC-SORT-based architectures. Through a series of controlled experiments, we analyzed the impact of detector configuration, data association strategies, camera motion compensation, and post-processing techniques on both tracking accuracy and identity consistency.

The results demonstrate that OC-SORT, when combined with carefully selected enhancements, consistently outperforms baseline tracking approaches. While the baseline OC-SORT configuration already provides stable performance with relatively low numbers of false positives, its recall and identity consistency are limited. Significant improvements are achieved by integrating ByteTrack-style association, trajectory interpolation, and camera motion compensation, which together address the most common failure modes in crowded urban scenes.

The best-performing configuration—based on YOLO11x detection, OC-SORT tracking, ByteTrack association logic, linear interpolation, and optical-flow-based camera motion compensation—achieved IDF1 values above 50% and MOTA above 40%, exceeding commonly reported reference baselines on MOT17. These results confirm that a hybrid approach combining classical motion models with targeted heuristics remains highly competitive, even when compared to more complex end-to-end learning-based solutions.

A comparative evaluation of inference resolution revealed that increasing input resolution beyond 1280×1280 pixels yields only marginal gains in tracking metrics while substantially increasing computational cost. This finding highlights the importance of balancing accuracy and efficiency, particularly for real-time deployment scenarios such as autonomous driving systems.

Despite the achieved improvements, several limitations remain. Identity switches still occur in prolonged occlusions and dense interactions, and recall remains constrained by detector performance in challenging lighting and scale conditions. Future work will therefore focus on improving re-identification robustness, exploring adaptive confidence thresholds, and investigating learning-based association modules that could further reduce identity fragmentation without sacrificing efficiency.

Overall, the presented study provides practical insights into the design of robust MOT pipelines and demonstrates that well-engineered tracking-by-detection systems remain a strong and viable solution for real-world perception tasks.
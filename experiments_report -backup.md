# Experiment Report

This document summarizes the results of 18 tracking experiments conducted on the MOT17 dataset.

## 1. Baseline Norfair (20260107_182443)

**Configuration:**
*   **Model:** yolo11l.pt
*   **Tracker:** Norfair
*   **Hyperparameters:**
    *   CONFIDENCE_THRESHOLD: 0.4
    *   DISTANCE_THRESHOLD: 30
    *   HIT_INERTIA_MIN: 3
    *   HIT_INERTIA_MAX: 6

**Results:**

```text

Results:
                IDF1   IDP   IDR  Rcll  Prcn  GT MT  PT  ML   FP    FN IDs   FM  MOTA  MOTP IDt IDa IDm
MOT17-02-DPM   26.7% 61.9% 17.0% 24.3% 88.3%  62  7  14  41  597 14064  53   66 20.8% 0.155   3  51   1
MOT17-02-FRCNN 26.7% 61.9% 17.0% 24.3% 88.3%  62  7  14  41  597 14064  53   66 20.8% 0.155   3  51   1
MOT17-02-SDP   26.7% 61.9% 17.0% 24.3% 88.3%  62  7  14  41  597 14064  53   66 20.8% 0.155   3  51   1
MOT17-13-DPM   35.7% 70.0% 23.9% 28.6% 83.5% 110  7  36  67  657  8317  70  162 22.3% 0.213  44  39  16
MOT17-13-FRCNN 35.7% 70.0% 23.9% 28.6% 83.5% 110  7  36  67  657  8317  70  162 22.3% 0.213  44  39  16
MOT17-13-SDP   35.7% 70.0% 23.9% 28.6% 83.5% 110  7  36  67  657  8317  70  162 22.3% 0.213  44  39  16
OVERALL        30.3% 65.4% 19.7% 25.9% 86.2% 516 42 150 324 3762 67143 369  684 21.4% 0.179 141 270  51

```

---

## 2. DeepSORT (20260118_125316)

**Configuration:**
*   **Model:** yolo11x.pt
*   **Tracker:** DeepSORT (deep-sort-realtime)
*   **Hyperparameters:**
    *   CONFIDENCE_THRESHOLD: 0.5
    *   INFERENCE_SIZE: 1920
    *   MAX_AGE: 60
    *   MIN_HITS: 3
    *   POST_PROCESSING: Linear Interpolation (max_gap=20)
    *   CMC:  NOT applied

**Results:**

```text

Results:
                IDF1   IDP   IDR  Rcll  Prcn  GT  MT  PT  ML    FP    FN IDs    FM  MOTA  MOTP IDt IDa IDm
MOT17-02-DPM   36.9% 44.5% 31.4% 46.3% 65.6%  62  12  30  20  4522  9975 115   178 21.4% 0.218 114  19  22
MOT17-02-FRCNN 36.9% 44.5% 31.4% 46.3% 65.6%  62  12  30  20  4522  9975 115   178 21.4% 0.218 114  19  22
MOT17-02-SDP   36.9% 44.5% 31.4% 46.3% 65.6%  62  12  30  20  4522  9975 115   178 21.4% 0.218 114  19  22
MOT17-13-DPM   38.1% 36.2% 40.1% 54.6% 49.3% 110  23  50  37  6539  5280  76   191 -2.2% 0.235  93  21  44
MOT17-13-FRCNN 38.1% 36.2% 40.1% 54.6% 49.3% 110  23  50  37  6539  5280  76   191 -2.2% 0.235  93  21  44
MOT17-13-SDP   38.1% 36.2% 40.1% 54.6% 49.3% 110  23  50  37  6539  5280  76   191 -2.2% 0.235  93  21  44
OVERALL        37.4% 40.4% 34.8% 49.5% 57.5% 516 105 240 171 33183 45765 573  1107 12.3% 0.225 621 120 198

```

---

## 3. OC-SORT Baseline 1 (20260108_113715)

**Configuration:**
*   **Model:** yolo11l.pt
*   **Tracker:** OC-SORT 
*   **Hyperparameters:**
    *   IOU_THRESH: 0.3
    *   MAX_AGE: 30
    *   MIN_HITS: 3
    *   INERTIA: 0.2

**Results:**

```text

Results:
                IDF1   IDP   IDR  Rcll  Prcn  GT MT  PT  ML   FP    FN IDs   FM  MOTA  MOTP IDt IDa IDm
MOT17-02-DPM   28.1% 71.8% 17.5% 22.4% 91.9%  62  7  13  42  366 14416  30   88 20.3% 0.149   3  26   0
MOT17-02-FRCNN 28.1% 71.8% 17.5% 22.4% 91.9%  62  7  13  42  366 14416  30   88 20.3% 0.149   3  26   0
MOT17-02-SDP   28.1% 71.8% 17.5% 22.4% 91.9%  62  7  13  42  366 14416  30   88 20.3% 0.149   3  26   0
MOT17-13-DPM   36.9% 82.5% 23.8% 27.4% 94.9% 110 12  32  66  170  8457  41  142 25.5% 0.205  16  32   8
MOT17-13-FRCNN 36.9% 82.5% 23.8% 27.4% 94.9% 110 12  32  66  170  8457  41  142 25.5% 0.205  16  32   8
MOT17-13-SDP   36.9% 82.5% 23.8% 27.4% 94.9% 110 12  32  66  170  8457  41  142 25.5% 0.205  16  32   8
OVERALL        31.6% 76.3% 19.9% 24.3% 93.2% 516 57 135 324 1608 68619 213  690 22.3% 0.173  57 174  24

```

---

## 4. OC-SORT Baseline 2 (20260108_114515)

**Configuration:**
*   **Model:** yolo11l.pt
*   **Tracker:** OC-SORT
*   **Hyperparameters:**
    *   IOU_THRESH: 0.3
    *   MAX_AGE: 30
    *   MIN_HITS: 3
    *   INERTIA: 0.2

**Results:**

```text
Results:
                IDF1   IDP   IDR  Rcll  Prcn  GT MT  PT  ML   FP    FN IDs   FM  MOTA  MOTP IDt IDa IDm
MOT17-02-DPM   28.1% 71.8% 17.5% 22.4% 91.9%  62  7  13  42  366 14416  30   88 20.3% 0.149   3  26   0
MOT17-02-FRCNN 28.1% 71.8% 17.5% 22.4% 91.9%  62  7  13  42  366 14416  30   88 20.3% 0.149   3  26   0
MOT17-02-SDP   28.1% 71.8% 17.5% 22.4% 91.9%  62  7  13  42  366 14416  30   88 20.3% 0.149   3  26   0
MOT17-13-DPM   37.0% 82.6% 23.8% 27.3% 94.9% 110 12  32  66  170  8459  40  142 25.5% 0.205  16  31   8
MOT17-13-FRCNN 37.0% 82.6% 23.8% 27.3% 94.9% 110 12  32  66  170  8459  40  142 25.5% 0.205  16  31   8
MOT17-13-SDP   37.0% 82.6% 23.8% 27.3% 94.9% 110 12  32  66  170  8459  40  142 25.5% 0.205  16  31   8
OVERALL        31.6% 76.4% 19.9% 24.3% 93.2% 516 57 135 324 1608 68625 210  690 22.3% 0.173  57 171  24


```

---

## 5. OC-SORT Tuned (20260113_115533)

**Configuration:**
*   **Model:** yolo11l.pt
*   **Tracker:** OC-SORT 
*   **Hyperparameters:**
    *   CONFIDENCE_THRESHOLD: 0.25
    *   INFERENCE_SIZE: 1280
    *   IOU_THRESHOLD: 0.3
    *   MAX_AGE: 30
    *   MIN_HITS: 3
    *   INERTIA: 0.2

**Results:**

```text
Results:
                IDF1   IDP   IDR  Rcll  Prcn  GT MT  PT  ML   FP    FN IDs    FM  MOTA  MOTP IDt IDa IDm
MOT17-02-DPM   38.5% 58.6% 28.6% 39.7% 81.2%  62  7  29  26 1710 11211  97   239 29.9% 0.187  26  66   2
MOT17-02-FRCNN 38.5% 58.6% 28.6% 39.7% 81.2%  62  7  29  26 1710 11211  97   239 29.9% 0.187  26  66   2
MOT17-02-SDP   38.5% 58.6% 28.6% 39.7% 81.2%  62  7  29  26 1710 11211  97   239 29.9% 0.187  26  66   2
MOT17-13-DPM   51.3% 72.3% 39.7% 50.5% 91.9% 110 24  49  37  519  5766  98   246 45.2% 0.222  49  64  21
MOT17-13-FRCNN 51.3% 72.3% 39.7% 50.5% 91.9% 110 24  49  37  519  5766  98   246 45.2% 0.222  49  64  21
MOT17-13-SDP   51.3% 72.3% 39.7% 50.5% 91.9% 110 24  49  37  519  5766  98   246 45.2% 0.222  49  64  21
OVERALL        43.5% 64.3% 32.9% 43.8% 85.6% 516 93 234 189 6687 50931 585  1455 35.8% 0.202 225 390  69

```

---

## 6. OC-SORT + ByteTrack (High Conf 0.6) (20260113_124033)

**Configuration:**
*   **Model:** yolo11l.pt
*   **Tracker:** OC-SORT + ByteTrack Logic
*   **Hyperparameters:**
    *   CONFIDENCE_THRESHOLD (High): 0.6
    *   CONFIDENCE_LOW: 0.1
    *   INFERENCE_SIZE: 1280
    *   IOU_THRESHOLD: 0.3
    *   MAX_AGE: 30
    *   MIN_HITS: 3
    *   INERTIA: 0.2

**Results:**

```text
Results:
                IDF1   IDP   IDR  Rcll  Prcn  GT MT  PT  ML   FP    FN IDs   FM  MOTA  MOTP IDt IDa IDm
MOT17-02-DPM   41.5% 71.4% 29.2% 36.2% 88.4%  62  7  26  29  884 11862  34  143 31.2% 0.175   6  30   2
MOT17-02-FRCNN 41.5% 71.4% 29.2% 36.2% 88.4%  62  7  26  29  884 11862  34  143 31.2% 0.175   6  30   2
MOT17-02-SDP   41.5% 71.4% 29.2% 36.2% 88.4%  62  7  26  29  884 11862  34  143 31.2% 0.175   6  30   2
MOT17-13-DPM   51.4% 81.8% 37.5% 43.0% 94.0% 110 22  42  46  318  6632  34  107 40.0% 0.212  20  28  14
MOT17-13-FRCNN 51.4% 81.8% 37.5% 43.0% 94.0% 110 22  42  46  318  6632  34  107 40.0% 0.212  20  28  14
MOT17-13-SDP   51.4% 81.8% 37.5% 43.0% 94.0% 110 22  42  46  318  6632  34  107 40.0% 0.212  20  28  14
OVERALL        45.4% 75.7% 32.4% 38.8% 90.7% 516 87 204 225 3606 55482 204  750 34.6% 0.191  78 174  48

```

---

## 7. OC-SORT + ByteTrack (High Conf 0.4) (20260113_125154)

**Configuration:**
*   **Model:** yolo11l.pt
*   **Tracker:** OC-SORT + ByteTrack Logic
*   **Hyperparameters:**
    *   CONFIDENCE_THRESHOLD (High): 0.4
    *   CONFIDENCE_LOW: 0.1
    *   INFERENCE_SIZE: 1280
    *   IOU_THRESHOLD: 0.3
    *   MAX_AGE: 30
    *   MIN_HITS: 3
    *   INERTIA: 0.2

**Results:**

```text
Results:
                IDF1   IDP   IDR  Rcll  Prcn  GT  MT  PT  ML   FP    FN IDs    FM  MOTA  MOTP IDt IDa IDm
MOT17-02-DPM   40.9% 62.9% 30.3% 39.4% 82.0%  62   8  26  28 1612 11257  72   216 30.4% 0.186  22  50   3
MOT17-02-FRCNN 40.9% 62.9% 30.3% 39.4% 82.0%  62   8  26  28 1612 11257  72   216 30.4% 0.186  22  50   3
MOT17-02-SDP   40.9% 62.9% 30.3% 39.4% 82.0%  62   8  26  28 1612 11257  72   216 30.4% 0.186  22  50   3
MOT17-13-DPM   52.9% 75.4% 40.8% 50.1% 92.5% 110  26  44  40  474  5814  72   206 45.4% 0.222  50  38  18
MOT17-13-FRCNN 52.9% 75.4% 40.8% 50.1% 92.5% 110  26  44  40  474  5814  72   206 45.4% 0.222  50  38  18
MOT17-13-SDP   52.9% 75.4% 40.8% 50.1% 92.5% 110  26  44  40  474  5814  72   206 45.4% 0.222  50  38  18
OVERALL        45.6% 68.1% 34.3% 43.5% 86.3% 516 102 210 204 6258 51213 432  1266 36.1% 0.202 216 264  63

```

---

## 8. OC-SORT + ByteTrack (Max Age 60) (20260113_130316)

**Configuration:**
*   **Model:** yolo11l.pt
*   **Tracker:** OC-SORT + ByteTrack Logic
*   **Hyperparameters:**
    *   CONFIDENCE_THRESHOLD (High): 0.4
    *   CONFIDENCE_LOW: 0.05
    *   INFERENCE_SIZE: 1280
    *   IOU_THRESHOLD: 0.3
    *   MAX_AGE: 60
    *   MIN_HITS: 3
    *   INERTIA: 0.2

**Results:**

```text
Results:
                IDF1   IDP   IDR  Rcll  Prcn  GT MT  PT  ML   FP    FN IDs    FM  MOTA  MOTP IDt IDa IDm
MOT17-02-DPM   42.0% 64.6% 31.1% 39.4% 81.9%  62  8  26  28 1618 11256  68   215 30.3% 0.186  26  42   3
MOT17-02-FRCNN 42.0% 64.6% 31.1% 39.4% 81.9%  62  8  26  28 1618 11256  68   215 30.3% 0.186  26  42   3
MOT17-02-SDP   42.0% 64.6% 31.1% 39.4% 81.9%  62  8  26  28 1618 11256  68   215 30.3% 0.186  26  42   3
MOT17-13-DPM   52.9% 75.5% 40.7% 49.9% 92.5% 110 25  44  41  471  5832  76   206 45.2% 0.221  53  41  21
MOT17-13-FRCNN 52.9% 75.5% 40.7% 49.9% 92.5% 110 25  44  41  471  5832  76   206 45.2% 0.221  53  41  21
MOT17-13-SDP   52.9% 75.5% 40.7% 49.9% 92.5% 110 25  44  41  471  5832  76   206 45.2% 0.221  53  41  21
OVERALL        46.3% 69.1% 34.8% 43.5% 86.3% 516 99 210 207 6267 51264 432  1263 36.1% 0.201 237 249  72

```

---

## 9. OC-SORT + ByteTrack + Interpolation (20260113_135953)

**Configuration:**
*   **Model:** yolo11x.pt
*   **Tracker:** OC-SORT + ByteTrack Logic + Interpolation
*   **Hyperparameters:**
    *   CONFIDENCE_THRESHOLD (High): 0.4
    *   CONFIDENCE_LOW: 0.05
    *   INFERENCE_SIZE: 1280
    *   IOU_THRESHOLD: 0.3
    *   MAX_AGE: 60
    *   MIN_HITS: 3
    *   INERTIA: 0.2
    *   POST_PROCESSING: Linear Interpolation (max_gap=20)

**Results:**

```text

Results:
                IDF1   IDP   IDR  Rcll  Prcn  GT  MT  PT  ML    FP    FN IDs   FM  MOTA  MOTP IDt IDa IDm
MOT17-02-DPM   42.7% 55.3% 34.8% 46.9% 74.7%  62  10  30  22  2959  9862  80  123 30.6% 0.203  34  46   3
MOT17-02-FRCNN 42.7% 55.3% 34.8% 46.9% 74.7%  62  10  30  22  2959  9862  80  123 30.6% 0.203  34  46   3
MOT17-02-SDP   42.7% 55.3% 34.8% 46.9% 74.7%  62  10  30  22  2959  9862  80  123 30.6% 0.203  34  46   3
MOT17-13-DPM   57.5% 71.5% 48.0% 60.0% 89.4% 110  33  49  28   831  4657  70  112 52.3% 0.225  46  42  19
MOT17-13-FRCNN 57.5% 71.5% 48.0% 60.0% 89.4% 110  33  49  28   831  4657  70  112 52.3% 0.225  46  42  19
MOT17-13-SDP   57.5% 71.5% 48.0% 60.0% 89.4% 110  33  49  28   831  4657  70  112 52.3% 0.225  46  42  19
OVERALL        48.5% 61.8% 39.9% 52.0% 80.6% 516 129 237 150 11370 43557 450  705 38.9% 0.213 240 264  66
```

---

## 10. OC-SORT + ByteTrack + Interp + CMC (20260113_152122)

**Configuration:**
*   **Model:** yolo11x.pt
*   **Tracker:** OC-SORT + ByteTrack + Interpolation + CMC
*   **Hyperparameters:**
    *   CONFIDENCE_THRESHOLD (High): 0.5
    *   CONFIDENCE_LOW: 0.05
    *   INFERENCE_SIZE: 1280
    *   IOU_THRESHOLD: 0.3
    *   MAX_AGE: 60
    *   MIN_HITS: 3
    *   INERTIA: 0.2
    *   POST_PROCESSING: Linear Interpolation (max_gap=20)
    *   CMC: Optical Flow (Affine)

**Results:**

```text

Results:
                IDF1   IDP   IDR  Rcll  Prcn  GT  MT  PT  ML   FP    FN IDs   FM  MOTA  MOTP IDt IDa IDm
MOT17-02-DPM   43.6% 58.2% 34.8% 45.7% 76.4%  62  10  28  24 2628 10086  65  118 31.2% 0.199  38  33   9
MOT17-02-FRCNN 43.6% 58.2% 34.8% 45.7% 76.4%  62  10  28  24 2628 10086  65  118 31.2% 0.199  38  33   9
MOT17-02-SDP   43.6% 58.2% 34.8% 45.7% 76.4%  62  10  28  24 2628 10086  65  118 31.2% 0.199  38  33   9
MOT17-13-DPM   64.1% 80.7% 53.2% 60.6% 91.9% 110  36  51  23  620  4590  27   81 55.0% 0.223  30  18  23
MOT17-13-FRCNN 64.1% 80.7% 53.2% 60.6% 91.9% 110  36  51  23  620  4590  27   81 55.0% 0.223  30  18  23
MOT17-13-SDP   64.1% 80.7% 53.2% 60.6% 91.9% 110  36  51  23  620  4590  27   81 55.0% 0.223  30  18  23
OVERALL        51.7% 67.4% 41.9% 51.4% 82.7% 516 138 237 141 9744 44028 276  597 40.4% 0.210 204 153  96

```

---

## 11. OC-SORT + ByteTrack + Interp + CMC (1920px) (20260118_120603)

**Configuration:**
*   **Model:** yolo11x.pt
*   **Tracker:** OC-SORT + ByteTrack + Interpolation + CMC
*   **Hyperparameters:**
    *   CONFIDENCE_THRESHOLD (High): 0.5
    *   CONFIDENCE_LOW: 0.05
    *   INFERENCE_SIZE: 1920
    *   IOU_THRESHOLD: 0.3
    *   MAX_AGE: 60
    *   MIN_HITS: 3
    *   INERTIA: 0.2
    *   POST_PROCESSING: Linear Interpolation (max_gap=20)
    *   CMC: Optical Flow (Affine)

**Results:**

```text
    Results:
                IDF1   IDP   IDR  Rcll  Prcn  GT  MT  PT  ML   FP    FN IDs   FM  MOTA  MOTP IDt IDa IDm
MOT17-02-DPM   44.8% 64.3% 34.4% 43.2% 80.9%  62   7  30  25 1902 10547  70  136 32.6% 0.209  28  39   3
MOT17-02-FRCNN 44.8% 64.3% 34.4% 43.2% 80.9%  62   7  30  25 1902 10547  70  136 32.6% 0.209  28  39   3
MOT17-02-SDP   44.8% 64.3% 34.4% 43.2% 80.9%  62   7  30  25 1902 10547  70  136 32.6% 0.209  28  39   3
MOT17-13-DPM   62.7% 75.9% 53.4% 62.1% 88.2% 110  33  55  22  968  4410  32   86 53.5% 0.228  35  22  25
MOT17-13-FRCNN 62.7% 75.9% 53.4% 62.1% 88.2% 110  33  55  22  968  4410  32   86 53.5% 0.228  35  22  25
MOT17-13-SDP   62.7% 75.9% 53.4% 62.1% 88.2% 110  33  55  22  968  4410  32   86 53.5% 0.228  35  22  25
OVERALL        52.1% 69.5% 41.7% 50.5% 84.2% 516 120 255 141 8610 44871 306  666 40.7% 0.218 189 183  84

```
---

## 12. OC-SORT + ByteTrack + GSI + CMC (20260118_140050)

**Configuration:**
*   **Model:** yolo11x.pt
*   **Tracker:** OC-SORT + ByteTrack + GSI (Gaussian Smoothing) + CMC
*   **Hyperparameters:**
    *   CONFIDENCE_THRESHOLD (High): 0.5
    *   CONFIDENCE_LOW: 0.05
    *   INFERENCE_SIZE: 1920
    *   IOU_THRESHOLD: 0.3
    *   MAX_AGE: 60
    *   MIN_HITS: 3
    *   INERTIA: 0.2
    *   POST_PROCESSING: Gaussian Smoothed Interpolation (sigma=1.0)
    *   CMC: Optical Flow (Affine)

**Results:**

```text
Results:
                IDF1   IDP   IDR  Rcll  Prcn  GT  MT  PT  ML   FP    FN IDs   FM  MOTA  MOTP IDt IDa IDm
MOT17-02-DPM   44.9% 64.4% 34.4% 43.2% 80.9%  62   7  30  25 1900 10545  70  124 32.6% 0.208  28  38   3
MOT17-02-FRCNN 44.9% 64.4% 34.4% 43.2% 80.9%  62   7  30  25 1900 10545  70  124 32.6% 0.208  28  38   3
MOT17-02-SDP   44.9% 64.4% 34.4% 43.2% 80.9%  62   7  30  25 1900 10545  70  124 32.6% 0.208  28  38   3
MOT17-13-DPM   62.4% 75.4% 53.1% 61.8% 87.7% 110  33  53  24 1008  4450  35   88 52.8% 0.227  39  22  26
MOT17-13-FRCNN 62.4% 75.4% 53.1% 61.8% 87.7% 110  33  53  24 1008  4450  35   88 52.8% 0.227  39  22  26
MOT17-13-SDP   62.4% 75.4% 53.1% 61.8% 87.7% 110  33  53  24 1008  4450  35   88 52.8% 0.227  39  22  26
OVERALL        52.1% 69.4% 41.6% 50.4% 84.0% 516 120 249 147 8724 44985 315  636 40.4% 0.217 201 180  87
```

---

## 13. OC-SORT + ByteTrack + GSI + CMC Optimized (20260118_143636)

**Configuration:**
*   **Model:** yolo11x.pt
*   **Tracker:** OC-SORT + ByteTrack + GSI (Gaussian Smoothing) + CMC
*   **Hyperparameters:**
    *   CONFIDENCE_THRESHOLD (High): 0.3
    *   CONFIDENCE_LOW: 0.05
    *   INFERENCE_SIZE: 1920
    *   IOU_THRESHOLD: 0.25
    *   MAX_AGE: 90
    *   MIN_HITS: 3
    *   INERTIA: 0.2
    *   POST_PROCESSING: Gaussian Smoothed Interpolation (sigma=0.6)
    *   CMC: Optical Flow (Affine)

**Results:**

```text
Results:
                IDF1   IDP   IDR  Rcll  Prcn  GT  MT  PT  ML    FP    FN IDs   FM  MOTA  MOTP IDt IDa IDm
MOT17-02-DPM   40.8% 55.3% 32.4% 44.8% 76.6%  62   9  31  22  2546 10256 101  148 30.6% 0.213  51  49   9
MOT17-02-FRCNN 40.8% 55.3% 32.4% 44.8% 76.6%  62   9  31  22  2546 10256 101  148 30.6% 0.213  51  49   9
MOT17-02-SDP   40.8% 55.3% 32.4% 44.8% 76.6%  62   9  31  22  2546 10256 101  148 30.6% 0.213  51  49   9
MOT17-13-DPM   61.5% 69.9% 54.9% 65.9% 84.0% 110  37  56  17  1464  3974  48  118 52.9% 0.232  47  31  31
MOT17-13-FRCNN 61.5% 69.9% 54.9% 65.9% 84.0% 110  37  56  17  1464  3974  48  118 52.9% 0.232  47  31  31
MOT17-13-SDP   61.5% 69.9% 54.9% 65.9% 84.0% 110  37  56  17  1464  3974  48  118 52.9% 0.232  47  31  31
OVERALL        49.4% 62.0% 41.0% 52.9% 80.0% 516 138 261 117 12030 42690 447  798 39.2% 0.222 294 240 120

```

---

## 14. OC-SORT + ByteTrack + GSI + CMC Optimized 2 (20260118_150721)

**Configuration:**
*   **Model:** yolo11x.pt
*   **Tracker:** OC-SORT + ByteTrack + GSI (Gaussian Smoothing) + CMC
*   **Hyperparameters:**
    *   CONFIDENCE_THRESHOLD (High): 0.45
    *   CONFIDENCE_LOW: 0.05
    *   INFERENCE_SIZE: 1920
    *   IOU_THRESHOLD: 0.25
    *   MAX_AGE: 60
    *   MIN_HITS: 3
    *   INERTIA: 0.2
    *   POST_PROCESSING: Gaussian Smoothed Interpolation (sigma=1.0)
    *   CMC: Optical Flow (Affine)

**Results:**

```text
Results:
                IDF1   IDP   IDR  Rcll  Prcn  GT  MT  PT  ML    FP    FN IDs   FM  MOTA  MOTP IDt IDa IDm
MOT17-02-DPM   46.2% 64.3% 36.0% 44.3% 79.1%  62   9  29  24  2168 10351  75  126 32.2% 0.211  43  32   8
MOT17-02-FRCNN 46.2% 64.3% 36.0% 44.3% 79.1%  62   9  29  24  2168 10351  75  126 32.2% 0.211  43  32   8
MOT17-02-SDP   46.2% 64.3% 36.0% 44.3% 79.1%  62   9  29  24  2168 10351  75  126 32.2% 0.211  43  32   8
MOT17-13-DPM   62.4% 73.5% 54.3% 63.6% 86.2% 110  34  56  20  1183  4236  39   98 53.1% 0.230  45  18  25
MOT17-13-FRCNN 62.4% 73.5% 54.3% 63.6% 86.2% 110  34  56  20  1183  4236  39   98 53.1% 0.230  45  18  25
MOT17-13-SDP   62.4% 73.5% 54.3% 63.6% 86.2% 110  34  56  20  1183  4236  39   98 53.1% 0.230  45  18  25
OVERALL        52.9% 68.5% 43.0% 51.7% 82.4% 516 129 255 132 10053 43761 342  672 40.3% 0.220 264 150  99

```

---

## 15. SAM2 + OC-SORT (20260109_191300)

**Configuration:**
*   **Detector:** SAM2 (Automatic Mask Generator)
*   **Tracker:** OC-SORT
*   **Hyperparameters:** **please check this** (Hyperparameters were not listed in the provided log file)

**Results:**

```text

Results:
                IDF1   IDP   IDR  Rcll  Prcn  GT MT PT  ML    FP    FN IDs   FM   MOTA  MOTP IDt IDa IDm
MOT17-02-DPM   16.8% 22.1% 13.5% 15.4% 25.1%  62  3 12  47  8529 15719  23  106 -30.6% 0.153   3  20   2
MOT17-02-FRCNN 16.8% 22.1% 13.5% 15.4% 25.1%  62  3 12  47  8529 15719  23  106 -30.6% 0.153   3  20   2
MOT17-02-SDP   16.8% 22.1% 13.5% 15.4% 25.1%  62  3 12  47  8529 15719  23  106 -30.6% 0.153   3  20   2
MOT17-13-DPM   10.1% 10.5%  9.6% 10.1% 11.1% 110  4 17  89  9431 10467  18  112 -71.1% 0.204   2  15   2
MOT17-13-FRCNN 10.1% 10.5%  9.6% 10.1% 11.1% 110  4 17  89  9431 10467  18  112 -71.1% 0.204   2  15   2
MOT17-13-SDP   10.1% 10.5%  9.6% 10.1% 11.1% 110  4 17  89  9431 10467  18  112 -71.1% 0.204   2  15   2
OVERALL        13.9% 16.5% 12.0% 13.4% 18.4% 516 21 87 408 53880 78558 123  654 -46.2% 0.168  15 105  12

```

---

## 16. Trained Norfair (20260108_101135)

**Configuration:**
*   **Model:** yolo11l.pt (Finetuned on MOT17 all classes)
*   **Tracker:** Norfair
*   **Hyperparameters:**
    *   EPOCHS: 10
    *   IMG_SIZE: 640
    *   BATCH_SIZE: 4
    *   CONFIDENCE_THRESHOLD: 0.4
    *   DISTANCE_THRESHOLD: 30
    *   HIT_INERTIA_MIN: 3
    *   HIT_INERTIA_MAX: 6

**Results:**

```text

Results:
                IDF1   IDP   IDR  Rcll  Prcn  GT MT  PT  ML   FP    FN IDs   FM  MOTA  MOTP IDt IDa IDm
MOT17-02-DPM   27.1% 54.2% 18.0% 25.5% 76.6%  62  9  13  40 1446 13845  50   60 17.4% 0.181   7  41   2
MOT17-02-FRCNN 27.1% 54.2% 18.0% 25.5% 76.6%  62  9  13  40 1446 13845  50   60 17.4% 0.181   7  41   2
MOT17-02-SDP   27.1% 54.2% 18.0% 25.5% 76.6%  62  9  13  40 1446 13845  50   60 17.4% 0.181   7  41   2
MOT17-13-DPM   24.6% 56.0% 15.7% 19.1% 67.9% 110  6  25  79 1051  9421  37   89  9.7% 0.254  26  23  13
MOT17-13-FRCNN 24.6% 56.0% 15.7% 19.1% 67.9% 110  6  25  79 1051  9421  37   89  9.7% 0.254  26  23  13
MOT17-13-SDP   24.6% 56.0% 15.7% 19.1% 67.9% 110  6  25  79 1051  9421  37   89  9.7% 0.254  26  23  13
OVERALL        26.1% 54.8% 17.1% 23.0% 73.6% 516 45 114 357 7491 69798 261  447 14.5% 0.204  99 192  45
```

---

## 17. Trained Best Tracker (20260113_170416)

**Configuration:**
*   **Model:** yolo11l.pt (Finetuned on MOT17)
*   **Tracker:** OC-SORT + ByteTrack + Interpolation + CMC
*   **Hyperparameters:**
    *   EPOCHS: 10
    *   CONFIDENCE_THRESHOLD: 0.5
    *   CONFIDENCE_LOW: 0.05
    *   MAX_AGE: 60
    *   CMC: Optical Flow (Affine)
    *   POST_PROCESSING: Linear Interpolation (max_gap=20)

**Results:**

```text
Results:
                IDF1   IDP   IDR  Rcll  Prcn  GT MT  PT  ML    FP    FN IDs   FM  MOTA  MOTP IDt IDa IDm
MOT17-02-DPM   35.2% 60.0% 25.0% 27.8% 66.9%  62  9  15  38  2559 13408  31   56 13.9% 0.188  20  18   9
MOT17-02-FRCNN 35.2% 60.0% 25.0% 27.8% 66.9%  62  9  15  38  2559 13408  31   56 13.9% 0.188  20  18   9
MOT17-02-SDP   35.2% 60.0% 25.0% 27.8% 66.9%  62  9  15  38  2559 13408  31   56 13.9% 0.188  20  18   9
MOT17-13-DPM   39.2% 64.8% 28.1% 33.2% 76.5% 110 19  35  56  1184  7780  28   86 22.8% 0.272  36  15  25
MOT17-13-FRCNN 39.2% 64.8% 28.1% 33.2% 76.5% 110 19  35  56  1184  7780  28   86 22.8% 0.272  36  15  25
MOT17-13-SDP   39.2% 64.8% 28.1% 33.2% 76.5% 110 19  35  56  1184  7780  28   86 22.8% 0.272  36  15  25
OVERALL        36.8% 61.9% 26.2% 29.9% 70.7% 516 84 150 282 11229 63564 177  426 17.3% 0.224 168  99 102

```

---

## 18. Trained OC-SORT (20260108_133020)

**Configuration:**
*   **Model:** yolo11l.pt (Finetuned on MOT17 all classes)
*   **Tracker:** OC-SORT (Custom Implementation)
*   **Hyperparameters:**
    *   EPOCHS: 10
    *   IMG_SIZE: 640
    *   BATCH_SIZE: 4
    *   IOU_THRESH: 0.3
    *   MAX_AGE: 30
    *   MIN_HITS: 3
    *   INERTIA: 0.2

**Results:**

```text
Results:
                IDF1   IDP   IDR  Rcll  Prcn  GT MT  PT  ML   FP    FN IDs   FM  MOTA  MOTP IDt IDa IDm
MOT17-02-DPM   31.2% 61.4% 20.9% 26.6% 78.1%  62  8  14  40 1387 13640  44  133 18.9% 0.186  15  32   5
MOT17-02-FRCNN 31.2% 61.4% 20.9% 26.6% 78.1%  62  8  14  40 1387 13640  44  133 18.9% 0.186  15  32   5
MOT17-02-SDP   31.2% 61.4% 20.9% 26.6% 78.1%  62  8  14  40 1387 13640  44  133 18.9% 0.186  15  32   5
MOT17-13-DPM   33.1% 73.2% 21.4% 24.1% 82.4% 110 11  34  65  600  8839  40  142 18.6% 0.260  28  29  17
MOT17-13-FRCNN 33.1% 73.2% 21.4% 24.1% 82.4% 110 11  34  65  600  8839  40  142 18.6% 0.260  28  29  17
MOT17-13-SDP   33.1% 73.2% 21.4% 24.1% 82.4% 110 11  34  65  600  8839  40  142 18.6% 0.260  28  29  17
OVERALL        31.9% 65.5% 21.1% 25.6% 79.6% 516 57 144 315 5961 67437 252  825 18.8% 0.212 129 183  66

```

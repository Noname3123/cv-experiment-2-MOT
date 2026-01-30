---
title: MOT17 Eksperimenti Praćenja Objekata
subtitle: Usporedna Analiza Sustava za Praćenje i Modela
---

## Eksperiment 1: Norfair (Osnovni)
**Zapis:** `experiment_20260107_182443.txt`

**Konfiguracija**
*   **Model:** yolo11l.pt
*   **Sustav za praćenje:** Norfair
*   **Hiperparametri:**
    *   Prag pouzdanosti: 0.4
    *   Prag udaljenosti: 30
    *   Hit Inertia: 3 (min) - 6 (max)

**Rezultati**

| Metrika | Ukupno | MOT17-02 | MOT17-13 |
| :--- | :--- | :--- | :--- |
| **IDF1** | **30.3%** | 26.7% | 35.7% |
| **MOTA** | **21.4%** | 20.8% | 22.3% |
| **ID Sw.** | 369 | 53 | 70 |

## Eksperiment 2: Norfair (Fino podešen)
**Zapis:** `experiment_trained_20260108_101135.txt`

**Konfiguracija**
*   **Model:** yolo11l.pt (Fino podešen na MOT17)
*   **Sustav za praćenje:** Norfair
*   **Hiperparametri:**
    *   Prag pouzdanosti: 0.4
    *   Prag udaljenosti: 30
    *   Hit Inertia: 3 (min) - 6 (max)

**Rezultati**

| Metrika | Ukupno | MOT17-02 | MOT17-13 |
| :--- | :--- | :--- | :--- |
| **IDF1** | **26.1%** | 27.1% | 24.6% |
| **MOTA** | **14.5%** | 17.4% | 9.7% |
| **ID Sw.** | 261 | 50 | 37 |

## Eksperiment 3: OC-SORT (Osnovni)
**Zapis:** `experiment_ocsort_20260108_114515.txt`

**Konfiguracija**
*   **Model:** yolo11l.pt
*   **Sustav za praćenje:** OC-SORT (Vlastita implementacija)
*   **Hiperparametri:**
    *   IoU Prag: 0.3
    *   Maksimalna starost: 30
    *   Minimalni pogodci: 3
    *   Inercija: 0.2

**Rezultati**

| Metrika | Ukupno | MOT17-02 | MOT17-13 |
| :--- | :--- | :--- | :--- |
| **IDF1** | **31.6%** | 28.1% | 37.0% |
| **MOTA** | **22.3%** | 20.3% | 25.5% |
| **ID Sw.** | 210 | 30 | 40 |

## Eksperiment 4: OC-SORT (Fino podešen)
**Zapis:** `experiment_trained_ocsort_20260108_133020.txt`

**Konfiguracija**
*   **Model:** yolo11l.pt (Fino podešen na MOT17)
*   **Sustav za praćenje:** OC-SORT (Vlastita implementacija)
*   **Hiperparametri:**
    *   IoU Prag: 0.3
    *   Maksimalna starost: 30
    *   Minimalni pogodci: 3
    *   Inercija: 0.2

**Rezultati**

| Metrika | Ukupno | MOT17-02 | MOT17-13 |
| :--- | :--- | :--- | :--- |
| **IDF1** | **31.9%** | 31.2% | 33.1% |
| **MOTA** | **18.8%** | 18.9% | 18.6% |
| **ID Sw.** | 252 | 44 | 40 |

## Eksperiment 5: SAM2 + OC-SORT
**Zapis:** `experiment_sam2_ocsort_20260109_191300.txt`

**Konfiguracija**
*   **Detektor:** SAM2 (Automatski generator maski)
*   **Sustav za praćenje:** OC-SORT
*   **Napomena:** Zero-shot segmentacija korištena za detekciju.

**Rezultati**

| Metrika | Ukupno | MOT17-02 | MOT17-13 |
| :--- | :--- | :--- | :--- |
| **IDF1** | **13.9%** | 16.8% | 10.1% |
| **MOTA** | **-46.2%** | -30.6% | -71.1% |
| **ID Sw.** | 123 | 23 | 18 |

## Eksperiment 6: OC-SORT (Optimiziran)
**Zapis:** `experiment_ocsort_20260113_115533.txt`

**Konfiguracija**
*   **Model:** yolo11l.pt
*   **Sustav za praćenje:** OC-SORT
*   **Hiperparametri:**
    *   Prag pouzdanosti: 0.25
    *   Veličina inferencije: 1280
    *   Maksimalna starost: 30

**Rezultati**

| Metrika | Ukupno | MOT17-02 | MOT17-13 |
| :--- | :--- | :--- | :--- |
| **IDF1** | **43.5%** | 38.5% | 51.3% |
| **MOTA** | **35.8%** | 29.9% | 45.2% |
| **ID Sw.** | 585 | 97 | 98 |

## Eksperiment 7: OC-SORT + ByteTrack (Visoka pouzdanost)
**Zapis:** `experiment_ocsort_bytetrack_20260113_124033.txt`

**Konfiguracija**
*   **Model:** yolo11l.pt
*   **Sustav za praćenje:** OC-SORT + ByteTrack logika
*   **Hiperparametri:**
    *   Prag pouzdanosti (Visoki): 0.6
    *   Prag pouzdanosti (Niski): 0.1
    *   Veličina inferencije: 1280

**Rezultati**

| Metrika | Ukupno | MOT17-02 | MOT17-13 |
| :--- | :--- | :--- | :--- |
| **IDF1** | **45.4%** | 41.5% | 51.4% |
| **MOTA** | **34.6%** | 31.2% | 40.0% |
| **ID Sw.** | 204 | 34 | 34 |

## Eksperiment 8: OC-SORT + ByteTrack (Srednja pouzdanost)
**Zapis:** `experiment_ocsort_bytetrack_20260113_125154.txt`

**Konfiguracija**
*   **Model:** yolo11l.pt
*   **Sustav za praćenje:** OC-SORT + ByteTrack logika
*   **Hiperparametri:**
    *   Prag pouzdanosti (Visoki): 0.4
    *   Prag pouzdanosti (Niski): 0.1
    *   Veličina inferencije: 1280

**Rezultati**

| Metrika | Ukupno | MOT17-02 | MOT17-13 |
| :--- | :--- | :--- | :--- |
| **IDF1** | **45.6%** | 40.9% | 52.9% |
| **MOTA** | **36.1%** | 30.4% | 45.4% |
| **ID Sw.** | 432 | 72 | 72 |

## Eksperiment 9: OC-SORT + ByteTrack (Niska pouzdanost)
**Zapis:** `experiment_ocsort_bytetrack_20260113_130316.txt`

**Konfiguracija**
*   **Model:** yolo11l.pt
*   **Sustav za praćenje:** OC-SORT + ByteTrack logika
*   **Hiperparametri:**
    *   Prag pouzdanosti (Visoki): 0.4
    *   Prag pouzdanosti (Niski): 0.05
    *   Maksimalna starost: 60

**Rezultati**

| Metrika | Ukupno | MOT17-02 | MOT17-13 |
| :--- | :--- | :--- | :--- |
| **IDF1** | **46.3%** | 42.0% | 52.9% |
| **MOTA** | **36.1%** | 30.3% | 45.2% |
| **ID Sw.** | 432 | 68 | 76 |

## Eksperiment 10: OC-SORT + Byte + Interp (YOLOv11x)
**Zapis:** `experiment_ocsort_bytetrack_interp_20260113_135953.txt`

**Konfiguracija**
*   **Model:** yolo11x.pt 
*   **Sustav za praćenje:** OC-SORT + ByteTrack + Interpolacija
*   **Hiperparametri:**
    *   Prag pouzdanosti (Visoki): 0.4
    *   Naknadna obrada: Linearna interpolacija (razmak=20)

**Rezultati**

| Metrika | Ukupno | MOT17-02 | MOT17-13 |
| :--- | :--- | :--- | :--- |
| **IDF1** | **48.5%** | 42.7% | 57.5% |
| **MOTA** | **38.9%** | 30.6% | 52.3% |
| **ID Sw.** | 450 | 80 | 70 |

## Eksperiment 11: OC-SORT + Byte + Interp + CMC (YOLOv11x)
**Zapis:** `experiment_ocsort_bytetrack_interp_cmc_20260113_152122.txt`

**Konfiguracija**
*   **Model:** yolo11x.pt 
*   **Sustav za praćenje:** OC-SORT + ByteTrack + Interpolacija + CMC
*   **Hiperparametri:**
    *   CMC: Optički tok (Afini)
    *   Prag pouzdanosti (Visoki): 0.5
    *   Veličina inferencije: 1280

**Rezultati**

| Metrika | Ukupno | MOT17-02 | MOT17-13 |
| :--- | :--- | :--- | :--- |
| **IDF1** | **51.7%** | 43.6% | 64.1% |
| **MOTA** | **40.4%** | 31.2% | 55.0% |
| **ID Sw.** | 276 | 65 | 27 |

## Eksperiment 12: OC-SORT + Byte + Interp + CMC (Fino podešen)
**Zapis:** `experiment_trained_best_20260113_170416.txt`

**Konfiguracija**
*   **Model:** yolo11l.pt (Fino podešen)
*   **Sustav za praćenje:** OC-SORT + ByteTrack + Interpolacija + CMC
*   **Hiperparametri:**
    *   CMC: Optički tok (Afini)
    *   Prag pouzdanosti: 0.5
    *   Maksimalna starost: 60

**Rezultati**

| Metrika | Ukupno | MOT17-02 | MOT17-13 |
| :--- | :--- | :--- | :--- |
| **IDF1** | **36.8%** | 35.2% | 39.2% |
| **MOTA** | **17.3%** | 13.9% | 22.8% |
| **ID Sw.** | 177 | 31 | 28 |

## Eksperiment 13: DeepSORT
**Zapis:** `experiment_deepsort_20260118_125316.txt`

**Konfiguracija**
*   **Model:** yolo11x.pt
*   **Sustav za praćenje:** DeepSORT (deep-sort-realtime)
*   **Hiperparametri:**
    *   Prag pouzdanosti: 0.5
    *   Veličina inferencije: 1920
    *   Maksimalna starost: 60
    *   Naknadna obrada: Linearna interpolacija (razmak=20)

**Rezultati**

| Metrika | Ukupno | MOT17-02 | MOT17-13 |
| :--- | :--- | :--- | :--- |
| **IDF1** | **37.4%** | 36.9% | 38.1% |
| **MOTA** | **12.3%** | 21.4% | -2.2% |
| **ID Sw.** | 573 | 115 | 76 |

## Eksperiment 14: OC-SORT + Byte + Interp + CMC (1920px)
**Zapis:** `experiment_ocsort_bytetrack_interp_cmc_1920_20260118_120603.txt`

**Konfiguracija**
*   **Model:** yolo11x.pt
*   **Sustav za praćenje:** OC-SORT + ByteTrack + Interpolacija + CMC
*   **Hiperparametri:**
    *   Prag pouzdanosti (Visoki): 0.5
    *   Veličina inferencije: 1920
    *   CMC: Optički tok (Afini)

**Rezultati**

| Metrika | Ukupno | MOT17-02 | MOT17-13 |
| :--- | :--- | :--- | :--- |
| **IDF1** | **52.1%** | 44.8% | 62.7% |
| **MOTA** | **40.7%** | 32.6% | 53.5% |
| **ID Sw.** | 306 | 70 | 32 |

## Eksperiment 15: OC-SORT + Byte + GSI + CMC
**Zapis:** `experiment_ocsort_bytetrack_gsi_cmc_20260118_140050.txt`

**Konfiguracija**
*   **Model:** yolo11x.pt
*   **Sustav za praćenje:** OC-SORT + ByteTrack + GSI + CMC
*   **Hiperparametri:**
    *   Prag pouzdanosti (Visoki): 0.5
    *   Veličina inferencije: 1920
    *   Naknadna obrada: Gaussovo zaglađivanje (sigma=1.0)
    *   CMC: Optički tok (Afini)

**Rezultati**

| Metrika | Ukupno | MOT17-02 | MOT17-13 |
| :--- | :--- | :--- | :--- |
| **IDF1** | **52.1%** | 44.9% | 62.4% |
| **MOTA** | **40.4%** | 32.6% | 52.8% |
| **ID Sw.** | 315 | 70 | 35 |

## Eksperiment 16: OC-SORT + Byte + GSI + CMC (Optimiziran)
**Zapis:** `experiment_ocsort_bytetrack_gsi_cmc_opt_20260118_143636.txt`

**Konfiguracija**
*   **Model:** yolo11x.pt
*   **Sustav za praćenje:** OC-SORT + ByteTrack + GSI + CMC
*   **Hiperparametri:**
    *   Prag pouzdanosti (Visoki): 0.3
    *   Veličina inferencije: 1920
    *   Maksimalna starost: 90
    *   Naknadna obrada: Gaussovo zaglađivanje (sigma=0.6)

**Rezultati**

| Metrika | Ukupno | MOT17-02 | MOT17-13 |
| :--- | :--- | :--- | :--- |
| **IDF1** | **49.4%** | 40.8% | 61.5% |
| **MOTA** | **39.2%** | 30.6% | 52.9% |
| **ID Sw.** | 447 | 101 | 48 |

## Eksperiment 17: OC-SORT + Byte + GSI + CMC (Optimiziran 2)
**Zapis:** `experiment_ocsort_bytetrack_gsi_cmc_opt2_20260118_150721.txt`

**Konfiguracija**
*   **Model:** yolo11x.pt
*   **Sustav za praćenje:** OC-SORT + ByteTrack + GSI + CMC
*   **Hiperparametri:**
    *   Prag pouzdanosti (Visoki): 0.45
    *   Veličina inferencije: 1920
    *   Naknadna obrada: Gaussovo zaglađivanje (sigma=1.0)

**Rezultati**

| Metrika | Ukupno | MOT17-02 | MOT17-13 |
| :--- | :--- | :--- | :--- |
| **IDF1** | **52.9%** | 46.2% | 62.4% |
| **MOTA** | **40.3%** | 32.2% | 53.1% |
| **ID Sw.** | 342 | 75 | 39 |

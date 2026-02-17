# Popravljanje rezultata MOT trackera pomoću metaheurističkih algoritama

Ovaj dokument predstavlja rezultate optimizacije hiperparametara za sustav praćenja objekata temeljen na YOLO11x detektoru i OC-SORT pratitelju, s integriranim ByteTrack logikom, kompenzacijom kretanja kamere (CMC) i linearnom interpolacijom. Cilj je bio poboljšati performanse praćenja na MOT17 skupu podataka koristeći metaheurističke algoritme.

## Opis predstavljenih metrika

| Metrika | Opis |
| :--- | :--- |
| **IDF1** | ID F1 rezultat. Harmonična sredina ID preciznosti i ID odziva, mjeri koliko dosljedno pratitelj održava ispravan ID objekta. |
| **IDP** | ID Preciznost. Postotak detektiranih objekata kojima je dodijeljen ispravan ID. |
| **IDR** | ID Odziv. Postotak objekata iz stvarne istine (ground truth) koji su praćeni s ispravnim ID-om. |
| **Rcll** | Odziv (Recall). Postotak objekata iz stvarne istine koji su ispravno detektirani, bez obzira na ID. |
| **Prcn** | Preciznost (Precision). Postotak detektiranih objekata koji odgovaraju stvarnim objektima. |
| **GT** | Stvarna istina (Ground Truth). Ukupan broj jedinstvenih tragova objekata u videu. |
| **MT** | Uglavnom praćeni (Mostly Tracked). Tragovi koji pokrivaju >80% putanje stvarne istine. |
| **PT** | Djelomično praćeni (Partially Tracked). Tragovi koji pokrivaju 20-80% putanje stvarne istine. |
| **ML** | Uglavnom izgubljeni (Mostly Lost). Tragovi koji pokrivaju <20% putanje stvarne istine. |
| **FP** | Lažno pozitivni (False Positives). Broj lažnih detekcija (duhovi). |
| **FN** | Lažno negativni (False Negatives). Broj propuštenih detekcija stvarnih objekata. |
| **IDs** | ID zamjene (ID Switches). Broj puta kada pratitelj pogrešno promijeni ID objekta koji je već bio praćen. |
| **FM** | Fragmentacija (Fragmentation). Broj puta kada je trag prekinut i kasnije nastavljen. |
| **MOTA** | Točnost praćenja više objekata (Multiple Object Tracking Accuracy). Ukupna metrika koja kombinira FP, FN i ID zamjene. |
| **MOTP** | Preciznost praćenja više objekata (Multiple Object Tracking Precision). Prosječno preklapanje (IoU) između predviđenih okvira i okvira stvarne istine. |
| **IDt** | ID prijenos (ID transfer). Broj puta kada je ID traga prenesen na drugi objekt stvarne istine. |
| **IDa** | ID uspon (ID ascension). Broj puta kada je ID pratitelja dodijeljen objektu stvarne istine koji je prethodno praćen drugim ID-om. |
| **IDm** | ID migracija (ID migration). Broj puta kada ID pratitelja pređe na objekt stvarne istine dok je prethodni ID još aktivan. |

## 1. Osnovna konfiguracija pratitelja

### 1.1. Opis cjevovoda pratitelja

Korišteni sustav praćenja objekata temelji se na arhitekturi "tracking-by-detection", kombinirajući snažan detektor objekata s naprednim strategijama asocijacije i post-procesiranja. Ključne komponente cjevovoda su:

*   **Detektor objekata (YOLO11x):** Koristi se YOLO11x model, prethodno treniran na COCO skupu podataka, za detekciju pješaka u svakom kadru. Detekcije se filtriraju na temelju praga pouzdanosti.
*   **OC-SORT (Observation-Centric SORT):** Jezgra pratitelja je OC-SORT algoritam, koji koristi Kalmanov filtar za procjenu stanja objekta. OC-SORT smanjuje oslanjanje na predviđanje kretanja tijekom dvosmislenih asocijacija, što ga čini robusnijim u gužvama.
*   **ByteTrack logika asocijacije:** Implementirana je dvostupanjska strategija asocijacije. Detekcije visoke pouzdanosti (iznad `CONFIDENCE_THRESHOLD`) prvo se asociraju s postojećim tragovima. Preostale detekcije niže pouzdanosti (između `CONFIDENCE_LOW` i `CONFIDENCE_THRESHOLD`) zatim se pokušavaju asocirati s neuparenim tragovima, što pomaže u održavanju tragova tijekom djelomičnih okluzija.
*   **Kompenzacija kretanja kamere (CMC):** Modul za CMC koristi Lucas-Kanade optički tok za procjenu afine transformacije između uzastopnih kadrova. Ova transformacija se primjenjuje na vektore stanja Kalmanovog filtra, čime se kompenzira kretanje kamere i poboljšava točnost predviđanja kretanja objekata.
*   **Linearna interpolacija:** Kao korak post-procesiranja, linearna interpolacija se koristi za popunjavanje praznina u putanjama tragova (do `MAX_GAP` kadrova), što značajno poboljšava odziv sustava i smanjuje fragmentaciju.

### 1.2. Osnovni hiperparametri

Sljedeći hiperparametri korišteni su u osnovnoj konfiguraciji pratitelja:

| Hiperparametar | Vrijednost | Opis |
| :--- | :--- | :--- |
| **CONFIDENCE_THRESHOLD** | 0.5 | Visoki prag pouzdanosti za inicijalizaciju tragova i prvu fazu ByteTrack asocijacije. |
| **CONFIDENCE_LOW** | 0.05 | Niski prag pouzdanosti za drugu fazu ByteTrack asocijacije, omogućujući praćenje detekcija niže pouzdanosti. |
| **IOU_THRESHOLD** | 0.3 | Prag preklapanja (Intersection over Union) za podudaranje detekcija s tragovima. |
| **MAX_AGE** | 60 | Maksimalan broj kadrova koliko izgubljeni trag može ostati aktivan prije nego što se ukloni. |
| **MIN_HITS** | 3 | Minimalan broj uzastopnih detekcija potreban za potvrdu novog traga. |
| **INERTIA** | 0.2 | Težina inercije u matrici troškova, utječe na glatkoću putanje u odnosu na poziciju. |
| **DELTA_T** | 3 | Razlika u vremenskom koraku za izračun brzine u Kalmanovom filtru. |
| **MAX_GAP** | 20 | Maksimalni broj kadrova za popunjavanje praznina u tragovima linearnom interpolacijom. |

### 1.3. Rezultati osnovne konfiguracije

Sljedeća tablica prikazuje ukupne metrike performansi za osnovnu konfiguraciju na MOT17 skupu podataka:

| Metrika | Vrijednost |
| :--- | :--- |
| **IDF1** | 48.9% |
| **IDP** | 56.5% |
| **IDR** | 43.1% |
| **Rcll** | 55.6% |
| **Prcn** | 72.9% |
| **GT** | 516 |
| **MT** | 165 |
| **PT** | 231 |
| **ML** | 120 |
| **FP** | 18765 |
| **FN** | 40245 |
| **IDs** | 417 |
| **FM** | 765 |
| **MOTA** | 34.5% |
| **MOTP** | 0.219 |
| **IDt** | 321 |
| **IDa** | 198 |
| **IDm** | 126 |


Osnovna konfiguracija postiže MOTA od 34.5% i IDF1 od 48.9%. Preciznost (Prcn) je relativno visoka (72.9%), što ukazuje na to da većina detekcija koje se prate odgovara stvarnim objektima. Međutim, odziv (Rcll) od 55.6% sugerira da se značajan broj stvarnih objekata propušta. Broj lažno pozitivnih (FP) detekcija je visok (18765), a broj zamjena ID-ova (IDs) od 417 ukazuje na prostor za poboljšanje u održavanju identiteta objekata, posebno u scenarijima okluzije.

## 2. Optimizacija pomoću TPE (Tree-structured Parzen Estimator)

U ovom dijelu, hiperparametri pratitelja optimizirani su pomoću algoritma TPE (Tree-structured Parzen Estimator) iz Optuna biblioteke. Optimizacija je provedena kroz 300 iteracija, s ciljem maksimiziranja kombiniranog rezultata MOTA i IDF1.

### 2.1. Optimizirani hiperparametri (TPE)

Sljedeća tablica prikazuje hiperparametre dobivene optimizacijom pomoću TPE algoritma:

| Hiperparametar | Osnovna vrijednost | TPE optimizirana vrijednost |
| :--- | :--- | :--- |
| CONFIDENCE_THRESHOLD | 0.5 | 0.517 |
| CONFIDENCE_LOW | 0.05 | 0.198 |
| IOU_THRESHOLD | 0.3 | 0.231 |
| MAX_AGE | 60 | 38 |
| MIN_HITS | 3 | 4 |
| INERTIA | 0.2 | 0.157 |
| DELTA_T | 3 | 3 |
| MAX_GAP | 20 | 31 |

### 2.2. Usporedba performansi (Osnovna vs. TPE)

Sljedeća tablica uspoređuje ukupne metrike performansi osnovne konfiguracije s onima optimiziranima pomoću TPE algoritma:

| Metrika | Osnovna | TPE optimizirana | $\Delta$ |
| :--- | :--- | :--- | :--- |
| **IDF1** | 48.9% | 53.8% | +4.9% |
| **IDP** | 56.5% | 67.0% | +10.5% |
| **IDR** | 43.1% | 45.0% | +1.9% |
| **Rcll** | 55.6% | 54.1% | -1.5% |
| **Prcn** | 72.9% | 80.6% | +7.7% |
| **FP** | 18765 | 11814 | -6951 |
| **FN** | 40245 | 41643 | +1398 |
| **IDs** | 417 | 273 | -144 |
| **FM** | 765 | 588 | -177 |
| **MOTA** | 34.5% | 40.7% | +6.2% |
| **MOTP** | 0.219 | 0.220 | +0.001 |


Optimizacija pomoću TPE algoritma rezultirala je značajnim poboljšanjem ukupnih performansi. MOTA je porasla za 6.2% (s 34.5% na 40.7%), a IDF1 za 4.9% (s 48.9% na 53.8%). Ovo poboljšanje uglavnom je posljedica smanjenja lažno pozitivnih detekcija (FP) za gotovo 7000 i smanjenja zamjena ID-ova (IDs) za 144. Preciznost (Prcn) je također značajno porasla (s 72.9% na 80.6%).

Analizirajući promjene hiperparametara, primjećujemo:
*   **CONFIDENCE_LOW** je značajno povećan (s 0.05 na 0.198), što sugerira da su detekcije s vrlo niskom pouzdanošću (ispod 0.198) bile previše bučne i doprinosile lažno pozitivnim rezultatima. Povećanje ovog praga pomoglo je u filtriranju buke, što je dovelo do manje FP-ova i veće preciznosti.
*   **IOU_THRESHOLD** je smanjen (s 0.3 na 0.231), što omogućuje labavije podudaranje detekcija s tragovima. To može pomoći u održavanju tragova u scenarijima gdje se bounding boxovi objekata malo mijenjaju.
*   **MAX_AGE** je smanjen (sa 60 na 38), što znači da se izgubljeni tragovi brže uklanjaju. To je vjerojatno pridonijelo smanjenju FP-ova, jer se tragovi ne održavaju predugo bez detekcija.
*   **MIN_HITS** je blago povećan (s 3 na 4), što zahtijeva više uzastopnih detekcija za inicijalizaciju traga, dodatno smanjujući inicijalizaciju lažnih tragova.
*   **INERTIA** je smanjena (s 0.2 na 0.157), što daje manju težinu inerciji u matrici troškova, čineći pratitelja osjetljivijim na promjene pozicije.
*   **DELTA_T** je ostao isti (3).
*   **MAX_GAP** je povećan (s 20 na 31), što omogućuje popunjavanje dužih praznina u tragovima, što je moglo pomoći u održavanju kontinuiteta tragova unatoč blagom padu odziva.

Unatoč blagom porastu lažno negativnih (FN) detekcija i blagom padu odziva (Rcll), ukupni dobitak u preciznosti i smanjenju zamjena ID-ova čini TPE optimiziranu konfiguraciju znatno boljom od osnovne.

## 3. Optimizacija pomoću Simulated Annealing (Simulirano kaljenje)

U ovom dijelu, hiperparametri pratitelja optimizirani su pomoću algoritma simuliranog kaljenja (Simulated Annealing). Optimizacija je provedena kroz 500 iteracija, također s ciljem maksimiziranja kombiniranog rezultata MOTA i IDF1.

### 3.1. Optimizirani hiperparametri (Simulirano kaljenje)

Sljedeća tablica prikazuje hiperparametre dobivene optimizacijom pomoću algoritma simuliranog žarenja:

| Hiperparametar | Osnovna vrijednost | SA optimizirana vrijednost |
| :--- | :--- | :--- |
| CONFIDENCE_THRESHOLD | 0.5 | 0.428 |
| CONFIDENCE_LOW | 0.05 | 0.098 |
| IOU_THRESHOLD | 0.3 | 0.131 |
| MAX_AGE | 60 | 117 |
| MIN_HITS | 3 | 5 |
| INERTIA | 0.2 | 0.313 |
| DELTA_T | 3 | 4 |
| MAX_GAP | 20 | 36 |

### 3.2. Usporedba performansi (Osnovna vs. SA)

Sljedeća tablica uspoređuje ukupne metrike performansi osnovne konfiguracije s onima optimiziranima pomoću algoritma simuliranog žarenja:

| Metrika | Osnovna | SA optimizirana | $\Delta$ |
| :--- | :--- | :--- | :--- |
| **IDF1** | 48.9% | 49.6% | +0.7% |
| **IDP** | 56.5% | 55.7% | -0.8% |
| **IDR** | 43.1% | 44.8% | +1.7% |
| **Rcll** | 55.6% | 57.7% | +2.1% |
| **Prcn** | 72.9% | 71.7% | -1.2% |
| **FP** | 18765 | 20604 | +1839 |
| **FN** | 40245 | 38367 | -1878 |
| **IDs** | 417 | 438 | +21 |
| **FM** | 765 | 840 | +75 |
| **MOTA** | 34.5% | 34.5% | 0.0% |
| **MOTP** | 0.219 | 0.225 | +0.006 |


Optimizacija pomoću simuliranog kaljenja donijela je marginalna poboljšanja u odnosu na osnovnu konfiguraciju. MOTA je ostala ista (34.5%), dok je IDF1 blago porasla za 0.7% (s 48.9% na 49.6%). Zabilježen je blagi porast odziva (Rcll) za 2.1%, ali uz cijenu smanjenja preciznosti (Prcn) za 1.2% i povećanja lažno pozitivnih detekcija (FP) za gotovo 1800. Broj zamjena ID-ova (IDs) također je blago porastao.

Analizirajući promjene hiperparametara:
*   **CONFIDENCE_THRESHOLD** je smanjen (s 0.5 na 0.428), što omogućuje uključivanje više detekcija u prvu fazu asocijacije.
*   **CONFIDENCE_LOW** je povećan (s 0.05 na 0.098), slično kao kod TPE, ali u manjoj mjeri.
*   **IOU_THRESHOLD** je značajno smanjen (s 0.3 na 0.131), što ukazuje na vrlo labav kriterij podudaranja. To je vjerojatno pridonijelo povećanju FP-ova i FM-ova.
*   **MAX_AGE** je značajno povećan (sa 60 na 117), što omogućuje tragovima da ostanu aktivni dulje vrijeme bez detekcija. To je moglo pridonijeti smanjenju FN-ova, ali i povećanju FP-ova i ID-ova.
*   **MIN_HITS** je povećan (s 3 na 5), što zahtijeva strože uvjete za inicijalizaciju traga.
*   **INERTIA** je povećana (s 0.2 na 0.313), što daje veću težinu inerciji, čineći putanje glatkijima.
*   **DELTA_T** je blago povećan (s 3 na 4).
*   **MAX_GAP** je povećan (s 20 na 36), što omogućuje popunjavanje dužih praznina.

U usporedbi s TPE optimizacijom, simulirano kaljenje nije uspjelo postići značajna poboljšanja. Iako je smanjilo lažno negativne detekcije i povećalo odziv, to je došlo uz cijenu povećanja lažno pozitivnih detekcija i zamjena ID-ova, što je rezultiralo stagnacijom MOTA metrike. To sugerira da je TPE algoritam bio učinkovitiji u pronalaženju boljeg balansa hiperparametara za ovaj problem.

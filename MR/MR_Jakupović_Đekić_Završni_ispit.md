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

## 1. Teoretski temelji i mehanizmi metaheurističke optimizacije hiperparametara

Suvremeni sustavi za praćenje više objekata (Multiple Object Tracking – MOT) predstavljaju integracije različitih algoritama računalnog vida, gdje se precizna detekcija objekata spaja s kompleksnim metodama vremenske i prostorne asocijacije. Arhitektura sustava koja koristi YOLO11x kao primarni detektor, nadopunjena algoritmima OCSORT i ByteTrack za asocijaciju, CMC (Camera Motion Compensation) za stabilizaciju pokreta kamere te linearnu interpolaciju za popunjavanje praznina u putanjama, generira iznimno složen prostor hiperparametara. Svaka od ovih komponenti posjeduje specifične parametre čije međusobno djelovanje izravno utječe na ključne metrike performansi, kao što su točnost praćenja (MOTA), dosljednost identiteta (IDF1) i broj promjena identiteta (ID Switches). Pronalaženje optimalne konfiguracije u ovakvom visokodimenzionalnom i nelinearnom prostoru predstavlja značajan izazov koji nadilazi mogućnosti ručnog podešavanja ili jednostavnih pretraga.4 U tom kontekstu, primjena metaheurističkih algoritama, poput stablom strukturiranih Parzenovih procjenitelja (Tree-structured Parzen Estimator – TPE) i simuliranog kaljenja (Simulated Annealing – SA), postaje nužna za sustavno istraživanje prostora rješenja.6

### 1.1. Izazovi optimizacije u domenama praćenja objekata

Prije detaljne analize samih optimizacijskih algoritama, nužno je razumjeti prirodu prostora koji se optimizira. Sustavi bazirani na paradigmi praćenja detekcijom (tracking-by-detection) ovise o kvaliteti ulaznih detekcija, ali i o robustnosti mehanizama koji te detekcije povezuju u koherentne putanje kroz vrijeme. YOLO11x, kao najsuvremenija iteracija detektora, nudi visoku preciznost, ali pragovi pouzdanosti (confidence thresholds) moraju biti usklađeni s mehanizmima ByteTrack algoritma koji koristi i detekcije niske pouzdanosti za rješavanje problema okluzije. Istovremeno, OCSORT uvodi parametre vezane uz Kalmanove filtre i inerciju kretanja, dok CMC dodaje parametre za procjenu afinih transformacija između kadrova.

Interakcija ovih parametara stvara "nazubljen" (rugged) krajolik funkcije cilja, gdje mala promjena u jednom parametru može uzrokovati drastične promjene u ukupnim performansama zbog kaskadnih efekata kroz cjevovod obrade.3 Tradicionalne metode poput pretraživanja po mreži (grid search) pate od "prokletstva dimenzionalnosti", postajući eksponencijalno skuplje sa svakim novim parametrom, dok nasumično pretraživanje (random search), iako efikasnije u nekim slučajevima, ne koristi informacije iz prethodnih pokušaja za usmjeravanje pretrage.11 Metaheuristički pristupi rješavaju ove probleme uvođenjem inteligencije u proces odabira sljedeće konfiguracije.6

| Metoda optimizacije | Pristup pretraživanju | Upravljanje poviješću | Glavna prednost u MOT kontekstu |
| :---- | :---- | :---- | :---- |
| **Grid Search** | Deterministički, iscrpan | Nema povijesti | Jednostavnost, ali neekonomičnost 12 |
| **Random Search** | Stohastički, neovisan | Nema povijesti | Bolje pokrivanje prostora od mreže 11 |
| **TPE** | Bayesov, modelom vođen | Aktivno uči iz povijesti | Izvrsno rukuje uvjetnim parametrima 6 |
| **Simulated Annealing** | Metaheuristički, probabilistički | Lokalni trendovi | Sposobnost bijega iz lokalnih optimuma 17 |

### 1.2. Arhitektura i mehanika stablom strukturiranih Parzenovih procjenitelja (TPE)

Algoritam stablom strukturiranih Parzenovih procjenitelja (TPE) predstavlja napredni oblik Bayesove optimizacije koji se temelji na sekvencijalnom modeliranju performansi (Sequential Model-Based Optimization – SMBO).6 Za razliku od standardnih Bayesovih metoda koje koriste Gaussove procese za izravno modeliranje funkcije cilja ![][image1] (gdje je ![][image2] metrika, a ![][image3] konfiguracija parametara), TPE invertira ovaj pristup modeliranjem gustoće vjerojatnosti konfiguracija s obzirom na njihovu uspješnost ![][image4].6 Ovaj pristup omogućuje algoritmu da efikasno rukuje diskretnim, kontinuiranim i, što je najvažnije, uvjetnim prostorima pretraživanja koji su inherentni sustavima s više alternativnih komponenti poput OCSORT-a i ByteTrack-a.2

#### 1.2.1. Razdvajanje prostora na bazi uspješnosti

Temeljna ideja TPE-a je podjela svih dosadašnjih evaluacija u dvije distinktne skupine na temelju odabranog kvantila performansi, koji se obično označava simbolom ![][image5].6 Na primjer, ako je ![][image5] postavljen na 0.15, algoritam će razvrstati najboljih 15% konfiguracija u prvu skupinu, dok će preostalih 85% činiti drugu skupinu.6 Za svaku od ovih skupina, TPE konstruira zasebnu funkciju gustoće vjerojatnosti koristeći metodu procjene gustoće jezgre (Kernel Density Estimation – KDE).11

Funkcija ![][image6] predstavlja distribuciju "dobrih" konfiguracija (one koje su rezultirale visokim performansama), dok funkcija ![][image7] predstavlja distribuciju preostalih konfiguracija.6 Matematički gledano, TPE traži onu konfiguraciju ![][image3] koja maksimizira očekivano poboljšanje (Expected Improvement – EI), što se u ovom modelu svodi na maksimiziranje omjera *l(x)/g(x)* 6. Apstraktno, to znači da algoritam traži parametre koji su vrlo vjerojatni u skupini uspješnih rezultata, a vrlo nevjerojatni u skupini neuspješnih.11

#### 1.2.2. Multinomialni i multivarijatni aspekti TPE-a

Multinomialni TPE-u ima sposobnost rukovati sa kategorijskim i diskretnim varijablama unutar jedinstvenog vjerojatnosnog okvira.10 U sustavu za praćenje, mnogi izbori su diskretni – primjerice, odabir tipa CMC algoritma ili uključivanje/isključivanje linearne interpolacije.1 Multinomialni TPE koristi prilagođene jezgre za ove varijable, osiguravajući da se diskretni izbori ne tretiraju kao kontinuirane vrijednosti, već kao nezavisne kategorije s vlastitim vjerojatnostima uspjeha.6

Nadalje, uvođenje multivarijatnog modeliranja omogućuje algoritmu da prepozna i iskoristi korelacije između parametara.16 U MOT sustavima, parametri su rijetko neovisni; na primjer, optimalni prag detekcije za YOLO11x može ovisiti o postavkama Kalmanovog filtra u OCSORT-u.1 Multivarijatni TPE modelira ove parametre zajednički, što rezultira bržom konvergencijom prema globalnom optimumu u usporedbi s nezavisnim TPE-om koji svaki parametar promatra izolirano.10

#### 1.2.3. Dinamika rada TPE algoritma

Proces započinje fazom nasumičnog uzorkovanja (startup trials), tijekom koje algoritam prikuplja inicijalne podatke o prostoru pretraživanja.2 Bez ovih početnih točaka, nemoguće je konstruirati smislene distribucije ![][image6] i ![][image7].11 Nakon što se prikupi dovoljan broj uzoraka, TPE ulazi u iterativni ciklus:

1. Razvrstavanje povijesnih podataka u "dobru" i "lošu" skupinu na temelju zadanog praga.6  
2. Prilagodba Parzenovih procjenitelja (KDE) za svaku skupinu.11  
3. Generiranje velikog broja kandidata iz distribucije ![][image6].6  
4. Bodovanje kandidata pomoću omjera *l(x)/g(x)* i odabir onog s najvišim rezultatom.6  
5. Evaluacija odabrane konfiguracije na stvarnom sustavu za praćenje i ažuriranje povijesti.11

Ovaj mehanizam osigurava da algoritam neprestano balansira između istraživanja nepoznatih dijelova prostora i fokusiranja na regije koje su se već pokazale obećavajućima.10

### 1.3. Simulirano kaljenje (Simulated Annealing – SA): Termodinamička metaheuristika

Simulirano kaljenje je stohastička tehnika optimizacije koja vuče izravnu paralelu s procesima u fizici čvrstog stanja, točnije s procesom kaljenja metala.17 U metalurgiji, kaljenje uključuje zagrijavanje materijala do visokih temperatura, čime se atomi oslobađaju iz svojih fiksnih pozicija, a zatim polagano hlađenje kako bi se atomi presložili u stabilnu strukturu s minimalnom energijom.7 U kontekstu optimizacije hiperparametara, energija sustava predstavlja vrijednost ciljne funkcije (npr. inverzna vrijednost MOTA metrike), dok temperatura služi kao kontrolni parametar koji određuje razinu nasumičnosti u pretraživanju.17

#### 1.3.1. Princip "bijega" iz lokalnih optimuma

Ključna prednost simuliranog kaljenja u odnosu na jednostavnije algoritme lokalnog pretraživanja je njegova sposobnost da prihvati lošija rješenja.17 Algoritmi koji uvijek prihvaćaju samo poboljšanja neizbježno će se zaglaviti u prvom lokalnom optimumu na koji naiđu.7 SA rješava ovaj problem uvođenjem probabilističkog kriterija prihvaćanja, poznatog kao Metropolis-Hastings kriterij.3

Kada algoritam generira novu konfiguraciju hiperparametara (susjedno stanje), on uspoređuje njezinu izvedbu s trenutnom.12 Ako je nova konfiguracija bolja, ona se prihvaća odmah.17 Ako je lošija, ona se i dalje može prihvatiti s određenom vjerojatnošću koja ovisi o dva faktora: veličini pogoršanja i trenutnoj temperaturi sustava.3 Matematički se ova vjerojatnost izražava kao ![][image9], gdje je ![][image10] razlika u "energiji", a ![][image11] temperatura.3 Pri visokim temperaturama, vjerojatnost prihvaćanja lošijih rješenja je velika, što omogućuje algoritmu da slobodno istražuje cijeli prostor, dok pri niskim temperaturama sustav postaje "konzervativan" i teži samo poboljšanjima.14

#### 1.3.2. Komponente i parametri procesa kaljenja

Uspjeh simuliranog kaljenja u optimizaciji YOLO11x trackera ovisi o preciznom definiranju nekoliko ključnih komponenti 12:

1. **Početna temperatura (![][image12]):** Mora biti dovoljno visoka da omogući inicijalno istraživanje širokog prostora parametara.12  
2. **Raspored hlađenja (Cooling Schedule):** Određuje brzinu smanjenja temperature kroz iteracije.12 Najčešće se koristi eksponencijalno hlađenje (![][image13]), gdje je ![][image14] faktor hlađenja (obično između 0.8 i 0.99).12  
3. **Funkcija perturbacije (Neighborhood Function):** Definira kako se generira nova konfiguracija iz postojeće.12 U kontekstu hiperparametara, to obično uključuje malu, nasumičnu promjenu jedne ili više vrijednosti (npr. promjena IoU praga za ![][image15]).3  
4. **Kriterij zaustavljanja:** Algoritam završava kada temperatura padne ispod određenog praga ili kada se nakon određenog broja iteracija ne zabilježi značajno poboljšanje.12

| Parametar SA | Funkcija u procesu | Utjecaj na optimizaciju |
| :---- | :---- | :---- |
| **Početna temperatura** | Postavlja razinu inicijalnog kaosa | Visoka temperatura sprječava rano zaglavljivanje 12 |
| **Faktor hlađenja (![][image14])** | Kontrolira brzinu stabilizacije | Sporije hlađenje povećava šansu za globalni optimum 17 |
| **Energija (![][image16])** | Predstavlja metriku performansi | Motivira kretanje prema boljim konfiguracijama 17 |
| **Iteracije po temp.** | Osigurava termičku ravnotežu | Više iteracija omogućuje dublje istraživanje regije 12 |

### 1.4. Definiranje ciljeva optimizacije

Primijenjeni metaheuristički algoritmi korišteni su s  ciljem simultanog poboljšanja više aspekata kvalitete praćenja. Konkretno, proces optimizacije bio je usmjeren na:

*   **Maksimizaciju MOTA (Multiple Object Tracking Accuracy):** Kako bi se osigurala visoka ukupna točnost detekcije i praćenja.
*   **Maksimizaciju IDF1 (ID F1 Score):** Kako bi se povećala dosljednost očuvanja identiteta objekata kroz vrijeme.
*   **Minimizaciju IDsw (ID Switches):** Kako bi se reducirao broj pogrešnih promjena identiteta, što je ključno za stabilnost putanja.




## 2. Osnovna konfiguracija pratitelja

### 2.1. Opis cjevovoda pratitelja

Korišteni sustav praćenja objekata temelji se na arhitekturi "tracking-by-detection", kombinirajući snažan detektor objekata s naprednim strategijama asocijacije i post-procesiranja. Ključne komponente cjevovoda su:

*   **Detektor objekata (YOLO11x):** Koristi se YOLO11x model, prethodno treniran na COCO skupu podataka, za detekciju pješaka u svakom kadru. Detekcije se filtriraju na temelju praga pouzdanosti.
*   **OC-SORT (Observation-Centric SORT):** Jezgra pratitelja je OC-SORT algoritam, koji koristi Kalmanov filtar za procjenu stanja objekta. OC-SORT smanjuje oslanjanje na predviđanje kretanja tijekom dvosmislenih asocijacija, što ga čini robusnijim u gužvama.
*   **ByteTrack logika asocijacije:** Implementirana je dvostupanjska strategija asocijacije. Detekcije visoke pouzdanosti (iznad `CONFIDENCE_THRESHOLD`) prvo se asociraju s postojećim tragovima. Preostale detekcije niže pouzdanosti (između `CONFIDENCE_LOW` i `CONFIDENCE_THRESHOLD`) zatim se pokušavaju asocirati s neuparenim tragovima, što pomaže u održavanju tragova tijekom djelomičnih okluzija.
*   **Kompenzacija kretanja kamere (CMC):** Modul za CMC koristi Lucas-Kanade optički tok za procjenu afine transformacije između uzastopnih kadrova. Ova transformacija se primjenjuje na vektore stanja Kalmanovog filtra, čime se kompenzira kretanje kamere i poboljšava točnost predviđanja kretanja objekata.
*   **Linearna interpolacija:** Kao korak post-procesiranja, linearna interpolacija se koristi za popunjavanje praznina u putanjama tragova (do `MAX_GAP` kadrova), što značajno poboljšava odziv sustava i smanjuje fragmentaciju.

### 2.2. Osnovni hiperparametri

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

### 2.3. Rezultati osnovne konfiguracije

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

## 3. Optimizacija pomoću TPE (Tree-structured Parzen Estimator)

U ovom dijelu, hiperparametri pratitelja optimizirani su pomoću algoritma TPE (Tree-structured Parzen Estimator) iz Optuna biblioteke. Optimizacija je provedena kroz 300 iteracija, s ciljem maksimiziranja kombiniranog rezultata MOTA i IDF1.

### 3.1. Optimizirani hiperparametri (TPE)

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

### 3.2. Usporedba performansi (Osnovna vs. TPE)

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

Analizirajući promjene hiperparametara, može se primijetiti:
*   **CONFIDENCE_LOW** je značajno povećan (s 0.05 na 0.198), što sugerira da su detekcije s vrlo niskom pouzdanošću (ispod 0.198) bile previše bučne i doprinosile lažno pozitivnim rezultatima. Povećanje ovog praga pomoglo je u filtriranju buke, što je dovelo do manje FP-ova i veće preciznosti.
*   **IOU_THRESHOLD** je smanjen (s 0.3 na 0.231), što omogućuje labavije podudaranje detekcija s tragovima. To može pomoći u održavanju tragova u scenarijima gdje se bounding boxovi objekata malo mijenjaju.
*   **MAX_AGE** je smanjen (sa 60 na 38), što znači da se izgubljeni tragovi brže uklanjaju. To je vjerojatno pridonijelo smanjenju FP-ova, jer se tragovi ne održavaju predugo bez detekcija.
*   **MIN_HITS** je blago povećan (s 3 na 4), što zahtijeva više uzastopnih detekcija za inicijalizaciju traga, dodatno smanjujući inicijalizaciju lažnih tragova.
*   **INERTIA** je smanjena (s 0.2 na 0.157), što daje manju težinu inerciji u matrici troškova, čineći pratitelja osjetljivijim na promjene pozicije.
*   **DELTA_T** je ostao isti (3).
*   **MAX_GAP** je povećan (s 20 na 31), što omogućuje popunjavanje dužih praznina u tragovima, što je moglo pomoći u održavanju kontinuiteta tragova unatoč blagom padu odziva.

Unatoč blagom porastu lažno negativnih (FN) detekcija i blagom padu odziva (Rcll), ukupni dobitak u preciznosti i smanjenju zamjena ID-ova čini TPE optimiziranu konfiguraciju znatno boljom od osnovne.

## 4. Optimizacija pomoću Simulated Annealing (Simulirano žarenje)

U ovom dijelu, hiperparametri pratitelja optimizirani su pomoću algoritma simuliranog kaljenja (Simulated Annealing). Optimizacija je provedena kroz 500 iteracija, također s ciljem maksimiziranja kombiniranog rezultata MOTA i IDF1.

### 4.1. Optimizirani hiperparametri (Simulirano žarenje)

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

### 4.2. Usporedba performansi (Osnovna vs. SA)

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

## 5. Optimizacija pomoću NSGA-II (Non-dominated Sorting Genetic Algorithm II)

U ovom dijelu, hiperparametri pratitelja optimizirani su pomoću algoritma NSGA-II, elitističkog genetskog algoritma za višeciljnu optimizaciju. Za razliku od metoda koje optimiziraju jednu skalariziranu funkciju cilja, NSGA-II prirodno radi s više ciljeva istovremeno, pri čemu nastoji pronaći skup rješenja koja čine aproksimaciju Pareto-fronte. U kontekstu MOT sustava, to je korisno jer su poželjne metrike često u napetom kompromisu: primjerice, agresivno filtriranje detekcija može smanjiti FP i povećati preciznost, ali istovremeno povećati FN i sniziti recall, dok pokušaji stabilizacije identiteta (manje ID switch-eva) ponekad dolaze uz cijenu fragmentacija ili smanjenja ukupnog odziva.

NSGA-II kombinira tri ključna mehanizma: (1) brzo nedominirano sortiranje populacije po Pareto-rangovima, (2) elitizam kroz spajanje roditeljske i potomne populacije, te (3) održavanje raznolikosti rješenja pomoću crowding distance mjere. Time algoritam izbjegava preranu konvergenciju na jedno “lokalno dobro” rješenje i umjesto toga održava različite oblike kompromisa među ciljevima, što je posebno relevantno u “nazubljenom” prostoru hiperparametara kakav nastaje kombinacijom YOLO detekcije, OC-SORT dinamike, ByteTrack asocijacije, CMC korekcije i interpolacije.

### 5.1. Optimizirani hiperparametri (NSGA-II)

Budući da NSGA-II generira skup Pareto-optimalnih rješenja, u izvještaju je odabrana konfiguracija koja predstavlja dobar kompromis između MOTA i IDF1 metrika. Kao reprezentativno rješenje odabran je **Trial 58**, koji postiže visoku vrijednost obje metrike uz umjeren broj ID zamjena.

| Hiperparametar       | Osnovna vrijednost | NSGA-II optimizirana vrijednost (Trial 58) |
| :------------------- | :----------------- | :----------------------------------------- |
| CONFIDENCE_THRESHOLD | 0.5                | **0.4895**                                 |
| CONFIDENCE_LOW       | 0.05               | **0.1919**                                 |
| IOU_THRESHOLD        | 0.3                | **0.2793**                                 |
| MAX_AGE              | 60                 | **40**                                     |
| MIN_HITS             | 3                  | **2**                                      |
| INERTIA              | 0.2                | **0.4469**                                 |
| DELTA_T              | 3                  | **2**                                      |
| MAX_GAP              | 20                 | **16**                                     |

Ova konfiguracija pokazuje jasnu tendenciju prema:

- umjerenom smanjenju glavnog praga pouzdanosti (`CONFIDENCE_THRESHOLD`)
- značajnom povećanju donjeg praga (`CONFIDENCE_LOW`)
- nešto nižem IoU pragu
- kraćem zadržavanju neaktivnih tragova (`MAX_AGE`)
- manjem broju potrebnih inicijalnih pogodaka (`MIN_HITS`)
- povećanoj inerciji modela

Takva kombinacija parametara sugerira stabilniju asocijaciju detekcija, uz smanjenu oscilaciju identiteta i brže potvrđivanje novih tragova.

### 5.2. Usporedba performansi (Osnovna vs. NSGA-II)

Rezultati za odabranu NSGA-II konfiguraciju (Trial 58) prikazani su u nastavku:

| Metrika  | Osnovna | NSGA-II optimizirana | Δ            |
| :------- | :------ | :------------------- | :----------- |
| **IDF1** | 48.9%   | **52.99%**           | **+4.09 pp** |
| **MOTA** | 34.5%   | **40.05%**           | **+5.55 pp** |
| **IDsw** | 417     | **318**              | **−99**      |

Osnovna konfiguracija ostvaruje MOTA od 34.5% i IDF1 od 48.9%. NSGA-II optimizacija povećava MOTA na 40.05%, dok IDF1 raste na 52.99%. Istovremeno, broj ID zamjena smanjen je sa 417 na 318, što predstavlja smanjenje od gotovo 24%.

Posebno je značajno da se poboljšanje MOTA i IDF1 događa simultano. U mnogim MOT sustavima povećanje jedne metrike često dolazi na štetu druge, no u ovom slučaju NSGA-II je pronašao regiju hiperparametarskog prostora u kojoj dolazi do općeg poboljšanja performansi.

### 5.3. Analiza Pareto fronte i ponašanja algoritma

Analizom svih evaluacija može se uočiti jasna konvergencija algoritma prema regiji optimalnih rješenja. Najviši postignuti MOTA iznosi 0.4016 (Trial 125), dok najviši IDF1 doseže 0.5373 (Trial 154). Ta rješenja predstavljaju različite točke na Pareto fronti, gdje jedno favorizira ukupnu točnost (MOTA), a drugo konzistentnost identiteta (IDF1).

U rasponu trialova nakon približno 120. evaluacije rezultati se stabiliziraju u intervalu:

- MOTA ≈ 0.38 – 0.40
- IDF1 ≈ 0.50 – 0.53

što ukazuje na konvergenciju populacije prema optimalnoj regiji prostora.

Promatrajući distribuciju parametara u najboljim rješenjima, mogu se uočiti sljedeći obrasci:

- `CONFIDENCE_THRESHOLD` se stabilizira u intervalu 0.48 – 0.60
- `CONFIDENCE_LOW` raste prema 0.13 – 0.20
- `IOU_THRESHOLD` se najčešće nalazi u rasponu 0.27 – 0.38
- `DELTA_T` je gotovo uvijek 1 ili 2
- `MIN_HITS` se često smanjuje na 2

Ovi obrasci sugeriraju da stabilnost identiteta proizlazi iz balansiranja između strogoće prihvata detekcija i trajanja zadržavanja traga, dok niže vrijednosti `DELTA_T` doprinose bržoj reakciji modela na promjene.

Elitistička selekcija i crowding distance mehanizam NSGA-II algoritma omogućili su održavanje raznolikosti populacije tijekom optimizacije, čime je izbjegnuta prerana konvergencija na suboptimalno rješenje. Upravo taj mehanizam omogućio je pronalazak konfiguracija koje istovremeno poboljšavaju MOTA i IDF1, umjesto da optimizacija favorizira samo jednu metriku.

## 6. Optimizacija pomoću Bayesove optimizacije (BayesOpt)

U ovom dijelu, hiperparametri pratitelja optimizirani su pomoću Bayesove optimizacije. Bayesova optimizacija spada u skupinu metoda sekvencijalne optimizacije vođene modelom (SMBO), gdje se skupi i vremenski zahtjevni eksperimenti (u ovom slučaju kompletno pokretanje trackera nad sekvencama) zamjenjuju iterativnim procesom: nakon svake evaluacije trenira se aproksimacijski (surrogate) model funkcije cilja, a zatim se pomoću akvizicijske funkcije odlučuje koja konfiguracija parametara ima najveći potencijal za poboljšanje. Klasična formulacija često koristi Gaussove procese kao surrogate model, uz akvizicijske funkcije poput Expected Improvement ili srodnih kriterija, čime se postiže efikasno korištenje povijesnih evaluacija i “pametnije” pretraživanje prostora nego kod čistog random search-a.

U kontekstu MOT sustava, Bayesova optimizacija je posebno privlačna jer je svaka evaluacija skupa (obrada više video sekvenci), a prostor hiperparametara je kontinuiran i nelinearan. Surrogate model omogućava da se već nakon relativno malog broja iteracija počne favorizirati područja prostora koja pokazuju bolji kompromis između metrika, dok akvizicijska funkcija osigurava da se ipak povremeno istražuju i slabije poznate regije kako bi se izbjeglo prerano “zaključavanje” na lokalno dobro rješenje.

### 6.1. Optimizirani hiperparametri (BayesOpt)

Najbolje rješenje dobiveno Bayesovom optimizacijom ostvaruje sljedeće vrijednosti:

- **MOTA = 0.4005**
- **IDF1 = 0.5440**
- **IDsw = 315**
- **Score = 0.4723**

Optimizirani hiperparametri prikazani su u nastavku.

| Hiperparametar       | Osnovna vrijednost | BayesOpt optimizirana vrijednost |
| :------------------- | :----------------- | :------------------------------- |
| CONFIDENCE_THRESHOLD | 0.5                | **0.4995**                       |
| CONFIDENCE_LOW       | 0.05               | **0.1765**                       |
| IOU_THRESHOLD        | 0.3                | **0.2393**                       |
| MAX_AGE              | 60                 | **22**                           |
| MIN_HITS             | 3                  | **3**                            |
| INERTIA              | 0.2                | **0.1505**                       |
| DELTA_T              | 3                  | **4**                            |
| MAX_GAP              | 20                 | **39**                           |

U odnosu na osnovnu konfiguraciju, vidljivo je da BayesOpt:

- zadržava prag pouzdanosti vrlo blizu početne vrijednosti,
- značajno povećava donji prag (`CONFIDENCE_LOW`),
- smanjuje IoU prag,
- skraćuje trajanje zadržavanja neaktivnih tragova,
- zadržava stabilnu vrijednost `MIN_HITS`,
- smanjuje inerciju modela.

### 6.2. Usporedba performansi (Osnovna vs. BayesOpt)

| Metrika  | Osnovna | BayesOpt optimizirana | Δ            |
| :------- | :------ | :-------------------- | :----------- |
| **IDF1** | 48.9%   | **54.40%**            | **+5.50 pp** |
| **MOTA** | 34.5%   | **40.05%**            | **+5.55 pp** |
| **IDsw** | 417     | **315**               | **−102**     |

Osnovna konfiguracija postiže MOTA od 34.5% i IDF1 od 48.9%. Bayesova optimizacija povećava MOTA na 40.05% te IDF1 na 54.40%, dok se broj ID zamjena smanjuje za više od 100. Time je postignuto simultano poboljšanje obje ključne metrike uz značajno smanjenje pogrešnih zamjena identiteta.

### 6.3. Analiza konvergencije i interpretacija rezultata

Analizom tijeka optimizacije može se uočiti da se nakon približno 90–100 evaluacija score stabilizira u relativno uskom rasponu vrijednosti (približno 0.458 – 0.472). Takvo ponašanje karakteristično je za Bayesovu optimizaciju kada Gaussian Process model uspješno aproksimira lokalnu strukturu funkcije cilja i akvizicijska funkcija počne predlagati konfiguracije unutar ograničene, ali optimalne regije prostora.

Distribucija najboljih rješenja pokazuje jasnu konvergenciju prema:

- `CONFIDENCE_THRESHOLD` ≈ 0.45–0.55
- `CONFIDENCE_LOW` ≈ 0.17–0.20
- `IOU_THRESHOLD` ≈ 0.23–0.30
- `MAX_AGE` ≈ 15–25

Ovi obrasci sugeriraju da poboljšanje performansi proizlazi iz preciznijeg balansiranja između prihvaćanja novih detekcija i stabilnosti postojećih tragova. Smanjeni `MAX_AGE` ograničava trajanje potencijalno pogrešnih identiteta, dok povećani `CONFIDENCE_LOW` stabilizira sekundarne asocijacije i smanjuje fragmentacije. Umjereno sniženi IoU prag omogućava fleksibilnije povezivanje detekcija bez značajnog povećanja lažnih asocijacija.

U usporedbi s NSGA-II optimizacijom, BayesOpt pokazuje slične razine MOTA, ali nešto višu IDF1 metriku, što ukazuje na snažniji fokus prema konzistentnosti identiteta. S obzirom na to da Bayesova optimizacija koristi eksplicitni probabilistički model funkcije cilja, njezina sposobnost brzog pronalaska stabilne regije hiperparametara potvrđuje učinkovitost model-vođenog pristupa u optimizaciji složenih sustava računalnog vida.

## 7. Zaključak

U ovom radu prikazan je sustavan pristup poboljšanju performansi sustava za praćenje više objekata na MOT17 skupu podataka kroz optimizaciju hiperparametara cjevovoda temeljenog na YOLO11x detektoru i OC-SORT pratitelju, uz integriranu ByteTrack logiku, kompenzaciju kretanja kamere (CMC) i linearnu interpolaciju. Polazna konfiguracija pokazala je solidnu preciznost (Prcn 72.9%), ali ograničen odziv (Rcll 55.6%) te izražene probleme u stabilnosti identiteta (IDF1 48.9% uz 417 ID zamjena), što je potvrđeno i relativno niskom ukupnom točnošću praćenja (MOTA 34.5%). Takav početni profil tipičan je za tracking-by-detection sustave u kojima kompromis između “stroge” filtracije detekcija i očuvanja kontinuiteta tragova izravno determinira FP/FN balans, a posljedično i MOTA/IDF1 ponašanje.

Rezultati optimizacije pokazuju da je primjena metaheurističkih i modelom vođenih metoda opravdana i učinkovita, jer je omogućila pronalazak regija hiperparametarskog prostora koje ručno podešavanje teško može pouzdano doseći. Najznačajniji iskorak ostvaren je metodama koje koriste povijesne evaluacije za inteligentno usmjeravanje pretrage. TPE je povećao MOTA na 40.7% i IDF1 na 53.8%, uz snažno smanjenje FP (−6951) i ID zamjena (−144), što ukazuje da je optimizacija dominantno djelovala kroz čišćenje buke i stabilizaciju asocijacije. Slično tomu, NSGA-II je demonstrirao prednost višeciljnog pristupa: iako algoritam ne traži jedno jedino optimum rješenje, pronađena je točka kompromisa (Trial 58) koja simultano poboljšava i MOTA (40.05%) i IDF1 (52.99%), uz smanjenje ID zamjena na 318. Time je potvrđeno da je u ovom sustavu moguće pronaći konfiguracije koje ne “trguju” jednom metrikom za drugu, nego djeluju povoljno na obje, što je u MOT domeni relativno rijedak, ali vrlo vrijedan ishod.

Najbolji ukupni balans u ovom radu ostvarila je Bayesova optimizacija, koja postiže IDF1 = 54.40% i MOTA = 40.05% uz IDsw = 315. Ovakav rezultat sugerira da je probabilistički surrogate model uspješno uhvatio lokalnu strukturu funkcije cilja te nakon inicijalne faze istraživanja koncentrirao pretragu na stabilnu, optimalnu regiju prostora. Važno je uočiti zajedničke obrasce u najboljim konfiguracijama (TPE, NSGA-II i BayesOpt): (1) značajno povećanje donjeg praga pouzdanosti (CONFIDENCE_LOW) u pravilu smanjuje ulaznu detekcijsku “buku” u drugoj fazi ByteTrack asocijacije, čime se smanjuju FP i stabilizira identitet; (2) smanjenje IoU praga omogućuje fleksibilnije uparivanje detekcija i tragova u situacijama varijabilnih okvira, ali zahtijeva pažljiv balans kako ne bi generiralo lažne asocijacije; (3) kraći MAX_AGE često djeluje kao mehanizam kontrole “tvrdoglavih” tragova koji bez dovoljno dokaza ostaju aktivni i akumuliraju greške identiteta. U kombinaciji, ovi pomaci objašnjavaju zašto se dobici u MOTA i IDF1 pojavljuju paralelno: sustav se istovremeno čisti od lažnih tragova i smanjuje broj pogrešnih prijelaza identiteta.

Suprotno tome, simulirano kaljenje pokazalo se najmanje učinkovitim u ovom eksperimentalnom okruženju. Iako je povećalo odziv i smanjilo FN (što je intuitivno posljedica “liberalnijeg” zadržavanja tragova kroz visoki MAX_AGE i vrlo niski IoU prag), to je došlo uz porast FP, fragmentacija i ID zamjena, što je rezultiralo stagnacijom MOTA i tek marginalnim porastom IDF1. Ovaj ishod je konzistentan s karakterom SA algoritma: bez eksplicitnog modeliranja prostora cilja, stohastički hod može lako “privući” konfiguracije koje poboljšavaju jednu komponentu (npr. recall) na štetu globalnog balansa metrika koji je presudan za MOT evaluaciju.

Sveukupno, provedeni eksperimenti potvrđuju da je najveći potencijal poboljšanja u sustavu ležao u kontroliranju detekcijskog šuma i stabilizaciji asocijacije, a ne u agresivnom povećavanju odziva pod svaku cijenu. Metaheurističke metode i Bayesova optimizacija omogućile su da se taj balans postigne precizno: u odnosu na osnovnu konfiguraciju ostvaren je skok MOTA za približno +5.5 do +6.2 postotnih bodova te skok IDF1 za približno +4.1 do +5.5 postotnih bodova, uz smanjenje ID zamjena za oko 100–144, ovisno o metodi. Time je potvrđena hipoteza rada da optimizacija hiperparametara može “popraviti” rezultate trackera bez promjene osnovne arhitekture, isključivo kroz bolju sinergiju komponenti (YOLO11x, OC-SORT, ByteTrack, CMC, interpolacija).

Konačno, praktična vrijednost ovih nalaza je jasna: za sustave nadzora, analitike ili autonomne percepcije, gdje je važno istovremeno zadržati što više objekata u praćenju i očuvati konzistentnost identiteta, pristupi poput TPE, NSGA-II i BayesOpt predstavljaju metodološki utemeljen i inženjerski isplativ put prema boljim performansama. Posebno se Bayesova optimizacija pokazuje kao najkorisniji izbor kada je broj evaluacija ograničen, dok NSGA-II pruža dodatnu fleksibilnost u situacijama gdje se u proizvodnom sustavu ciljevi mogu mijenjati (npr. preferiranje identiteta naspram ukupne točnosti), jer prirodno isporučuje skup kompromisnih rješenja umjesto jedne jedine konfiguracije.

## Citirani radovi

2. Hyperparameter Optimization with Optuna \- Medium, pristupljeno veljače 17, 2026, https://medium.com/@mjgmario/hyperparameter-optimization-with-optuna-8fca06ea5491  
3. Simulated Annealing-Based Hyperparameter Optimization of a Convolutional Neural Network for MRI Brain Tumor Classification \- MDPI, pristupljeno veljače 17, 2026, https://www.mdpi.com/2504-4990/7/2/50  
4. Hyperparameter optimization: Foundations, algorithms, best practices, and open challenges, pristupljeno veljače 17, 2026, https://www.repo.uni-hannover.de/bitstream/123456789/15816/1/Hyperparameter-optimization-Foundations.pdf  
5. Hyperparameter optimization \- Wikipedia, pristupljeno veljače 17, 2026, https://en.wikipedia.org/wiki/Hyperparameter\_optimization  
6. Tree-Structured Parzen Estimators (TPE) \- Emergent Mind, pristupljeno veljače 17, 2026, https://www.emergentmind.com/topics/tree-structured-parzen-estimators-tpe  
7. Simulated Annealing As a Machine Learning Model: Principles, Applications, and Comparative Analysis, pristupljeno veljače 17, 2026, https://ijsret.com/wp-content/uploads/IJSRET\_V11\_issue3\_952.pdf  
8. Algorithms for Hyper-Parameter Optimization \- NIPS, pristupljeno veljače 17, 2026, http://papers.neurips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf  
9. optuna.samplers.TPESampler \- Read the Docs, pristupljeno veljače 17, 2026, https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html  
10. Tree-Structured Parzen Estimator: Understanding Its Algorithm Components and Their Roles for Better Empirical Performance \- arXiv.org, pristupljeno veljače 17, 2026, https://arxiv.org/html/2304.11127v4  
11. Building a Tree-Structured Parzen Estimator from Scratch (Kind Of) | Towards Data Science, pristupljeno veljače 17, 2026, https://towardsdatascience.com/building-a-tree-structured-parzen-estimator-from-scratch-kind-of-20ed31770478/  
12. Hyperparameter Tuning using Simulated Annealing \- GitHub Pages, pristupljeno veljače 17, 2026, https://santhoshhari.github.io/simulated\_annealing/  
13. (PDF) Algorithms for Hyper-Parameter Optimization \- ResearchGate, pristupljeno veljače 17, 2026, https://www.researchgate.net/publication/216816964\_Algorithms\_for\_Hyper-Parameter\_Optimization  
14. Simulated Annealing \- Algorithm Afternoon, pristupljeno veljače 17, 2026, https://algorithmafternoon.com/physical/simulated\_annealing/  
15. (PDF) Algorithms for hyper-parameter optimization \- ResearchGate, pristupljeno veljače 17, 2026, https://www.researchgate.net/publication/304781977\_Algorithms\_for\_hyper-parameter\_optimization  
16. optuna.samplers.TPESampler \- Read the Docs, pristupljeno veljače 17, 2026, https://optuna.readthedocs.io/en/v2.10.1/reference/generated/optuna.samplers.TPESampler.html  
17. Simulated annealing \- Cornell University Computational Optimization Open Textbook, pristupljeno veljače 17, 2026, https://optimization.cbe.cornell.edu/index.php?title=Simulated\_annealing  
18. An Introduction to a Powerful Optimization Technique: Simulated Annealing, pristupljeno veljače 17, 2026, https://towardsdatascience.com/an-introduction-to-a-powerful-optimization-technique-simulated-annealing-87fd1e3676dd/  
19. Tree Parzen Estimator (TPE) for Hyperparameter Optimization \- Emergent Mind, pristupljeno veljače 17, 2026, https://www.emergentmind.com/topics/tree-parzen-estimator-tpe  
20. "Multivariate" TPE Makes Optuna Even More Powerful \- Preferred Networks Tech Blog, pristupljeno veljače 17, 2026, https://tech.preferred.jp/en/blog/multivariate-tpe-makes-optuna-even-more-powerful/  
21. Tree-Structured Parzen Estimator (TPE) for Hyperparameter Tuning | by Nishtha kukreti, pristupljeno veljače 17, 2026, https://medium.com/@nishthakukreti.01/tree-structured-parzen-estimator-tpe-for-hyperparameter-tuning-305b76ce509b  
22. Comparison of methods for tuning machine learning model hyper-parameters: with application to predicting high-need high-cost health care users \- PMC, pristupljeno veljače 17, 2026, https://pmc.ncbi.nlm.nih.gov/articles/PMC12083160/  
23. Simulated annealing \- Wikipedia, pristupljeno veljače 17, 2026, https://en.wikipedia.org/wiki/Simulated\_annealing  
24. Simulated Annealing \- SURFACE at Syracuse University, pristupljeno veljače 17, 2026, https://surface.syr.edu/cgi/viewcontent.cgi?article=1158\&context=eecs\_techreports  
25. Hyperparameter Tuning with Simulated Annealing and Genetic Algorithm \- SSRN, pristupljeno veljače 17, 2026, https://papers.ssrn.com/sol3/Delivery.cfm/5fcb79e3-92d6-4826-8db3-edf94046e6ef-MECA.pdf?abstractid=4432719\&mirid=1  
26. 7.5. Simulated Annealing (SA) — or-tools User's Manual \- acrogenesis.com, pristupljeno veljače 17, 2026, https://acrogenesis.com/or-tools/documentation/user\_manual/manual/metaheuristics/SA.html  
27. Hybrid Simulated Annealing with Meta-Heuristic Methods to Solve UCT Problem \- Semantic Scholar, pristupljeno veljače 17, 2026, https://pdfs.semanticscholar.org/e58b/345ad3b9325f04e8cda3e6075f6eda23f740.pdf  
28. Flexible Global Optimization with Simulated-Annealing \- CRAN, pristupljeno veljače 17, 2026, https://cloud.r-project.org/web/packages/optimization/vignettes/vignette\_master.pdf  
29. Note on the convergence of simulated annealing algorithms, pristupljeno veljače 17, 2026, https://research.utwente.nl/en/publications/note-on-the-convergence-of-simulated-annealing-algorithms/
30. Kalyanmoy Deb, Amrit Pratap, Sameer Agarwal, T. Meyarivan. A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II (2002), https://sci2s.ugr.es/sites/default/files/files/Teaching/OtherPostGraduateCourses/Metaheuristicas/Deb_NSGAII.pdf
31. Jasper Snoek, Hugo Larochelle, Ryan P. Adams. Practical Bayesian Optimization of Machine Learning Algorithms (2012), https://proceedings.neurips.cc/paper/2012/file/05311655a15b75fab86956663e1819cd-Paper.pdf
32. Carl Edward Rasmussen, Christopher K. I. Williams. Gaussian Processes for Machine Learning (2006), https://gaussianprocess.org/gpml/chapters/RW.pdf

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADMAAAAYCAYAAABXysXfAAAEIklEQVR4Xr1WW8hWRRTd4y0vGOIFL/SQWniph4I0JVHIHgQfkkgigvx/H0RFEUrohpqIIoG++SAYCmHeQMMu3h7CgiRFRQUtH1RECTFQDHyJsLVmz3fOnjkz3/fpgwvWOfusvc/s2XPO7HNEOsKlQgzv7hDTAfXddpxuxszF5DSDUor8JCxKeoxGVBDsOj3mmnUfqXDzcNicql1ieCpEyK9Yy94FvmLUx0FzNGAi+AuuBiR6Yhcl3ltGW6eMAc+DY1PHk+AZ8Cz4auqIUJqQ6idKbos2Mb3gjzR8TDEw46glb60Ef62kDGx8ZjjihD+2nUlGryUu6N/ga5WSi2/CBnn7d5x6Kj0+Jcir0iomoBhVIZtrJ7jPCorMaF5q6i+Aj8DRvKjdzcAOiIppi8JEIC3G8V+wf+pqwN5u7A/B+/VlDpnETbQvpqsh5EXRhZ2ml666b4logh/A58CfwN/A64h4V0MItxWH0/V1hLfBg6L3zjA62/Al3KuaJswV0wOeEu1U34Cvi75Ge8H5dVgEPpn3rDAe3A2OdlrpFWScFXyLYP+D87hwvR88FmyFvgaTcDgUFC7CHrO6H8DmuBMqpVlMD/g52BccFOLPgSPAk+CpwsO6Ay63wmpwLjjbF+OiVaDNgReG6+9EV6qGZvkCfEP0+8P4T03ELsTcoKF1ey0t5rDU7smhGHbNQaL55phYiz/ANWq2ytXzWvAhOCSoxCeik+PXnuCT2W9WPYaTTaLxfFVbuAn92zoZ4YrdDPayUMyERK8so18G19eXNfgROppo34tObmS43i7mNcvUdDGZ6Mui96+MJ+GOV2YlVRYKl6uxlIE6/8Lpo8Qj/US7lH09Rok+KW76Fvgvxk2aw0DRiX9mtFVBS/+lbDHMvRFkoxng9GP4tfEvwMx7vdWs7j+pt0CF6aJJuSq8pY9wE+MfSnSSLXBQJkvgs/CeW+C6IA3D8TY8d4PPhIZi1H5TNPcKcGmw+brSzQXFm+CG+kgLJ8/j+AgxU1MXH9UD8EvRlT8DfiX626DQxHyPmWxKJdnVcjIbB3a1I7A5Ya7cARPRgn0yz4pu8pO4l6/wW6Id8YDT/cCOlgNb8j1nFyqA34d0vwTEsxX/7fEdMILT9j3CxKMwX3hvFVSjuWcaKP3XVeoW2NsiPVR2D9xQ3xwPlAzKPXEhlmS44/5y0YLgCTkWzvaaolBMnClfjNe5z27Aein18XeAK/hOLZWG8RgI9zWcZxqN3xd+jfm7Q7wvukDTCosSFdM2WwM+mv9lOxKHx5+ixXC1P1YpDO9P2VQs5GfRFWqB+44NhE2DzcN+8VPYBmDQEHLgnwpzDO4cX/I3dbRM3zAk5+yADanQCSYDm0Xa6p8Gwt4r1VrSSyjFR3rbjN0gc29DCkJDL6W3QqmzxfgfVXqkAgQEFssAAAAASUVORK5CYII=>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAoAAAAYCAYAAADDLGwtAAABNUlEQVR4Xn2SsUpDUQyGz+mkW9GtONWhi4OLUBD6Cs4iCD6ACE6Kg1ufw0kHwW5VdHPwAfQBFNHJwc1FRL+c5Dbpva2BnPz5k5zk5p6UE1IOM4ZdslEzMp1qVP0jVUGscRw71CS2iwUhN84UJd5atZ6XZLh5a8S1C2tMlFmtU9oCXYHHEP2QtMT5hNcXtwczstAD+EJhkR30F+2Kc4JuoqtCUnAUxjgDv3hrDQwlEbziVHoFnSssUuhH9C6Qa/DSYd/8so6FpLMce14+yMqtO5dSi8Ab9tR6tjHv4A/4llKTTaQBxwh4jd6Cf2AvNerSIbAswD5uYPPtaVi/Wpb6BbqZlOlenwGLzun+vtFdG2Mb84ndiHNV5yGGfeV7PPkzXeFtjLpMPaeAK2RljermTfPi02/0DyWfJodg1PGOAAAAAElFTkSuQmCC>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAYCAYAAAAs7gcTAAABGUlEQVR4Xn2TMW4CQQxFZxUUiSJpEBegyg0okaJ0aVJwAOiClFyAjgJuQYUUgejoKGlCBz1n4A7k7ax3x54Z8sV4v7/tvx4QzsUommB5reg01PVABN2UDBtk3np34F9Xoyd2eiBaQvcGIWmrIU7WPOEKqVLh7lQxIj2ineEreB9xw1nD36XXxxFnCn8ga8Nv8BPPDuUD/CiOHrsivOfF+Wb3jdRGXFMZhFbZSbon0twLDa620pfw+OFcGinoPmkR52hD+CPClXypOj84Y8/5vBLKC30hfrpqhYUYdQl72JPPwbOrLnEoCwy/kf/CtzxnrvxGLOyvKFIQap7UNZeQGKUFy1WTdYoEQWRt0gZ51SPvZdbI/Dsya/wBRH4bi75Hpz4AAAAASUVORK5CYII=>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADMAAAAYCAYAAABXysXfAAAEEklEQVR4XqVXW4hOURRe23WGSC5h8uCa68Mo12gUHqY8kEhSDA9CJjWUW26JpHjzoIiSwUwhd+ZBRo0QQrk9IJFEEeVF4lt77XP2Pnvvc+b/+er7997rsi9r7bPO+YnKggpH5ieriSFiYSeIqgtRrn0Id+FSZnNtIoEoRL5v6Qj8cjdRC+6zQ/JNA3jq3kkn7haXAsfBal/4PxgGtoJdctdsP4mt7QU/ImIMAB+BA31FiGCGQNAVfACO9xVloiUzCpaJwNosBy9H5PmSGGBVj+Z2RJ6PuDJ7GA03nXluGhzQL+AEK4pZB7JAcBes8+WBlYEr13270chhysIx8IwvNMjbTgbDYfUHbX9fkedt5cHhW3KdSsMK8BfY2VeUCLUUP998aST+0ZGbGUozE1hEEJErGkES2Im+aiXYApdLaAeBV9BvQ/sGXODYHQTvJYPIEnXgHZJqcwKcTHIVToNzrJmGf83mYsazaK+AU3h2Mz9KuHoqMh+KM7PIDomGoDlJcnX4pM/B6Ua9DPwBVpmwNqG5LiqzlD1RHbgF7AhWksz1EOwD3iI5ZAqVzcxI8JxRtUF0yvRZvYRkrqFmD0nD+ASuSUxZtgHNLLCGxMmNIPdZttCMz5NEOYYLZI82isSvXsnB2GdGYmjgZmYrPKfBnd9f7LeJhWay4+BbY2el0rwAt1mdxXbof8KoOw+My0aSyWuNoMkwgBMtblbjB35qqGtjoQ3TwyQRAPaSrMdXPcE7WDQ6Ni6egbt8IaD4JXTNE14kmbyvGR+m9JplNuELGtF/5WhiuJF0nHmeULbKjSOTYT1ykyL9j2CDv5FOJFVKp9egH3GmSPFDn4C/xdK778zB/ntIikUXkhfaUaumeSRvbRfpYTQUVeCHN77Zka6TDFO1u1+n/1vZRyDFJDMRR5RtO4D8EPL3V4VjxxvijWo4k84kieBaSFeZPl8ZBgeFs9lDj2x0b3gR5TXfgzvMuBf4AfxsdD4Gg38wxRhf0YDpv6PdSRL5++B+ks8GF/wM8EZHe/KeJA/5LcxzHQvMJl3aVTPJneaK5iObGQEXIa5qV0n0v8HmTNhsn0vyV+KDZoNCXN+vWVnhn603pCtgno0NfXYTAfzDVFH20DVKArfckbk4AB7yhXwynFDttqLCTfCdfswdd7+FHnFles2U/LfB85kpQJwhDlxlJDP8jL4FxzoKDf4c4AjMd4UFga2A/DXaqb6idOismszoRfj9gre5/lxiLAa/qsinCkPJd9kRX854SVIxONrrfWUUSh/kJkmEPMQjEIFfzfDcogBJ0eHik/OO0l8qbNPNV4RoZy+OmsvtTjukwnSmsGrnWpcFLjT/8Le5cF9ZZaGpRvsWjEKrPGWePERpltaqNPs4/seX6C8MeqQAWHLB9AAAAABJRU5ErkJggg==>

[image5]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAYCAYAAAAs7gcTAAABE0lEQVR4XoWPoW5CQRBFd0NCKxD4CtKE/kUFAskXVNRhUdiKiiaIJv2AyiJA1NQhUNX9gSosKPgCevbte7szb5Z0kp17986d2VnnQvgqN1COWCw4pWTL2lk2+X+erkM22wYzVcWlql5PodlVvV/LZVMpYqEHvMOP8D38odXwwRkE0uFs6DqDP5wTJ/Bx9Pk70mfkzj0yZM2km3qlLtqCs4pe90YexYe8e3FxjToq1gV+wWvONq3UMqUfAkvSHD5NloKp0Z9IO4R+HilNOQWYkZZG15fEWcFNtK4vQvJfwK0QBAoOXAEHWMfONQ3+nvTdFJUnX9Lo8LlnM6M0mtsrZCjESm8NzAXD7Rr6jRyyWcUFmfgDAeUYJ8BUuw0AAAAASUVORK5CYII=>

[image6]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACIAAAAYCAYAAACfpi8JAAADAUlEQVR4XqVWXYhNURRee9D4uY0XNSialJJpGsWIYmqmxIPw4id5uVOUodS8yIjiQUwjIg8klFH+okSU1DSePHmRiCIlv+XBG2l8a+99zll777Vd8tW31zrfWnvtfdY5+9xL9Lcw5ZBHg7AObZKmZZCkSiEJ/hey1SaBN8E5ccDCT8vMvgQuKi4yORk5xRmkbohFi6SEiaWZ4FNwVpJaoHoF7LAPvA3uqDIstoK3Is0hWzlBHan3YtEiqrGd3CYmQ78h9CbwPbjGXslHkPMTWLUZ5ivskihIcnILho9wF8DvhT8sstaBr21askoiqBBZF8Fril66R8AxL53FUL5Y8C9jOB3c8R/vXkGV2Af/J7kXP8EEkFu2n1zCqFwA/ksMu4VU6HlB7lbC0HyM44h0xSFGLwLjsCvBvfBXiNh0sjGzXmgSy8GH4BPwEbgUi12AvULuxdfAHdkci7zvIQzf4bbDHyxVh05ym+wuBAsX7gD5sdW8yo/2G4KdCB9FDs+rKc35BPYr/bIFRsGTmMAnRKLLd4sXjXEKnOt9rvsO5rq/HqL0E1Bs6AV4QHlyhrvxhfwXU8ZN1RG2IcLEdnJ5daGGqFrzHDwUt2kGcQET775MwuYMv1zd8Q5DmJ2+c21JpBxK8wEc8GGLqcQfLvsszbJKDlaZRm6BtVL0OdtgDnvhDviqivOGzAlxLfEL3FguY/i4GuINjMHvEYl8VGtiQ2/g76nCVm8mV5A/Th2QfsA+9gkTkTMCu9hfi1szbb5zC0tJRPvBcxDmQTpI8QkhexxHgkZZ3xzD8AB8BvZAuk+Ow0Z/uRl8bHGyqMmVCIM4KWY17BbyX7woXgc/c0zcWWAao/x1OE78Ky700KooF2vByL9Dm2TUISyQbDQEHhm9BfmExVAm+J7JCPxdMHeloM61kN0K6vSB57PTNCh3xh86/j+yKogkRROhQCu5DyefVAetgD5d3pnFFPj8P2V2VUOfGQNZV0n8VVSQL5SPULiJf9tQA0FD0pC8L4VE98jpvwGoyGUXoWKwpAAAAABJRU5ErkJggg==>

[image7]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACQAAAAYCAYAAACSuF9OAAADH0lEQVR4Xq1WTYiOURQ+l2GivmkKjZ+FaQqDEkXKgpIVGwqzYDEzkZ9MTSyUsrAgC7JSkr8ozfhsKAusxt9kgZIk5afIX4qU1aTxnHvue//f9/uSp573nvucc8+999z7vt9HZKG8xtgGGalJZAZlpBJkFpGB08sifKjqzfg1+A/oAq8XnWTiMltwCVya0X2Y3VTCbqkVz3tgZ6Abu3Eemgk+A2fFjgBNJCpwBhzMx2dK45fPufvAW7E7QCZVAujdaH6DbWUxGhlnJLVC+I52eRqaKoL8zk6A523PINlMfmw810UIw+n0mXEOoYreRzRbYz2Pshir94Nj4CTna4gg6VxwHFziixxTNnUCv3JE88Bx9FZYv4bL1g77ONqr4H1wC/gCnGD8m0gW1G76MXrBUcVvkKIrsFeCw+AQJtkQRDqMIb4nt6PF4FuSMjI6EPQB7XMXQgPgn6IT5egFD4ETwSkkC3+KmGkIHIE96sUKpFpf8dgbefR3hQc/KAQz2RuSV7zAAfCL1zfQR3bTOzl+E3lBA0oWhwrRGn8Dnv0KPOy6gl0kCbZ52nyjFRUDFFfoW3EPSiZgew8ePJa/5lbNnArjJfQjsbNOcttrnqNIyrstsJ0kTsOsKwe+g69jsQSfwf1iumR86d5FWh32p1Ci9UqqVrOKrKoFzVEYm2FPRssfPP2tMmM3gn1uA8Eu+E7yyyMwrn3gL5LzZuw0E/NCjaTRZfRlVhH3WmJdcR61W9tEx0zEDPA2WEuKqagTD37tF8UuvtSnSH7wHoKXSZImtx/gSvKd89FGcnFHSCZfBz4C64rvB79pefSAP4g/K8lqQ+wgWRBf7AAYdwHNjVjXKEtqjinjPgmeNm6LhSQLKD5+jGvgEzH9RNpaRXKxZzspjMlMrOHrsFvQvCf5/gXOOyQTzDF9nvAnuNpGeDDj+HgOBg4PZQuK0I/Ac6EkI/kS8ltxl+Qf4Fk4uiuTKpqO52Nym4irWA5xd5D8wZsa+CyaypHYC0h+p7RqTq54GGQSizSk+C+sEao3k9NiNIipdFc6/xWZImSkRHAFTCIs/gLQrHAnGX8SdQAAAABJRU5ErkJggg==>

[image8]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAFAAAAAYCAYAAABtGnqsAAAGAklEQVR4Xr1YZ8hdRRCdtcXYe43yYQlRY8PYa6KiEFF/WKIRzWePBUHEimIERcWGBUywJEYlmmDEgl0TK4oFRcQeUeyC7Z8i8ZydvdvvzXsvJAfOu7szs7N9dveJeJiQLHINnDRWVtL2U3fQF4KLSiUDoihdERSipccycNkjBqp5oEIOlTUSUJdatKkqq2BlcB64hVf1jpngLiHbVmuBWUmu/wV5OHh9LuwB64KPg2vlCo+i/kJQiO4Cj/a5in0HNgE/ADfNFR3YG7wtF/aBrcFXwVW8pGhzIYhxMPgcuGKu6MJlcDof3zMz+YmQP5bJllC/Q7AZBp8Oim6g2B347OvS/WIE+B64q80VDgpBG+6D6aW5MELi6AyQg7cqODfSrAB+L7odohK9NCIJzCOQ/g3fcdWyscjYWX83kpQoXCSC88HXYkENhYsSe4I/gyNzRV6Ye/0ncAw4AcqbItxR4JfSlBkI94OP5MIKJoLX5cJOpK17G5ySSDwq3aiIIiwCz9NkuyEby3hBTJck4Mts0e3UUVy8srCxAi89FfxX9EAKSG2IWeLbUHhMUaq3AReDG+eKNpQuEswEX7KpFkNuF26tK4QdM7IgVctn4mcgRYu/CIXFtqKd2z2IUhvDEGLsgTMoTgb/yIUJimbFKJSngX/mwthsgtFO7Q9eAu7XKCBfW1R3VCPIwJPyBdEtw1naQxh4RR4SeyApbLFQlivweJ8rMdlEZQOcAyPriF5NHhaNc8eCH4vGauIW8B2XrmEK+JborYC7i3GOYWWOaOjIcYDoGPAmURkCkRvBv8EdwMsz3c6ihelEETzsKLYBZg2XZwj4XbQMO8hyjS6umEH5nJAt8CQ4qtZQ0TZ+LRoKCG7T78CPvIXIo6LXj4DgbArS7CN3HQ8GtvF9cH3IF4oOrIMvtBO4GLmxmdyDHV8geudqZlFhuNUMK+Fg5bgdrrZ0aXr9VrTxzHBS8qtQg09hcWUudO3aAHw5VXjwaoLOmtdtLvTjK6Tv9jm9AHM11fCEhJJjjA4gT2wO5hxoDvSWAewj7Q7K5B5cfb+KfWHEo2vTzQrktwtcGbQbzhUprM9PwGmZosFZoh2qgTrWMTmSjXayZkUSnEQ7kRblgmkw1S2OrXJFhlHCOoy9WCsin5xxOmlbLXy2pVvYohhoNIZ2ZihStOFH8MJc6MB4upHPpZ2fKxo/14xkU2HD9vH65WCmm3wLO2RjyRj6uc+1D/RYN9DjcsVqohdmKvdq8bC6qP6IXAGcBF7j0twaX0S6IfDWKG/havhPNPCnUhvPzLOFPHy4LRd5tYKD+oMmvR/G3ySWOc1K4LXgMaLPO9487vVm+kwd9rkwHDxUOQZhYp2S1xYMnI2B44PSXll88IctG32BzysYjzgQPL0YH/+RcPNnQx8Ed3P5GEOijdk+k7NJrINXhjawXX9JeBXw5URfSbwzOggcHC9w4AOB9vRztmjZ5rK+oeiqjVd3g1PAXzQZnMXgiTjDMBYYuQpG2Xa11xIOSI4bQK4YXiE4Ac+Q8MNXTO3QIXh94UmdH1bEK6L/grSBk8ZVzevHG+ADooOQn+iMaZRvx0zUZb62ONgLRQfrECjfFF3F04QwbKshHeW5KDzz5fRfhDSZuKr3NhvBG8jPg+z8rujCo1o62pCAXuOqZtcG2l6yjDyLAWBcGApqjdlPJ10wjnPPmWbrDcSRhGvp2NBrIn9GvAHGae5IWT6yCIr8pOr11hQO5rC0wepFTP6P4SrkLRtCWGTmIVjvfUIEfv1kcnwKW9QjSBUJg1FUK4IcR39qY+zrETtL+ILni6Iy0wGAAAAAElFTkSuQmCC>

[image9]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADsAAAAYCAYAAABEHYUrAAADZklEQVR4XrVWS6hPQRj/jpBnecfCK2VDoiiF3G4pXRasJGVhcxcSt5skXY+65RWJzS0LySsLEcpbd6EsLESSYiFlY+WRZCF+3/nmnDPvmf//ur/6nZn5XvN9M3PmHCIvClvgFfmRbRiGE0IXFBPw6NUEXjghhhNqsoPgXUNhYgT4APwBvgUfMeE7iAB/0d+oGytsAxeo/mXwHYnfG5I4ZQzwN9gpZoXKJ7oEUWWtDliNA5+BnPRSS6ecyscY8BfYZewf0Sk0yzVRhcOqHQ+bq2hHqvEx8GZTGJ1Af67SCZo5g0kLspV1sEMkRT6n+O52FrIgk3iA/moRFz14TGvMSswAd6r+JjIXYxDcLd0yh36puy4+gLg2AMOJ36tLqr+BpBhrl+ok+Ki/qIVEZ1V/nmr10LtICmbMqaXYZZJjq88x27MFNgLiFEy3veAKTc67e11UTvx7JMf9HPgV6j5Lr6PfFki4Yh0e36g50mk4aUggbviIcRJ9hWotbq59iMaC56uBism7+wdcWMkVRoHfYbNejbeCa1WfC9AxH9xhySocwUx3mqFTSStIOCu1avaAi2pd48u7e7GRl1hFcuvKu1nQLJIFWAn2WPPuAyfrAg2Pwf3c0T0SWQ8ZE8FbmKUDrU0+glzYstJSwN/L19qYwbfzU3CqJT9a98wq+Nbn27wjXmnhyLog4Ov8CfiSPMfGiWHiAElBMd4AR4OnwU+I+IrkVRgA74MfwDPWTIsLOeI6eDHY7wpJ3Ntgd7Bgrc+ryRcIXxRTlIzbz5VBwC+ObMMk+IhyjkOEJDSA5ifamUrK13s3eFKNqc48u4BsQy9K7yYEv68O2tmA6STfqC9FeavRcZK/F/7VyoyR+nALojZhJd/g1Q1tIewUAv+x8JnvbX33FKxtSKIt+6GhCsE/1Vzsdk1XYU35bHMNokgUHNakEPdkLX8DLxD/gYhtUchNd02zGzbodeuppvuewjJES8D34EfwIcknYItu0CqiCeVAX4FG4AnnCEwk1P8N6XnCFn6NX9oacmIkbDx74Agc+bDBM2F88rhW1HmfHB059n4bM3O/TRhB+5AiJI/CWRQzSjpmwiKkDslTcBJVBegw170VeOK0FqAV5EzmCMy6dbRfdR7+R8y2YoSc/CtmImTjETnIsfHgH4baa51YKvIuAAAAAElFTkSuQmCC>

[image10]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAYCAYAAACbU/80AAACeElEQVR4XrVWTYvPURQ+RwajZEMhkvKy8AnssJGFLR/ARrIYSk1JiWRBlCRJyfgErKYsCQtNTU3UeEnKRkoR0iSe87tv595zrvnPwjM99577nHPPPffl92+IWnBuslCPUqMRBcdZJJNlROjcBoss5kwqUuP05zkLOOjpGiYmLeg7HNtBKTpek4l3rs+iDioY9Gk0E62nwEnvJ1sKcoYDsP+g/wSOW7eLY+BLUOYJH4OPwOfgO/CH8m0zydQxSvsMfEgh+Oy/N6idPI7uJ4x5FZCd+DuKfgEcq/0RMRWC+Cb6rTQk42/ot6iwDFMQ0z4KRV9LQonJ1kyWHKwA34Cb4/gG5iEh3y4hzqMr9kUKBRzMCtFO8HgweQyhD6Id3fU2TkC4XIa8keTumH6h3150jSrBU5JTY1odx3LU98Ejw8gcWY1V4ByC1slAHd0VCru62+jZiqM1aOR+00MbyKHfoANVpwbhkzvfqNJJQV9ZHg/3TmHAIQqLDTni7D3gixyRYE+CZfevwbWNnoxLFD7LKeXMiFFXKRSwP+0I3V4YF1IccB36bjXOOA2eGSxbnUBO4QucC3DvaJ0Rs+B3cGXriAWtp/C7YCAPZo7kDhN0EXE3wDkavgi6V/SMTRR2P12pNW6Bh1tREp1C+5FCdQNZ2YX8hMIiv8FdcXaC/AqKb9KpXU4EXxe9BZdlPWI5+IGal4uQ9Hr1Sy5kuiOToZ9EIwW+j75XZAqnz+HkeFLmuCinbNRFpZFh5ibBOGqU4vqBfc8I0JNHTuQEGkkXbRbp/RvQOIrdCa/gLNaDjamX1Qvb2P8Gf+8uluA2p9ida/x1pD/PV/uoS/sL54hqmBYxwUEAAAAASUVORK5CYII=>

[image11]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA4AAAAYCAYAAADKx8xXAAABRUlEQVR4Xn1Uq05DQRCdMVAED1EkCaYCwx/gQOKqcHwIkqAR/AkJSVEIEoJBYJo0wfEFJCBIczmzs4/ZuzucdOdx5szO3r03Jcpg4zi5Bh5v4G2Uuv7t7hR7zUUWJ4zYnClVi/pxDcvPkQxg1qCXyB+xvrAG8L/wL+Cf4NfQDPAn2sZ0D3OFaNecZSGNWDNzimOYb6ypZDsoPKT50U2iYKWpBb9HT5cwF6Yg5ozCNL4rVMAWhcfg8JvBbOaS4jo+81wU5fS8AXOUVWXDHD8jkufbM6WIpG66eJv0Jl8LZcuGSIjpeZx2k9iRrGC0yS3pxZx6E6rMiN7gfhDLK6ngNYg5JP0y5Muxiu7kfVahvJ9P1tcgzR+BYzrILVWz3bCBZV1hyyjcfwP/ujvzQlTpbaHjStwb30z2iMQ30zx0TuEcySOI/gCOjimK20qVlAAAAABJRU5ErkJggg==>

[image12]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACMAAAAYCAYAAABwZEQ3AAACwklEQVR4XqVUT6hPQRQ+47/yeMnDhqQsLCiRIiQRG1l4SXY/C9mKlLASsuHpZaHsLJXwXhGlJCWUBQtSUrJQilIspOc7c86998ydmfsnX30z557zzZwzc2eGHAG+6QIrDAdFkUZH/csgG2DkJu08qAVZqSaL8kd2doYA3VS9kJiy0dUYDB2JxQtsXK1oV6omROSKHIrQn1vAKD6n0P8F38F+hPhP2FOw/6B/Dj7ROOu22rpyqT0agxaF0NEkmrOwFpjoQ5LEq1TDzVo0v8BF3SpRWI1dhYW65oP3gwDRHJKkH+LB7k00t4XqI79H2msxgOhQzbeLZFeu1fxziX9hgMbS8juhqIf5N8wuvyRwnrgYR6M1/yxwdekrobMlciZclFxALCw9z0h2ZjgW5QZHjkTO3NgCcWDI8Q1y9IKDcdjjGHil7qzDjN1CskCLATjTW4WwGqCJHe1Fw7tysQyVKNUbwI0mkCu6wGJQfrkkWYqGnw4ppgFXMYCL2Vm5WlIpuqmIhfvRPlA7CPhGO8ZrdL9h8/U2Wm8NgefAx+TfGu8/hWYCYd52fqv4Bl6XmMcJ8C64Qz7dBGb6DOMtyHYWK0gOrlxfU6F2R528QZ/AlSQFHQHH4b+Ffho4Avu76jeD68Gb4MAs7Cma7eVnBTdCkpz5heRKc0Efvc/RMiNmexv4Sr/nEb89/rD7Q8rYDfI3g/Vc/DeQ83CNfEP5vMhzEqw1QnMUGEfwOHEi0SwEf4AzdBDfMvwuN6z6A+AdcAnJO7XHVQ8n+9qQqERc08Gv5CdxY6raB05WGvee+CF1dFnCdJvkJl0iecFxxtwFJztzWjUpJIoIwYKXaMbQr1H9GfCkUfB54DOySafjA30D9kFVrIN9D/1hcLn6LFqLINakVWlv3l9sYN1LDYFesBP892T90Dm1BnsvuF2bUCSq6p3Y4B+fXVgaHLptGQAAAABJRU5ErkJggg==>

[image13]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAFAAAAAYCAYAAABtGnqsAAADz0lEQVR4Xt2XS6hOURTH15G4HtejSELdRJJCMjUgjwkGHikzGRiJmRLRRYyUomTEQJIUedxQpLwyoQhRBmTEBHFDctfaa3/nrLP3Xvvs/d37UX61vu/s/3rsffbZZ3/7AxgSCv9aShlUafkF8jMiBG7pr+L16QkSzanfhZbRFqZYkVkzL/q/wM5TGE33yJ3oJqLVos7OM5juA7kBqR2GqEy7JHcfCJRSwN2Mvvo2oP1B+432Cu0WRn612i+0R2h3rZ+0JTYP3JFULa2rFFLuVNMHxTS0fjD3WLzH79to77ht7DnQ3AB8se3dNg+uou1FG98SkJs4RgqaLbT5aN/RJuk3oOk6+RkhAlU8yRNcdqFdQOsR2h7gydoqtIloT9HWU2McWp9wEl3YGU3UG0cnnrlCu2i3o+nJeAU8oU7lvo42tnIYaMXRBPY4dc6izaOLLWibpQdZAZx0wtFHARdMIDDogJTJNixxA78fAm8paxy/obmbYMRMtJNSwKgx+PUDr0KLhl7v4XRBr+jIug8OAU8g7Y3cHfc5Ar/mllGVLnkC1Z6RYgdNllfHE+hhPkB9tG2vBc5fbtt9mNEtB+uX8ATJZLQpZYtDVwH3cazUDca5oK5ZbBf3gRMnSF9HiN6TpYCd+PkZeJM3guUF2nm06WgXax61cMPk1vUjwPNg9roS+ZBaimh2A//yPq4kgddxgJSYdKai/UQ75TqA9+6X2OE+/N4Y6ZdW1ml0H+CmDKwnOZ57wCcP+ePaCO0rNOuHuSknVx+hPqSQoBEM3AE8nk2uA3j1fcS0a2D3oxqmnPlYDOZEUXyqB0ShSaMHR5MYJjhcft/l3pJLtQfyMci3ut7bSgyBYzwOHEebvMs5jKCbnMNN5Y54EazEi1muR+Jkt87GvVpVxvMWNAF0mOwyrUovr9ompYQfs7/gG6HjlmQY0C9yAd9KRebmXktY54VUwDLXqaWRowd4sOpRJdq/J9SJ5gYwMQUsBH6V1tkk+lyERq/tJeDx0i/o9lZKHV2qPMGY1/jZX9iFFIM6pwkj+wDVK/bWajPKyEA/FbpT9ySzFO0y2hUwYyqOAh9iqTQdb+hfAR+HYkQHYpxngOvTX7bWFkP75h201VVsBvIpRfvXkEnRJx+o7sUPFfkVTYaX5gkQ1lxSYgxpgeHBpaGlJeveM/QimmkrLyPeG6OnNmAHmJFhyI03eEmeECElNiVGIzk3OTCLthZKOlplTc+nefVp+j+gk7MdLRvtWOpajCXgbn4AnSZj/El4NdI60D0V3nMIJAUkD62ElqvpEhkzAJk4kg91+9bkAAAAAElFTkSuQmCC>

[image14]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAVCAYAAACQcBTNAAABF0lEQVR4XmNgAAFGMIkToEgTUAsGEDVgkhFJA5SNzQRkCWR5DDaS0agUFtuQADY7QQBZA5iLYGG3Gs0POKxDEkIxCMpkZHAEEuuBrCNAfBrILgXKMGExgzEGKPYCyFCDmqQEJF4D2Q0QaYYaMAkE5kD8C8j2RXP3DCD1HMhiBtJHYeLngBK3UHwCoSqB+D8QxwEFJoEEDKECUyCqIACqLwcqdwqIZUFC4YwgAUaGRLipCOVZUMVJMDEnIPEfKOWLxRl1YIMYGCxgYhxA6h4Q90NUgUWlgOJ9QAYoGH8D2SFAOgCsnBESTEuAnI1A9j4gvRAo7gq1IxXIPgGkd8B9ggEwnI8CsKRjZAEIhawIRQIJ4Ex9uF0GAEORIu+tLZ1zAAAAAElFTkSuQmCC>

[image15]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADEAAAAXCAYAAACiaac3AAAD4UlEQVR4XqVXS4iOURh+z7jMLORaxiUxaXIrykIuhZCVJGIsxiWSCeWWWLhmYcFCUpSyQSS5jSIzi5nCggVWVlYoShTFSuN5z/t+37l+///LU+853/u8z3tu3znn+38iQyk8zj5mNAGViVehUloZ8NGQiEQXabOpFWSWBkxVIEEotC02nEupVnwt4yDD49JwyjD+ha3kK+gKmKu2lPWIQkFbC2B3Yf1g+1AvtrEkiXLcJthjBF6gvg2bHMVXwVbDBqnPg9mOustJInh9NMO551wPwQjNfBSf4M5TYgXsN2yZ+qxxj9YrV2AX7A1srIaOwz4jVviMk7CB0oytX8ImehpBukCmBcX9mM3gFeyCT6CtG6ge+pwOvXhktJJMdr2lZL82ofyA+lAhBU7B3oH/jvo57AzUw7x4BO1FK57EgzKWRzvJCu2J+BOwH7DBEU/ecvGW4JWdXfIS6kX1pBCRtLWl9JLVTogALQh319F0kEyi08oKraH9yi9RJoToLpJoJgUxOVs/SRaRcYx4EpxT3C5BZ4rQLT1upDuOhjAHSAbCk/GxW/lt4nqdu+b4vLFmXMlI7JbyU5XlSeCsmGuo+2AfTNlu7dExdBIhXJJ9Okr+JFywC88DcHeKm+3qKcWTEB2fJ+ZnKHmYWGuKN2M2QPUHD4s0XoJvCE6sYcb3T2sed4DBmo5onDx4ngTve4vMNB5RfhLXlW9Xn7fbCIlanws+/HzIBZnGBTLz5E1E2EHS4UZH2a746uRDuzbfvmV5e3DueHZLnaGbyo8pqBhGJiC5AR20ZLnyYOcHYrGOpLGt7HiX6EHLG1paEBmcJ8ltc5TN52ud+SaNfYSdi0ahW9HMETc3QuHwJkzNN4FBTyHpcF8QsHe5/QaMtF5ujeTa5Ny5UeAZ6VYBvdxqDPUGCvnYDeCyGh7xUS/BdqqYqR2c6UdxJYrgQ2fu+Bz8ThQL9ZmLUSTfks2lhGgo7Ctsr/qtUL4l+0Uv3/MQkiu4RzU1IdvJJVehDfH3qCeoPxP2DTbNSYh/mvCq81fXxxoY/2YqfhfxGXtN7hvBwFbClWrs9mIcwZh+odat5HCZZGa+8Sv8kuHZeCso7BSnk/w84EN5ibwOdAH4JwavcI8QwbKshHeW5KDzz5fRfhDSZuKr3NhvBG8jPg+z8rujCo1o62pCAXuOqZtcG2l6yjDyLAWBcGApqjdlPJ10wjnPPmWbrDcSRhGvp2NBrIn9GvAHGae5IWT6yCIr8pOr11hQO5rC0wepFTP6P4SrkLRtCWGTmIVjvfUIEfv1kcnwKW9QjSBUJg1FUK4IcR39qY+zrETtL+ILni6Iy0wGAAAAAElFTkSuQmCC>

[image16]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA0AAAAWCAYAAAAb+hYkAAABNklEQVR4Xp1TvUoDQRCeLTRYCnkAO8HWzkbQInVA0uQRfAk7sbDSV7FKkVSChSAIQfIEKQKxMU2Kyzc7e7ff3O01DszON9/87m0iQhIIZezF+HgWkhLfwSXpizu+vBJXeUwer1EY1OVdyhX0A7qHVoh8w86gc+Af2F/kV32LaGIFPWvxKpf+hoZPcOygn607mDgiOyOxKY98B+DnQkEz7Un3hndDwXuY1zw5FVDdu9ikKhUnLBPO/0dRvZbhU7HPPW8iQQY4N8BDl2oSmbFY14dMhWPRJ8gFd8k2n/ZF7FGv64wUZlm0mSW8P1BHDUOXhZmKNqaSC9HVgrxRUjyCHbcwW+BzdfU99GfzBaxfbB39oFyYBcOr2FC0YWrHC+Zt/F+F+ZojhzAXu0Yc75NOUyd+QhmLHAD7UzUuoLUfnAAAAABJRU5ErkJggg==>

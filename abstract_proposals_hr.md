# Prijedlozi sažetaka za konferencijski rad 


### Prijedlog 1: Fokus na sigurnost i robusnost u dinamičkim uvjetima

**Naslov:** Evaluacija robusnosti algoritama za praćenje pješaka u sustavima autonomne vožnje uz kompenzaciju kretanja kamere

**Sažetak:**
Precizna detekcija i kontinuirano praćenje pješaka predstavljaju kritične komponente sigurnosnih sustava u autonomnim vozilima. Ovaj rad predstavlja sveobuhvatnu komparativnu analizu modernih algoritama za praćenje više objekata (Norfair, DeepSORT i OC-SORT) primijenjenih na MOT17 skup podataka, koristeći YOLO11x model za detekciju. S obzirom na dinamičku prirodu kretanja vozila, poseban naglasak stavljen je na evaluaciju tehnika kompenzacije kretanja kamere (CMC) i metoda asocijacije podataka. Kroz seriju od 18 eksperimenata, pokazano je da OC-SORT algoritam, integriran s ByteTrack logikom i CMC modulom, postiže superiorne rezultate (MOTA 40.4%, IDF1 51.7%) u usporedbi s osnovnim implementacijama Norfaira i DeepSORT-a. Rezultati ukazuju da DeepSORT pati od visokog broja lažno pozitivnih detekcija, dok OC-SORT uspješno rješava probleme okluzije i nelinearnog kretanja pješaka. Dodatno, rad demonstrira da fino podešavanje (fine-tuning) detektora na specifičnom skupu podataka može dovesti do degradacije performansi zbog prekomjernog prilagođavanja, sugerirajući da je za autonomnu vožnju robusnost algoritma praćenja važnija od specijalizacije detektora.

### Prijedlog 2: Fokus na optimizaciju performansi i računalnu učinkovitost

**Naslov:** Optimizacija cjevovoda za praćenje pješaka u stvarnom vremenu: Utjecaj rezolucije i post-procesiranja

**Sažetak:**
U kontekstu autonomne vožnje, sustavi računalnog vida moraju balansirati između visoke točnosti i računalne učinkovitosti kako bi omogućili reakcije u stvarnom vremenu. Ovaj rad istražuje optimizaciju cjevovoda za praćenje pješaka temeljenog na YOLO11 detektoru i OC-SORT algoritmu. Analiziran je utjecaj različitih hiperparametara, uključujući rezoluciju inferencije, pragove pouzdanosti i tehnike interpolacije putanja. Eksperimentalni rezultati na MOT17 skupu podataka otkrivaju da povećanje rezolucije slike s 1280px na 1920px donosi tek marginalna poboljšanja u točnosti (povećanje MOTA-e za samo 0.3%), uz značajno povećanje računalnih zahtjeva, što rezoluciju od 1280px čini optimalnim izborom za primjenu u vozilima. Također, rad pokazuje da primjena linearne interpolacije i Gaussovog zaglađivanja putanja značajno povećava odziv (Recall) sustava popunjavanjem praznina u detekcijama nastalih zbog privremenih prepreka. Konačna predložena konfiguracija postiže stabilno praćenje s minimalnim brojem zamjena identiteta, što je ključno za predviđanje putanje pješaka u prometu.

### Prijedlog 3: Fokus na usporedbu metoda asocijacije i detekcije

**Naslov:** Komparativna analiza Tracking-by-Detection pristupa za detekciju pješaka: Od Norfaira do OC-SORT-a

**Sažetak:**
Paradigma "praćenje putem detekcije" (Tracking-by-Detection) dominantan je pristup u modernim sustavima percepcije za autonomna vozila. Ovaj rad donosi detaljnu evaluaciju triju istaknutih algoritama—Norfair, DeepSORT i OC-SORT—koristeći najnoviji YOLO11 model. Cilj rada je identificirati arhitekturu koja najbolje rješava izazove praćenja pješaka poput naglih promjena smjera i međusobnog zaklanjanja. Istraživanje je pokazalo da segmentacijski modeli poput SAM2 nisu pogodni za ovaj zadatak zbog iznimno velikog broja lažnih detekcija i negativne točnosti praćenja (MOTA -46.2%). Nasuprot tome, OC-SORT algoritam, kada se nadogradi logikom asocijacije iz ByteTrack-a i kompenzacijom kretanja kamere (CMC), pokazuje iznimnu stabilnost. Rad također analizira utjecaj treniranja detektora, gdje se pokazalo da standardni YOLO modeli trenirani na širem skupu podataka (COCO) nude bolju generalizaciju od modela fino podešenih isključivo na MOT17 skupu, naglašavajući važnost generalizacije u nepredvidivim prometnim scenarijima.

### Prijedlog 4: Fokus na rješavanje problema zamjene identiteta (ID Switching)

**Naslov:** Smanjenje fragmentacije tragova i zamjene identiteta pješaka u sustavima mobilne robotike i vozila

**Sažetak:**
Jedan od najvećih izazova u praćenju pješaka iz perspektive pokretne kamere (npr. autonomnog vozila) je održavanje konzistentnog identiteta objekta kroz vrijeme. Česte zamjene identiteta (ID switches) mogu dovesti do pogrešnih predikcija ponašanja pješaka i ugroziti sigurnost. Ovaj rad analizira uzroke fragmentacije tragova na MOT17 skupu podataka i predlaže rješenja kroz napredne konfiguracije OC-SORT algoritma. Usporedbom osnovnih postavki s optimiziranim konfiguracijama, utvrđeno je da uvođenje kompenzacije kretanja kamere (CMC) temeljene na optičkom toku smanjuje broj zamjena identiteta za više od 50% (s 585 na 276). Također, integracija ByteTrack logike omogućuje asocijaciju detekcija niske pouzdanosti, čime se dodatno poboljšava kontinuitet praćenja. Rad zaključuje da je za pouzdanu percepciju u autonomnoj vožnji nužno kombinirati robusnu detekciju s eksplicitnim modeliranjem kretanja kamere i naprednim tehnikama asocijacije podataka.

### Prijedlog 5: Opći pregled i metodološki pristup (pogodno za širu publiku)

**Naslov:** Eksperimentalna evaluacija YOLO11 i OC-SORT algoritama za nadzor pješaka u inteligentnim transportnim sustavima

**Sažetak:**
Razvoj inteligentnih transportnih sustava (ITS) i autonomnih vozila zahtijeva pouzdane metode za detekciju i praćenje ranjivih sudionika u prometu, posebice pješaka. Ovaj rad prezentira rezultate opsežnog eksperimentalnog istraživanja provedenog na MOT17 skupu podataka, testirajući 18 različitih konfiguracija sustava za praćenje. Koristeći YOLO11 kao osnovni detektor, rad evaluira performanse različitih strategija praćenja, od jednostavnih heuristika udaljenosti (Norfair) do složenih algoritama temeljenih na Kalmanovim filtrima i optičkom toku (DeepSORT, OC-SORT). Ključni doprinos rada je identifikacija optimalne konfiguracije koja koristi OC-SORT s kompenzacijom kretanja kamere i interpolacijom, postižući IDF1 rezultat od 52.9%. Analiza također ukazuje na ograničenja trenutnih pristupa finom podešavanju modela i neučinkovitost "zero-shot" segmentacijskih modela za potrebe praćenja u stvarnom vremenu. Dobiveni rezultati pružaju jasne smjernice za implementaciju sustava percepcije u stvarnim prometnim uvjetima.

### Prijedlog 6: Fokus na metodološki doprinos i dizajn sustava (bez otkrivanja rezultata)

**Naslov:** Dizajn i evaluacija robusnog sustava za praćenje pješaka u dinamičkim urbanim scenarijima

**Sažetak:**
Pouzdano praćenje pješaka temelj je sigurnosnih sustava za autonomnu vožnju, no predstavlja značajan izazov u dinamičkim urbanim okruženjima. Ovaj rad donosi sustavnu analizu i optimizaciju cjelokupnog cjevovoda za praćenje objekata, temeljenog na paradigmi "praćenje-putem-detekcije". Istražene su i uspoređene različite strategije praćenja, od jednostavnih metoda temeljenih na Kalmanovom filtru do naprednijih pristupa koji koriste značajke izgleda objekata za ponovnu identifikaciju. Poseban naglasak stavljen je na rješavanje problema uzrokovanih gibanjem samog vozila. U tu svrhu, evaluiran je i integriran modul za kompenzaciju kretanja kamere, koji se pokazao ključnim za smanjenje broja pogrešnih zamjena identiteta. Nadalje, analizirane su metode asocijacije podataka koje uspješno riješavaju problem s privremenim okluzijama koristeći detekcije niske pouzdanosti. Zanimljiv zaključak proizašao je iz analize detektora, gdje se pokazalo da modeli pred-trenirani na općenitim skupovima podataka nude bolju generalizaciju od onih koji su dodatno fino podešeni na specifičnom skupu za praćenje. Rad zaključno predlaže optimiziranu arhitekturu koja postiže visoku robusnost i konzistentnost praćenja.

### Prijedlog 7: Fokus na izazove i rješenja u stvarnim scenarijima (bez otkrivanja rezultata)

**Naslov:** Prevladavanje izazova praćenja pješaka: Uloga kompenzacije gibanja i napredne asocijacije podataka

**Sažetak:**
Praćenje pješaka iz perspektive autonomnog vozila suočava se s brojnim izazovima, uključujući česte okluzije, nepredvidivo kretanje i konstantno gibanje kamere. Ovi faktori često dovode do fragmentacije putanja i pogrešne dodjele identiteta, što izravno ugrožava sposobnost sustava da predvidi namjere pješaka. Ovaj rad istražuje i kvantificira učinkovitost specifičnih tehnika za ublažavanje navedenih problema. Koristeći moderni detektor objekata kao temelj, provedena je komparativna studija različitih algoritama za praćenje. Istraživanje je identificiralo dva ključna poboljšanja. Prvo, eksplicitno modeliranje i kompenzacija gibanja kamere značajno poboljšavaju točnost predikcije stanja objekata, što rezultira drastičnim smanjenjem broja zamjena identiteta. Drugo, primjena dvostupanjske strategije asocijacije, koja zadržava i povezuje detekcije niske pouzdanosti, pokazala se iznimno učinkovitom za održavanje kontinuiteta praćenja tijekom okluzija. Kombinacija ovih pristupa rezultira sustavom koji pruža vremenski konzistentne tragove, što je preduvjet za pouzdanu procjenu rizika i planiranje putanje u autonomnoj vožnji.

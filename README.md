# Das Größte von n: wie sich ein Maximum verteilt – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-maximum-von-n-demo.streamlit.app/)**

Einstieg der Linie **„Extremwerte erklärt“** in der „Konzepte“-Reihe der Website „Sebastian Hanisch – Operations Research und Machine Learning“ (Bereich Statistik, Feld „Seltenes und Extremes“).
Es gibt keinen Vorgänger; die Kante ist ein **Kontrast**: Die Annahme, die hier fällt, steht nicht in einem früheren Stück, sondern in der Alltagsvorstellung „das Größte von n Beobachtungen
verhält sich wie eine Beobachtung, nur eben groß“. Gemessen: Es hat eine **eigene Verteilung F(x)ⁿ**, die mit n wandert (Exponential: Verschiebung um ln n bei Skala 1), sich staucht (Gleich) oder
wächst (Pareto-Verteilung). Gemessen: Bei n = 365 liegt der Abstand zum Gumbel-Gesetz für die Normalverteilung (Normierung B) 28,7-mal höher als für das Exponential; als Erklärung gilt die Randform der Einzelverteilung (Theorie, nicht isoliert gemessen).

Die Seite hat vier Experimente (das Größte von n simuliert gegen F(x)ⁿ, der exakt gerechnete Abstand zum Grenzgesetz, Skalieren und Typisieren, der Pegel der Normalverteilung) und eine Tabelle
„Was bleibt offen?“. **Alle Einzelverteilungen sind ausgedacht**; in den gelesenen Quellen gibt es keine Messreihe aus Logistik, Lager oder Hafen, die hier nachgebildet würde. Gezählt werden
Beobachtungen je Block (**n**) und simulierte Blöcke (**R**); was die Zahlen für einen echten Betrieb heißen, wird nicht behauptet.

## Warum dieses Problem

Wer das Größte von n Werten betrachtet (das Jahresmaximum aus 365 Tageswerten, den höchsten von 20 Würfen), rechnet oft mit einer der drei klassischen Grenzformen (Gumbel-, Weibull- oder Fréchet-Typ).
**Kernfrage:** Wie ist das Größte von n unabhängigen Werten verteilt, wie gut ist das Grenzgesetz bei endlichem n, und was macht eine schlechte Näherung mit dem Pegel, den man ablesen will?

## Modell

n unabhängige, gleich verteilte Beobachtungen X₁ … Xₙ mit stetiger Verteilungsfunktion F (Würfel: diskret), ihr Größtes Mₙ = max(X₁ … Xₙ) heißt **Block-Maximum**. Exakt gilt **P(Mₙ ≤ x) = F(x)ⁿ**.
Einzelverteilungen: Exponential(1), Gleichverteilung auf (0, 1), **Pareto-Verteilung** F(x) = 1 − x^(−α) für x ≥ 1 mit α = 1,5; 2 und 4, Normalverteilung(0, 1), dazu die Standard-Fréchet-Verteilung
exp(−1/x) (dort ist F^n selbst wieder Fréchet) und der Würfel. **Konventionen:** ξ > 0 Fréchet-Typ, ξ = 0 Gumbel-Typ, ξ < 0 **Weibull-Typ**; „Weibull-Typ“ ist nicht die Weibull-Verteilung der
Ausfallzeiten. `scipy.stats.genextreme` hat c = −ξ, `scipy.stats.genpareto` hat c = ξ (beides nur in den Tests, Gegenprobe mit c = +ξ liefert einen anderen Wert). **Pegel** heißt der Wert zu einer
Überschreitungswahrscheinlichkeit p je Block; „Wiederkehrperiode“ kommt hier nicht vor. Der Extremwertindex ξ wird nie mit dem Extremalindex θ vermischt (θ kommt nicht vor).
**Normierung** (gehört in jeden Satz): Exponential Verschiebung ln n, Skala 1; Gleich Skalierung n; Pareto-Verteilung Skalierung n^(1/α); Normal **B** (b = Φ⁻¹(1 − 1/n), a = 1/(n·φ(b))) oder **A**
(Lehrbuch-Konstanten c = (2 ln n)^(−1/2), d = √(2 ln n) − (ln ln n + ln 4π)/(2√(2 ln n))). **Abstand** ist der Kolmogorov-Abstand sup |Fⁿ(normiert) − Grenzgesetz|. Die Formeln stehen im Expander
„📐 Mathematische Formulierung“ der App.

## Methodik

* **Exakt:** F(x)ⁿ, Umkehrungen, Mittelwerte (Exponential Hₙ als harmonische Summe, Gleich n/(n + 1), Pareto-Verteilung über Gammafunktionen, Normal per Quadratur), Pegel aus Φⁿ = 1 − p. Rechnung nur mit
  numpy und der Standardbibliothek (`math.erfc`, `statistics.NormalDist.inv_cdf`); **scipy steht nur in den Tests und im Werkzeug `tools/mxn_make_reference.py`**, nie in der App.
* **Simulation:** Zufall aus SplitMix64 (`mxn_rng.py`, wörtliche Kopie der Vorlage des Portfolios). Ein Block besteht aus n gleichverteilten Zahlen, das Größte ist die Umkehrung des kleinsten
  v = 1 − u (die Umkehrung ist monoton): eine Umkehrung je Block, kein Box-Muller. Die Probe „n Umkehrungen, dann Maximum“ steht als Test. Obergrenze **n·R ≤ 2.000.000** Zufallszahlen.
* **Abstand zum Grenzgesetz:** exakt gerechnet, nicht simuliert (Gitter mit 200.001 Punkten, Auflösung des Supremums rund 10⁻⁸); für Exponential, Gleich und Pareto-Verteilung als
  sup_u |e^(−u) − (1 − u/n)ⁿ| (Substitution u = e^(−x) bzw. −x bzw. x^(−α)).
* **Eingefroren** (`mxn_reference.json`, Erzeuger und Prüfung `tools/mxn_make_reference.py --check`): Abstandstabellen und Kurve, kleinstes n unter einer Marke, bestes Gumbel und beste GEV je
  eingefrorenem n (für die Anpassungstabelle Differential Evolution mit zwei Zufalls-Seeds, danach Nelder-Mead; die Schwellen mit bestem Gumbel nur mit Nelder-Mead, in der Prüfung unabhängig bestätigt), Simulationsbänder.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen sind eigene Nachrechnung in diesem Modell (Voraussetzungen: unabhängig, gleich verteilt, die genannte Einzelverteilung und Normierung). Zähleinheit n bzw. R. Jede Zahl steht mit ihrem
Test in der Tabelle „Zahl → Test“ weiter unten.

1. **F(x)ⁿ gegen Simulation.** n = 20, R = 20.000 Blöcke, 100 Zufalls-Seeds je Verteilung (Exponential, Gleich, Pareto-Verteilung α = 2, Normal): mittlerer Abstand der Simulation zu F(x)ⁿ
   0,00602 / 0,00606 / 0,00613 / 0,00638 (Erwartung des Rauschens 0,8687/√R = 0,00614), größter Wert 0,00974 / 0,01597 / 0,01316 / 0,01148. Über der DKW-Schranke 0,0115 (99 %) liegen
   0, 1, 2 und 0 von je 100 Läufen (3 von 400), über 1,2 × Schranke = 0,0138 liegt 1 von 400 (Gleich, 0,01597). Die Vormessung hatte 4 von 400 über der Schranke und 0 über 1,2 × Schranke.
   Pareto-Verteilung mit α = 1,5 und α = 4 (n = 20, R = 20.000, 30 Zufalls-Seeds): mittlerer Abstand 0,00588 und 0,00643, über der Schranke 1 und 0 von 30 Läufen, über 1,2 × Schranke 0 von 30.
2. **Bei gleichem Zufalls-Seed ist der Abstand der Simulation zu F(x)ⁿ für alle stetigen Verteilungen derselbe Wert**, weil die Umkehrung monoton ist und dieselben Zufallszahlen dieselbe
   Rangordnung der Block-Maxima ergeben. Die Anzeige „unter der Schranke“ prüft also den Zufallsstrom und die Umkehrung, nicht die Verteilung.
3. **Exponential, Gleich und Pareto-Verteilung haben denselben Abstand zum Grenzgesetz** (Standardnormierung, exakt, für jedes α): n·Abstand = 0,2900 (n = 5), 0,2753 (n = 20), 0,2709 (n = 365) und
   0,27067 (n = 10⁶), Grenzwert 2/e² = 0,27067. n = 1 gibt e⁻¹ = 0,3679, n = 2 gibt 0,16190256.
4. **Normal gegen Exponential.** n = 365: Exponential (Standardnormierung) 0,00074, Normal (Normierung B) 0,0213, **Faktor 28,7**; mit den Lehrbuch-Konstanten (Normierung A) 0,0524, Faktor 70,6 gegen
   dasselbe Exponential; setzt man bei beiden das **beste Gumbel** mit freiem a und b ein (0,000140 und 0,01144), ist es der Faktor 81,5. Die drei Faktoren stellen verschiedene Normierungen gegeneinander
   und sind nicht austauschbar. Als Ursache gilt die Randform der Einzelverteilung (Theorie, nach Hall 1979 nur aus einer Zusammenfassung); isoliert gemessen ist sie hier nicht, denn kein Experiment ändert bei gleichem Rand die Mitte oder umgekehrt.
5. **Kleinstes n für einen Abstand unter 0,01** (dauerhaft): Exponential, Gleich, Pareto-Verteilung in Standardnormierung **n ≥ 28**, mit bestem Gumbel n ≥ 6; Normal in Normierung B **n ≥ 2,86·10⁵**, mit
   bestem Gumbel n ≥ 897; Normierung A erreicht 0,01 bis n = 10³⁰ nicht. Unter 0,05 liegt Normierung A erstmals bei n = 4, dauerhaft ab n = 622.
6. **Normierung A ist nicht monoton:** der Abstand steigt von n = 4 (0,0487) bis n = 17 (0,0657) und fällt danach (13 Anstiege im Gitter n = 2 bis 3.000). Normierung B und die Standardnormierung des
   Exponentials fallen auf den geprüften Gittern streng.
7. **Pegel der Normalverteilung** (exakt, Einheit Standardabweichung der Einzelwerte, p = Überschreitungswahrscheinlichkeit je Block): n = 365, p = 0,01 exakt 4,033; Gumbel A 4,147 (+0,114), Gumbel B
   4,272 (+0,239); die wahre Überschreitungswahrscheinlichkeit am Gumbel-B-Wert ist 0,0035 statt 0,01. n = 24, p = 0,001: Gumbel B +1,028. Bei p = 0,01 liegt Gumbel A zwischen +0,059 und +0,274 und
   Gumbel B zwischen +0,131 und +0,543 (n = 24, 365, 8.760). Die beste GEV mit freiem ξ (ξ̂ = −0,086) liegt bei n = 365, p = 0,01 um −0,020 daneben. **Die Langsamkeit betrifft vor allem „ξ = 0 festhalten“,
   weniger „eine GEV anpassen“:** die beste GEV hat schon bei n = 2 einen Abstand von 0,00365 (ξ̂ = −0,242) und bei n = 365 von 0,00088. In den Einstellungen der App (n bis 8.760, p = 0,1 bis 0,0001) liegt die beste GEV näher am exakten Pegel als die drei Gumbel-Verfahren, bei p = 0,01 und n ≥ 20 um höchstens 0,06 daneben, im Tail (p = 0,0001) um 0,075 bis 0,375 (n ≥ 20). Das gilt für die beste Anpassung an die exakte Verteilung, nicht für Schätzungen aus Daten.
8. **Wie viele Blöcke braucht man, um den Abstand zu sehen?** Das Rauschen liegt bei 1,628/√R (Kolmogorov, 99 %), der Abstand erreicht diese Schranke bei R ≈ (1,628/Abstand)²: bei n = 365 für Exponential
   rund 4,8 Millionen Blöcke, für Normal (Normierung B) rund 5.800. „Sichtbar“ ist das nur im Mittel: bei diesem R lag der Abstand in einer eigenen Probe (unabhängiger Zufallsstrom, nicht im Test) in 788 von 1.000 Läufen (Normal) bzw. 31 von 40 Läufen (Exponential) über der Schranke, bei doppeltem R in allen Läufen. Gemessen bei n = 365, R = 20.000, 30 Zufalls-Seeds: Exponential mittlerer Abstand zum Gumbel-Gesetz 0,00580 (Rauschen), über der Schranke
   in 0 von 30 Läufen; Normal 0,02299, über der Schranke in 30 von 30. Der Abstand wird deshalb gerechnet, nicht simuliert.
9. **Exponential-Momente:** E[Mₙ] = Hₙ = 3,59774 (n = 20, Bruch 55.835.135/15.519.504) gegen ln n + γ = 3,57295; Var[Mₙ] = Σ 1/k² = 1,59616 gegen π²/6 = 1,64493. Würfel: E[M₂] = 161/36 = 4,47
   gegen 3,5 für einen Wurf. Das Größte wächst beim Exponential nur wie ln n.
10. **Bootstrap (Verweis):** Anteil der Ziehungen, die die größte Beobachtung enthalten (bei n verschiedenen Werten, 1 − (1 − 1/n)ⁿ): 19/27 (n = 3) und 175/256 (n = 4), durch Aufzählung aller nⁿ Ziehungen; n = 50: 0,6358.

## Befunde und Korrekturen gegenüber dem Plan

* **Normierung A steigt bis n = 17, nicht bis n = 16.** Der Plan nannte „von n = 4 bis n = 16“; die 13 Anstiege sind die Schritte n → n + 1 für n = 4 bis 16, der größte Wert vor dem Fallen liegt bei n = 17.
* **Simulationsbänder neu gemessen** (SplitMix64, anderer Zufalls-Seed als die Vormessung): über der DKW-Schranke 3 von 400 Läufen statt 4 von 400; über 1,2 × Schranke **1 von 400** statt 0 von 400
  (Gleich, n = 20, Abstand 0,01597). Das ist mit der Wahrscheinlichkeit 0,1 % je Lauf vereinbar (erwartet 0,4 von 400), die Regel „unter 1,2 × Schranke“ hat also eine Fehlalarmrate von etwa 1 von 1.000
  Läufen und gilt in den Tests nur für feste Zufalls-Seeds.
* **Exponential-Momente, n = 20:** in 5 von 40 Läufen (12,5 %) liegt |z| des Mittels über 2 (erwartet 4,6 %), und das Mittel der 40 z-Werte liegt 2,75 Standardfehler unter 0. Eine Kontrolle mit 300 anderen
  Zufalls-Seeds (R = 20.000, n = 20) ergab Mittel −0,09 und Standardabweichung 0,95 der z-Werte (Anteil |z| > 2 von 4,7 %): das ist mit Zufall vereinbar, ein Fehler des Zufallsstroms zeigt sich nicht (die Kontrolle schließt kleine Verzerrungen nicht aus; gemessen wurden n = 20 und n = 365, bei n = 365 liegt das Mittel der 40 z-Werte bei +0,12 Standardfehlern, und hier steht das auffälligere n = 20; Messung dieser README, siehe Tabelle).
* **Gumbel A bei n = 365, p = 0,01 ist 4,147** (4,147488), nicht 4,148: die Rundung von 4,1475 im Plan führte auf die falsche letzte Stelle.
* **Das Mittel der Pareto-Verteilung mit α ≤ 2 ist kein Maß der Simulation.** Mittel und Median zeigen die Seite beide; die Varianz ist unendlich, ein Standardfehler existiert nicht.
* **Alle anderen Zahlen des Plans** (Tabelle der Abstände, Schwellen-n, Pegel, Faktoren 28,7 / 70,6 / 81,5, bestes GEV ab n = 2 unter 0,004) wurden im Bau neu gerechnet und stimmen bis zur letzten
  genannten Stelle mit dem Plan überein.
* **Nach einer unabhängigen Prüfung (2026-11-20) eingeschränkt:** „der Rand entscheidet“ ist gemessen nur als Vergleich Normal gegen Exponential (Ursache Randform: Theorie); „Gumbel A 4,148“ in der Tabelle war ein Tippfehler (4,147);
  „B ist die genaue Normierung“ gilt nicht (bei n = 2 bis 6 liegt A näher, B ist ab n ≈ 7 besser, beide liegen weiter vom Grenzgesetz als das beste Gumbel); „langsamste Annäherung der vier“ gilt für wachsendes n, im Abstand bei Normierung B
  erst ab n = 4 und bei A ab n = 6; „sichtbar“ bei R ≈ (1,628/Abstand)² gilt nur im Mittel (vier von fünf Läufen); das Verdikt nennt bei großem n (zum Beispiel 10⁶) keine Stellen mehr, die die Rechnung nicht deckt.

## Ehrliche Grenzen

* Gilt für **unabhängige, gleich verteilte** Beobachtungen und die genannten Einzelverteilungen; abhängige Beobachtungen und gemeinsame Extreme mehrerer Größen sind nicht gemessen.
* **Keine Schätzung aus Daten:** F ist bekannt, F(x)ⁿ exakt; die Anpassung einer GEV an gegebene Block-Maxima ist nicht Teil dieser Seite. „Bestes Gumbel“ und „beste GEV“ sind die besten Anpassungen
  an die **exakte** Verteilung (obere Schranken des besten Abstands), keine Schätzer.
* „Strikt fallend“ gilt nur für Exponential/Gleich/Pareto-Verteilung (n = 1 bis 2.000) und Normierung B (Gitter log₁₀ n = 0,7 bis 12 in Schritten 0,1) und ist kein Beweis; für Normierung A gilt es nicht.
* Die Schwellen-n der Normalverteilung haben die Auflösung des Gitters (rund 10⁻⁸ im Abstand, am Schwellenwert 0,01 (Normal, Normierung B) höchstens einige Einheiten in n, bei 2,86·10⁵ ohne Folge für die genannten drei Stellen).
* Zufallsergebnisse sind **Bänder, kein Test auf das letzte Bit** (`log` und `erfc` können sich je Plattform im letzten Bit unterscheiden); die Bänder sind unter Windows gemessen.
* **Rate der Annäherung:** ob die Rate 1/ln n oder (ln ln n)²/ln n ist, ist numerisch nicht trennbar. Bei Normierung B ist Abstand·ln n über n = 20 bis 10.000 flach (Tabelle „Was bleibt offen?“), was zu 1/ln n passt; bei Normierung A steigt es langsam weiter (n = 365 bis 10³⁰). Die Aussage von Hall (1979) wird nach einer Zusammenfassung zitiert, nicht nach dem Original; ob (ln ln n)²/ln n nur zur Normierung A gehört, ist nicht geprüft.
* Keine Aussage über den Nutzen einer Schätzung in Logistik, Lager oder Hafen; alle Verteilungen sind ausgedacht.

## 🚧 Bewusst nicht enthalten / nicht gemessen

* **Gleitende Blöcke**, abhängige Beobachtungen, jede Aussage über echte Daten: nicht Teil dieser Seite.
* **Linux-Lauf der Bänder** (Umkehrfunktionen im letzten Bit) und **Laufzeit in Streamlit Cloud**: nicht gemessen (nur Windows, lokal).
* **Pareto-Verteilung mit α ≠ 1,5; 2; 4:** nicht simuliert; der exakte Abstand ist für jedes α derselbe (Substitution).
* **Normal für n > 10¹²** und Normierung A jenseits 10³⁰: nicht gerechnet.
* **Grundlagen-Eintrag** „Maximum von n unabhängigen Werten“ und Erklärseite: nicht gebaut (Website-Einbindung steht aus); der Expander „💬 In einfachen Worten“ fehlt deshalb.
* **Echter Browser:** nicht geprüft; die Oberfläche ist nur mit AppTest und den Layout-Tests geprüft (Abschnitt „Verifikation“).

## Quellen (Stufe)

* Coles, S. (2001), *An Introduction to Statistical Modeling of Extreme Values*, Springer, Gleichung 3.1, Beispiele 3.1 bis 3.3, S. 45 bis 52: **gelesen** (W), anhand einer Kopie von einer Mitarbeiter-Seite
  mit unklaren Weitergaberechten; hier nur Befunde und Fundstellen, nichts wörtlich (Test `tests/test_literatur_maximum.py`).
* Leadbetter, Lindgren, Rootzén (1983), *Extremes and Related Properties of Random Sequences and Processes*, Kapitel 1.4, 1.7, 2.4: **gelesen** (W), gleiche Einschränkung.
* Fisher, Tippett (1928): **nur Abstract** (A), Original nicht gelesen. Hall (1979): **nur nach einer Zusammenfassung**, ungeprüft. Gnedenko (1943), Jenkinson (1955): nicht gelesen, nicht verwendet.
* **Keine Neuheit:** alles Lehrbuch; neu ist nur die Vorführbarkeit mit Zähleinheit, die exakte Tabelle der Schwellen-n und die Rechnung, wie viele Blöcke man braucht, um den Abstand zu sehen.
  Befunde ohne Quelle sind eigene Nachrechnung in diesem Modell.

## Zahl → Test

Die Zahlen der Befunde stehen in `tests/test_claims.py` (Zeichenkette der README gegen die Rechnung bzw. die eingefrorene Messung) und, wo angegeben, in einem zweiten Test mit anderem Rechenweg; Rechenzeiten, Testzahlen und Konstanten wie 1,628 sind dort nicht geprüft.
„Band“ heißt: Zufallsergebnis, geprüft mit Toleranz bzw. als Zählung; „eingefroren“: Messung dieser README aus `mxn_reference.json`, im Test mit der Datei verglichen und an kleinen Stichproben neu gemessen.

| Zahl | Art | Test |
|---|---|---|
| 0,00602 / 0,00606 / 0,00613 / 0,00638, größter Wert, 3 von 400, 1 von 400 (n = 20, R = 20.000, 100 Zufalls-Seeds) | eingefroren (Band), neu gemessen als Band bei 3 Zufalls-Seeds | `test_claims_baender`, `test_oracle_maximum::test_simulation_ks_unter_1_2_dkw_fuer_vier_verteilungen` |
| Pareto-Verteilung α = 1,5 und 4 (0,00588; 0,00643; 1 und 0 von 30) | eingefroren (Band) | `test_claims_baender` |
| gleicher Abstand für alle stetigen Verteilungen bei gleichem Zufalls-Seed | exakt (Gleichheit auf 1e-9) | `test_properties::test_gleicher_seed_gleiche_ks_fuer_alle_stetigen_verteilungen` |
| n·Abstand 0,2900 / 0,2753 / 0,2709 / 0,27067, e⁻¹, 0,16190256 | exakt (Gitter, drei unabhängige Gitter bis 1e-7) | `test_claims_abstaende`, `test_oracle_maximum::test_abstand_drei_verteilungen_gleich_auf_1e7` |
| 0,00074; 0,0213; 0,0524; Faktoren 28,7 / 70,6 / 81,5; 0,000140; 0,01144 | exakt bzw. eingefroren (beste Gumbel) | `test_claims_abstaende`, `test_oracle_maximum::test_normal_abstand_gegen_scipy_konstanten` |
| Schwellen 28, 6, 2,86·10⁵, 897, 622, 4, Normierung A nie unter 0,01 bis 10³⁰ | eingefroren, neu gerechnet | `test_claims_schwellen`, `test_reference::test_neu_rechnen_der_schnellen_teile_stimmt_mit_der_datei_ueberein` |
| A steigt von n = 4 (0,0487) bis n = 17 (0,0657), 13 Anstiege | exakt (Gitter 20.001) | `test_claims_schwellen`, `test_handbeispiele::test_normierung_a_ist_nicht_monoton_n4_gegen_n16` |
| Pegel 4,033 / 4,147 / 4,272, +0,114 / +0,239, 0,0035, +1,028, −0,020, ξ̂ −0,086, 0,00365, 0,00088 | exakt bzw. eingefroren | `test_claims_pegel`, `test_oracle_maximum::test_pegel_exakt_gegen_scipy_isf_und_bisektion` |
| 4,8 Millionen Blöcke, 5.800 (gerundet aus 5.823), 0,00580 / 0,02299, 0 von 30 / 30 von 30 | exakt bzw. eingefroren (Band) | `test_claims_sichtbarkeit` |
| Beste GEV näher am Pegel als die drei Gumbel-Verfahren, höchstens 0,06 bei p = 0,01 und n ≥ 20, 0,075 bis 0,375 bei p = 0,0001 und n ≥ 20 | eingefroren (beste Anpassungen), alle n und p der App | `test_claims_pegel` |
| 788 von 1.000 und 31 von 40 Läufen über der Schranke bei R ≈ (1,628/Abstand)², bei doppeltem R alle | Messung dieser Überarbeitung (numpy-Zufallsstrom), nicht im Test | – |
| 3,59774, 55.835.135/15.519.504, 3,57295, 1,59616, 1,64493, 161/36 = 4,47 | exakt (Brüche, Quadratur) | `test_claims_momente`, `test_handbeispiele::test_exponential_momente_renyi_gegen_quadratur` |
| 12,5 %, 2,75 Standardfehler, +0,12 Standardfehler bei n = 365, Kontrolle mit 300 Zufalls-Seeds | eingefroren (Band) | `test_claims_baender` |
| 19/27, 175/256, 0,6358 | exakt (Aufzählung) | `test_handbeispiele::test_bootstrap_anteil_gleiches_maximum_durch_aufzaehlung` |
| Rechenzeiten, Laufzeit der Suite | Messung dieser README, nicht im Test | – |

## Verifikation

* **Handbeispiele und Orakel:** Würfel (alle Ausgänge), Gleich, Exponential (u* = 1,59362, 0,16190256), Normal n = 2 und 3 in geschlossener Form, Pegel mit `math.erfc`-Bisektion, F(x)ⁿ gegen
  `scipy.stats`, drei unabhängige Gitter, zwei Gitterauflösungen, GEV-Zeugen (`tests/test_handbeispiele.py`, `tests/test_oracle_maximum.py`, `tests/test_literatur_maximum.py`).
* **Eigenschaften und Randfälle:** F(x)ⁿ wächst in x und fällt in n, F^(n·m) = (Fⁿ)^m, Lage-Skalen-Invarianz, Abstand hängt nicht an α, gleicher Zufalls-Seed gibt gleiche Ausgabe; n = 1 und 2, n = 10¹²,
  R = 1 und 0, Pareto-Verteilung und Fréchet ohne Mittelwert, p = 0 und 1 (`tests/test_properties.py`, `tests/test_edge_cases.py`).
* **Zufall:** die kopierte Vorlage ist wörtlich geprüft (`tests/test_copies.py`); die numpy-Messhilfe für die großen Bänder ist bitgleich zur Vorlage (`tests/test_rng_vector.py`).
* **Eingefrorene Tabellen:** `tools/mxn_make_reference.py --check` rechnet n ≤ 365, die ganzzahligen Schwellen und die Normal-B-Schwellen neu; `--write --lang --bands` erzeugt alles neu.
* **Verdikte, Presets, Oberfläche:** `tests/test_verdicts.py`, `tests/test_presets.py` (je Preset die erwartete Verdikt-Zeile), `tests/test_app.py` (jeder Knopf, jeder Regler an den Grenzen und in jeder
  Auswahl), `tests/test_layout.py`, `tests/test_footer.py`, `tests/test_runtime_imports.py`.
* **Testzahl und Laufzeit:** siehe „Tests ausführen“.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `app.py` | Streamlit-App (Kopf, Presets, vier Experimente, Kernfragen, Grenzen, Formeln) |
| `mxn_constants.py` | Regler-Grenzen, Auswahlwerte, Farben, Presets |
| `mxn_distributions.py` | F(x)ⁿ, Umkehrungen, Mittelwerte, Normierungen, Grenzgesetze, GEV (numpy und Standardbibliothek) |
| `mxn_distance.py` | Kolmogorov-Abstand zum Grenzgesetz, Sichtbarkeit, DKW-Schranke, Schwellen-n |
| `mxn_scenario.py` | Block-Maxima aus der reinen Vorlage, KS-Statistik |
| `mxn_rng.py` | wörtliche Kopie der Zufalls-Vorlage SplitMix64 |
| `mxn_evaluation.py` | Auswertungen der Experimente und alle Verdikte (reine Funktionen) |
| `mxn_reference.py`, `mxn_reference.json` | eingefrorene Tabellen mit Quelle und Stand je Abschnitt |
| `mxn_presets.py`, `mxn_layout.py` | Permalink, Presets, Zufalls-Seed-Knopf, fester Kopf |
| `mxn_texts.py`, `mxn_visualization.py` | Texte und Zahlenformate, Plotly-Abbildungen |
| `tools/mxn_make_reference.py` | erzeugt und prüft `mxn_reference.json` (scipy, Differential Evolution, numpy-SplitMix64) |
| `tests/` | Handbeispiele, Orakel, Literatur, Eigenschaften, Randfälle, Verdikte, Presets, Claims, Oberfläche, Footer, Laufzeit-Importe |
| `requirements.txt`, `requirements-dev.txt` | Laufzeit (numpy, streamlit, plotly, pandas) bzw. plus pytest und scipy (nur Tests) |

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py --server.port 10083     # Port der lokalen Vorschau, frei wählbar
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Die Suite hat **324 Tests**: Handbeispiele 40, Orakel 127, Literatur 13, Eigenschaften 33, Randfälle 15, eingefrorene Tabellen 10, Zufall und Kopie 8, Verdikte 14, Presets 15, Claims 8,
Oberfläche 19, Layout 19, Footer 2, Laufzeit-Importe 1. Laufzeit seriell 450 s (Windows, ein Thread, Python 3.14, Rechner durch andere Prozesse belastet; eine frühere Messung auf freiem Rechner ergab 124 s; Messung dieser README, nicht im Test); mit `pytest -n 8` kürzer.

**Rechenzeiten** (Messung dieser README, bester von sechs Läufen, lokal unter Windows auf einem durch andere Prozesse belasteten Rechner, einzelne Läufe bis zum Fünffachen langsamer; nicht im Test; Streamlit Cloud ist langsamer und nicht gemessen): Simulation mit n·R = 400.000 Zufallszahlen (Standardfall) rund 0,3 s, mit der
Obergrenze n·R = 2.000.000 rund 1,5 s (Exponential) bis 1,7 s (Normal); der Abstand der Normalverteilung auf dem Gitter mit 200.001 Punkten rund 0,06 s je Normierung; ein ganzer Lauf der Seite im
AppTest rund 0,5 s. Die App zeigt bei einer Simulation ab etwa 0,5 s Rechenzeit einen Spinner mit Zeitangabe.

Gebaut mit Streamlit, Plotly und NumPy.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Extremwerte: Blöcke, Schwelle, Pegel](https://sebastianhanisch.net/konzepte-extremwert.html).

"""Texte und Zahlenformate der App (Dezimalkomma, Tausenderpunkt). Zahlen in festen Texten stehen nur, wenn sie aus der Rechnung oder der eingefrorenen Referenz gebildet werden."""

import math


def de(x: float, nd: int = 2) -> str:
    """Dezimalzahl mit Komma: de(0.0213, 4) == '0,0213'."""
    return f"{x:.{nd}f}".replace(".", ",")


def de_signed(x: float, nd: int = 2) -> str:
    """Mit Vorzeichen und echtem Minuszeichen: de_signed(-0.02, 3) == '−0,020'."""
    return f"{x:+.{nd}f}".replace(".", ",").replace("-", "−")


def thousands(x: float) -> str:
    """Ganzzahl mit Tausenderpunkt: 20000 -> '20.000'."""
    return f"{int(round(x)):,}".replace(",", ".")


def p_text(p: float) -> str:
    """Überschreitungswahrscheinlichkeit ohne überflüssige Nullen: 0.01 -> '0,01', 0.0001 -> '0,0001'."""
    return f"{p:.6f}".rstrip("0").rstrip(".").replace(".", ",")


def sci(x: float, nd: int = 2) -> str:
    """Zahl mit Komma; ab 1000 als Mantisse·Zehnerpotenz (2,86·10⁵); None als „nicht erreicht“."""
    if x is None:
        return "nicht erreicht"
    if x < 1000:
        return de(x, nd) if abs(x - round(x)) > 1e-9 else str(int(round(x)))
    e = int(math.floor(math.log10(x)))
    m = x / 10 ** e
    return f"{de(m, nd)}·10{_sup(e)}"


def sig3(x: float) -> str:
    """Drei gültige Stellen mit Komma; ab 1000 als Mantisse·Zehnerpotenz: 18,8; 897; 2,86·10⁵."""
    if x is None:
        return "nicht erreicht"
    if x < 1000:
        return f"{x:.3g}".replace(".", ",")
    return sci(x, 2)


def _sup(e: int) -> str:
    return str(e).translate(str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻"))


def small_sci(x: float, nd: int = 1) -> str:
    """Kleine positive Zahl als Mantisse·Zehnerpotenz mit Komma: small_sci(2.7e-7) == '2,7·10⁻⁷'."""
    e = int(math.floor(math.log10(x)))
    m = x / 10 ** e
    if round(m, nd) >= 10:
        m, e = m / 10, e + 1
    return f"{de(m, nd)}·10{_sup(e)}"


def dist_text(d: float, nd: int = 5) -> str:
    """Abstand mit `nd` Stellen; unter 10⁻⁴ als Zehnerpotenz mit zwei gültigen Stellen (sonst stünde dort 0,00000)."""
    return de(d, nd) if d >= 1e-4 else small_sci(d, 1)


def round_sig(x: float, k: int = 2) -> str:
    """Große Zahl auf k gültige Stellen gerundet (Tausenderpunkt); ab 10⁹ als Zehnerpotenz: 4.810.817 -> '4.800.000'."""
    if x >= 1e9:
        return sci(x, k - 1)
    return thousands(float(f"{x:.{k - 1}e}"))


def factor_text(f: float) -> str:
    """Faktor mit einer Nachkommastelle; ab 1000 auf drei gültige Stellen (Zehnerpotenz), weil die Stellen dahinter nicht gedeckt sind."""
    return de(f, 1) if f < 1000 else sci(f, 2)


def n_label(n: float) -> str:
    """n als Text: Ganzzahlen mit Tausenderpunkt, ab 10⁶ als Zehnerpotenz, wenn n eine ist."""
    if n >= 1_000_000:
        e = math.log10(n)
        return "10" + _sup(int(round(e))) if abs(e - round(e)) < 1e-9 else sci(float(n), 2)
    return thousands(n)


# ------------------------------------------------------------------------------------------------------------------------------------------
# Feste Texte der App
# ------------------------------------------------------------------------------------------------------------------------------------------
INTRO = """
Das **Größte von n unabhängigen Werten** hat eine eigene Verteilung, **F(x)ⁿ**: Sie wandert mit n, staucht sich oder wächst; wie gut ein Grenzgesetz passt, unterscheidet sich
stark zwischen den Einzelverteilungen (als Erklärung gilt der **Rand** der Einzelverteilung, gemessen ist der Vergleich der Verteilungen). Die Seite rechnet sie **exakt**, **simuliert** sie und vergleicht sie mit ihrem **Grenzgesetz** (Gumbel-, Weibull- oder
Fréchet-Typ). Gezählt wird in **n** (Beobachtungen je Block) und **R** (simulierte Blöcke); die Normierung steht in jedem Satz. Alle Verteilungen sind ausgedacht
und stehen für nichts Betriebliches. Die Seite ist der Einstieg der Linie „Extremwerte erklärt“.

Weiter unten: die **Kernfragen**, vier **🔬 Experimente**, der Abschnitt **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""

HOW_IT_WORKS = """
1. **Block bilden:** n unabhängige Beobachtungen X₁ … Xₙ aus derselben Verteilung F, notiert wird das Größte, Mₙ = max(X₁, …, Xₙ). Das ist ein **Block-Maximum**.
2. **Exakt rechnen:** Mₙ ≤ x gilt genau dann, wenn alle n Beobachtungen ≤ x sind; bei Unabhängigkeit ist das **F(x)ⁿ**. Damit ist die Verteilung des Größten bekannt, ohne zu simulieren.
3. **Normieren und vergleichen:** Verschiebt und streckt man Mₙ passend (Exponential: um ln n verschieben), läuft F(x)ⁿ auf ein **Grenzgesetz** zu. **Wie schnell, unterscheidet sich zwischen den Einzelverteilungen; als Erklärung gilt ihr Rand (Theorie, nicht isoliert gemessen).**
"""

ASSUMPTIONS = """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Die n Beobachtungen sind unabhängig und gleich verteilt** | F(x)ⁿ gilt dann nicht; hier ist alles unabhängig, abhängige Beobachtungen sind nicht gemessen. | nicht Teil dieser Seite |
| **Die Einzelverteilung ist bekannt und exakt** | Aus Daten ist F unbekannt; diese Seite schätzt nichts. Die Anpassung einer GEV an gegebene Block-Maxima ist nicht enthalten. | nicht Teil dieser Seite |
| **Vier ausgedachte Verteilungen, dazu Würfel und Standard-Fréchet** | Andere Ränder können anders konvergieren; gemessen sind nur diese. Die Pareto-Verteilung ist mit α = 1,5; 2 und 4 simuliert, ihr exakter Abstand ist für jedes α derselbe. | nicht Teil dieser Seite |
| **Ein Maximum je Block, kein Mehrdimensionales** | Gemeinsame Extreme mehrerer Größen werden nicht betrachtet. | nicht Teil dieser Seite |
| **Keine Aussage über einen Betrieb** | Die Zahlen gelten für die ausgedachten Verteilungen. Was sie für Logistik, Lager oder Hafen heißen, wird nicht behauptet. | nicht Teil dieser Seite |
"""

THRESHOLD_NOTE = (
    "Die Spalten „Standardnormierung“ und „Normierung B“ stellen zwei übliche Normierungen nebeneinander (Standardnormierung des Exponentials, Normierung B der Normalverteilung). Die Spalten „bestes Gumbel“ zeigen jeweils das beste Gumbel mit freiem a und b, das F(x)ⁿ näher "
    "kommt als die beiden Normierungen (obere Schranke des besten Abstands, Optimierung außerhalb der App). Bei Normierung A ist der Abstand nicht monoton: Er steigt zunächst; "
    "„dauerhaft“ meint, dass er ab diesem n unter der Marke bleibt."
)

TYPE_TABLE = [
    {"Einzelverteilung": "Exponential(1)", "Typ": "Gumbel-Typ", "ξ": "0", "Normierung (z)": "Mₙ − ln n", "Grenzgesetz": "exp(−e^−z)"},
    {"Einzelverteilung": "Gleichverteilung auf (0, 1)", "Typ": "Weibull-Typ", "ξ": "−1", "Normierung (z)": "n·(Mₙ − 1)", "Grenzgesetz": "e^z für z ≤ 0"},
    {"Einzelverteilung": "Pareto-Verteilung mit α", "Typ": "Fréchet-Typ", "ξ": "1/α", "Normierung (z)": "Mₙ / n^(1/α)", "Grenzgesetz": "exp(−z^−α) für z > 0"},
    {"Einzelverteilung": "Standard-Fréchet exp(−1/x)", "Typ": "Fréchet-Typ", "ξ": "1", "Normierung (z)": "Mₙ / n", "Grenzgesetz": "exp(−1/z), exakt für jedes n"},
    {"Einzelverteilung": "Normalverteilung(0, 1)", "Typ": "Gumbel-Typ", "ξ": "0", "Normierung (z)": "(Mₙ − bₙ)/aₙ (B) bzw. (Mₙ − dₙ)/cₙ (A)", "Grenzgesetz": "exp(−e^−z)"},
]


def open_questions(ref: dict) -> str:
    """Tabelle „Was bleibt offen?“ aus der eingefrorenen Referenz (Zahlen kommen aus der Datei, nicht aus dem Text)."""
    th = ref["thresholds"]
    dt = ref["distance_tables"]["normal"]
    prod = [d * math.log(n) for n, d in zip(dt["n"], dt["B"]) if 20 <= n <= 10000]
    return (
        "| Frage | Stand |\n|---|---|\n"
        f"| Bestes Gumbel (a, b frei) statt Normierung B | Abstand unter 0,01 ab n = {sig3(th['normal_best_gumbel']['0.01'])} statt n ≈ {sig3(th['normal_B']['0.01'])} (Normal) |\n"
        f"| Rate der Annäherung bei der Normalverteilung | Abstand·ln n liegt für n = 20 bis 10.000 zwischen {de(min(prod), 3)} und {de(max(prod), 3)} (Normierung B); ob die Rate 1/ln n oder "
        "(ln ln n)²/ln n ist, ist numerisch nicht zu trennen (flach heißt: passt zu 1/ln n; bei Normierung A steigt Abstand·ln n weiter); die Aussage von Hall (1979) wird nach einer Zusammenfassung zitiert, nicht nach dem Original, "
        "und ob (ln ln n)²/ln n nur zur Normierung A gehört, ist nicht geprüft. |\n"
        "| Schätzung aus Daten | nicht Teil dieser Seite: hier ist die Einzelverteilung bekannt und F(x)ⁿ exakt. |\n"
        "| Wiederkehrperiode | nicht Teil dieser Seite: hier nur das Block-Quantil zur Überschreitungswahrscheinlichkeit p je Block. |\n"
    )


FORMULAS_INTRO = """
**Lesehilfe.** *Gegeben:* n unabhängige, gleich verteilte Beobachtungen X₁, …, Xₙ mit stetiger Verteilungsfunktion F (beim Würfel diskret). *Gesucht:* die Verteilung ihres Größten
Mₙ = max(X₁, …, Xₙ), ihr Abstand zum Grenzgesetz und der Pegel zur Überschreitungswahrscheinlichkeit p je Block. Das ist das vollständige Modell dieser Seite: alle
Verteilungen, Bereiche und Normierungen stehen unten; geschätzt wird nichts.
"""

SYMBOLS = """
| Symbol | Bedeutung |
|---|---|
| n | Beobachtungen je Block (Zähleinheit; ganzzahlig, 1 bis 365 in der Simulation, bis 10¹² beim Abstand) |
| R | simulierte Blöcke (1.000 bis 50.000, begrenzt durch n·R ≤ 2.000.000) |
| F, Φ, φ | Verteilungsfunktion der Einzelwerte; Verteilungsfunktion und Dichte der Standardnormalverteilung |
| Mₙ | das Größte der n Beobachtungen eines Blocks (Block-Maximum) |
| ξ | Extremwertindex: ξ > 0 Fréchet-Typ, ξ = 0 Gumbel-Typ, ξ < 0 Weibull-Typ (nicht der Extremalindex θ) |
| μ, σ | Lage und Skala der Gumbel-Verteilung bzw. der GEV |
| bₙ, aₙ | Normierung B der Normalverteilung |
| dₙ, cₙ | Normierung A der Normalverteilung (Lehrbuch-Konstanten) |
| Δₙ | Kolmogorov-Abstand der exakten Verteilung des normierten Maximums zum Grenzgesetz |
| p | Überschreitungswahrscheinlichkeit je Block; x_p der Pegel mit P(Mₙ > x_p) = p |
"""

FORMULA_BLOCKS = [
    "**Exakte Verteilung.** Bei Unabhängigkeit ist Mₙ ≤ x genau dann, wenn alle n Beobachtungen ≤ x sind:",
    r"$$P(M_n \le x) = F(x)^n$$",
    "**Normierung und Grenzgesetz.** Exponential(1): Verschiebung um ln n bei Skala 1; gleich verteilt: Skalierung mit n; Pareto-Verteilung mit $F(x) = 1 - x^{-\\alpha}$: Skalierung mit $n^{1/\\alpha}$:",
    r"$$P(M_n - \ln n \le z) = \left(1 - \frac{e^{-z}}{n}\right)^n \longrightarrow \exp\left(-e^{-z}\right)$$",
    r"$$P\big(n(M_n - 1) \le z\big) = \left(1 + \frac{z}{n}\right)^n \longrightarrow e^{z} \quad (z \le 0)$$",
    r"$$P\left(\frac{M_n}{n^{1/\alpha}} \le z\right) = \left(1 - \frac{z^{-\alpha}}{n}\right)^n \longrightarrow \exp\left(-z^{-\alpha}\right) \quad (z > 0)$$",
    "**Normalverteilung, Normierung B und A:**",
    r"$$b_n = \Phi^{-1}\!\left(1 - \tfrac{1}{n}\right), \qquad a_n = \frac{1}{n\,\varphi(b_n)}$$",
    r"$$c_n = (2 \ln n)^{-1/2}, \qquad d_n = \sqrt{2 \ln n} - \frac{\ln \ln n + \ln 4\pi}{2\sqrt{2 \ln n}}$$",
    "**Abstand zum Grenzgesetz.** Kolmogorov-Abstand; für Exponential, Gleich und Pareto-Verteilung derselbe Wert (Substitution $u = e^{-x}$, $-x$ bzw. $x^{-\\alpha}$):",
    r"$$\Delta_n = \sup_{0 \le u \le n} \left| e^{-u} - \left(1 - \tfrac{u}{n}\right)^n \right|, \qquad n\,\Delta_n \to \frac{2}{e^2}$$",
    r"$$\Delta_n^{\text{Normal}} = \sup_z \left| \Phi(b_n + a_n z)^n - \exp\left(-e^{-z}\right) \right|$$",
    "**GEV und Pegel.** Verteilungsfunktion und Wert zur Überschreitungswahrscheinlichkeit p je Block, mit $y = -\\ln(1 - p)$ (für ξ → 0 der Gumbel-Fall):",
    r"$$G(x) = \exp\left\{-\left[1 + \xi\,\frac{x - \mu}{\sigma}\right]^{-1/\xi}\right\}, \qquad x_p = \mu - \frac{\sigma}{\xi}\left(1 - y^{-\xi}\right)$$",
    "Der exakte Pegel der Normalverteilung folgt aus $\\Phi(x_p)^n = 1 - p$.",
    "**Momente, Exponential(1)** (Rényi): Mittelwert und Varianz des Größten sind die harmonische Summe und die Summe der Kehrwerte der Quadrate:",
    r"$$\mathrm{E}[M_n] = H_n = \sum_{k=1}^{n} \frac{1}{k}, \qquad \mathrm{Var}[M_n] = \sum_{k=1}^{n} \frac{1}{k^2}$$",
]

FORMULAS_OUTRO = """
**Vorzeichen von ξ.** Auf dieser Seite und im Code gilt: ξ > 0 Fréchet-Typ, ξ = 0 Gumbel-Typ, ξ < 0 Weibull-Typ. `scipy.stats.genextreme` hat c = −ξ, `scipy.stats.genpareto` hat c = ξ;
beide stehen nur in den Tests als Referenz, nicht in der App. „Weibull-Typ“ ist nicht die Weibull-Verteilung der Ausfallzeiten. „Pareto-Verteilung“ meint die Verteilung mit
Dichte ∝ x^(−α−1) für x ≥ 1, nicht die Pareto-Front. **Pegel** heißt hier der Wert zu einer Überschreitungswahrscheinlichkeit p je Block (Block-Quantil), nicht die Wiederkehrperiode.

**Quellen.** Coles (2001), *An Introduction to Statistical Modeling of Extreme Values*, Kapitel 3, die ersten drei Beispiele (S. 45 bis 52; gelesen, hier nur die Befunde, nichts wörtlich); Leadbetter u. a. (1983), Kapitel 1 und 2 (gelesen);
Fisher und Tippett (1928) nur nach dem Abstract; Hall (1979) nur nach einer Zusammenfassung. Alles übrige ist eigene Nachrechnung in diesem Modell.
"""

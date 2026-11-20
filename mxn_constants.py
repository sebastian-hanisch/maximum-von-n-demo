"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets.

Rechenaufwand am Regler-Maximum (DEMO-PLAYBOOK 1c):
* Simulation (Experiment 1): n·R <= SIM_BUDGET gleichverteilte Zahlen aus der reinen Vorlage `mxn_rng.py`, O(n·R) in reinem Python
  (rund 1 µs je Zahl, also bis etwa 2 s; Streamlit Cloud ist langsamer, deshalb Spinner mit Zeitangabe).
* Abstand zum Grenzgesetz (Experiment 2): ein Gitter mit 200 001 Punkten je Normierung, O(Gitter) mit `math.erfc` (etwa 0,1 s).
* Übrige Experimente: O(Gitter) mit höchstens 2 001 Punkten.
"""

# --- Verteilungen (Schlüssel -> Anzeigename; α der Pareto-Verteilung steht in PARETO_ALPHA) -----------------------------------------------
DIST_LABELS = {
    "wuerfel": "Würfel (1 bis 6, diskret)",
    "exp": "Exponential(1)",
    "gleich": "Gleichverteilung auf (0, 1)",
    "pareto15": "Pareto-Verteilung, α = 1,5",
    "pareto2": "Pareto-Verteilung, α = 2",
    "pareto4": "Pareto-Verteilung, α = 4",
    "frechet": "Standard-Fréchet exp(−1/x)",
    "normal": "Normalverteilung(0, 1)",
}
PARETO_ALPHA = {"pareto15": 1.5, "pareto2": 2.0, "pareto4": 4.0}
DIST_KEYS = tuple(DIST_LABELS)
DEFAULT_DIST = "exp"

# --- Blockgröße n, simulierte Blöcke R, Zufalls-Seed --------------------------------------------------------------------------------------
N_MIN, N_MAX, DEFAULT_N = 1, 365, 20
R_MIN, R_MAX, DEFAULT_R, R_STEP = 1000, 50000, 20000, 1000
SEED_MAX, DEFAULT_SEED = 999999, 35
SIM_BUDGET = 2_000_000          # n·R höchstens so viele gleichverteilte Zahlen je Lauf (V10 F: rund 2 s)
DKW_ALPHA = 0.01                # Schranke der Simulation: DKW zum Niveau 99 %
KS_999 = 1.628                  # Kolmogorov-Quantil (99 %): Rauschen der empirischen Verteilungsfunktion 1,628/√R

# --- Experiment 2: n für den Abstand zum Grenzgesetz (exakt gerechnet) -----------------------------------------------------------------------
N_LIM_OPTIONS = (2, 3, 5, 10, 20, 28, 100, 365, 897, 1000, 10_000, 100_000, 286_000, 1_000_000, 10 ** 8, 10 ** 10, 10 ** 12)
DEFAULT_N_LIM = 365
NORM_LABELS = {"B": "Normierung B (b = Φ⁻¹(1 − 1/n), a = 1/(n·φ(b)))", "A": "Normierung A (Lehrbuch-Konstanten c, d)"}
DEFAULT_NORM = "B"
GRID_POINTS = 200_001           # Gitter für den Abstand der Normalverteilung (Auflösung des Supremums rund 1e-8)
GRID_X = (-4.0, 16.0)           # Gitterbereich der standardisierten Größe z
MARKS = (0.05, 0.02, 0.01, 0.005, 0.001)

# --- Experiment 4: Pegel der Normalverteilung ----------------------------------------------------------------------------------------------
N_PEGEL_OPTIONS = (2, 3, 5, 10, 20, 24, 100, 365, 897, 8760)   # nur eingefrorene n: die Optimierung von „bestes Gumbel/GEV“ ist live nicht zuverlässig
DEFAULT_N_PEGEL = 365
P_OPTIONS = (0.1, 0.01, 0.001, 0.0001)
DEFAULT_P = 0.01
METHOD_LABELS = {
    "gumbelA": "Gumbel, Normierung A (ξ = 0 festgehalten)",
    "gumbelB": "Gumbel, Normierung B (ξ = 0 festgehalten)",
    "gumbelbest": "bestes Gumbel (a, b frei, ξ = 0)",
    "gevbest": "beste GEV (ξ frei)",
}
DEFAULT_METHOD = "gumbelB"

# --- Farben ----------------------------------------------------------------------------------------------------------------------------------
COLOR_EXACT = "#2F6B65"
COLOR_LIMIT = "#c0504d"
COLOR_SIM = "rgba(76,120,168,0.85)"
COLOR_OTHER = "#7f7f7f"
COLOR_B = "#e17c05"

# --- Presets ---------------------------------------------------------------------------------------------------------------------------------
STANDARD_PRESET = "Exponential, n = 20"
PRESETS = {
    "Würfel: das Größte von 2": {"dist": "wuerfel", "n": 2, "R": 20000, "seed": 35, "n_lim": 2, "norm": "B", "n_pegel": 2, "p": 0.01, "method": "gumbelB"},
    "Exponential, n = 20": {"dist": "exp", "n": 20, "R": 20000, "seed": 35, "n_lim": 365, "norm": "B", "n_pegel": 365, "p": 0.01, "method": "gumbelB"},
    "Gleich, n = 20 (Weibull-Typ)": {"dist": "gleich", "n": 20, "R": 20000, "seed": 35, "n_lim": 365, "norm": "B", "n_pegel": 365, "p": 0.01, "method": "gumbelB"},
    "Pareto-Verteilung, n = 20 (Fréchet-Typ)": {"dist": "pareto2", "n": 20, "R": 20000, "seed": 35, "n_lim": 365, "norm": "B", "n_pegel": 365, "p": 0.01, "method": "gumbelB"},
    "Normal, n = 365: 28,7-mal größerer Abstand": {"dist": "normal", "n": 365, "R": 5000, "seed": 35, "n_lim": 365, "norm": "B", "n_pegel": 365, "p": 0.01, "method": "gumbelB"},
    "Normal gegen Exponential, gleiche n": {"dist": "exp", "n": 365, "R": 5000, "seed": 35, "n_lim": 365, "norm": "A", "n_pegel": 365, "p": 0.01, "method": "gevbest"},
}
PRESET_HELP = {
    "Würfel: das Größte von 2": "Zwei Würfe, alle 36 Ausgänge: P(Maximum ≤ k) = (k/6)². Im Mittel 161/36 = 4,47 gegen 3,5 für einen Wurf.",
    "Exponential, n = 20": "Standardfall: 20 Beobachtungen je Block, 20.000 simulierte Blöcke. Die Simulation liegt auf F(x)ⁿ, mit Zufalls-Seed 35 innerhalb der DKW-Schranke.",
    "Gleich, n = 20 (Weibull-Typ)": "Das Größte von 20 Gleichverteilten rückt an den oberen Rand 1 heran (Weibull-Typ, ξ = −1).",
    "Pareto-Verteilung, n = 20 (Fréchet-Typ)": "Pareto-Verteilung mit α = 2: das Größte wächst wie n^(1/2) (Fréchet-Typ, ξ = 1/2).",
    "Normal, n = 365: 28,7-mal größerer Abstand": "Normalverteilung, n = 365: der Abstand zum Gumbel-Gesetz ist mit 0,0213 (Normierung B) 28,7-mal so groß wie beim Exponential (0,00074).",
    "Normal gegen Exponential, gleiche n": "Exponential mit n = 365 und Normierung A der Normalverteilung: Faktor 70,6 beim Abstand; beim Pegel die GEV mit freiem ξ.",
}

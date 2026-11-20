"""Kennzahlen, Verdikte (reine Funktionen) und Auswertungen der vier Experimente. Nur numpy und Standardbibliothek."""

import math
from dataclasses import dataclass

import numpy as np

import mxn_constants as C
import mxn_distance as X
import mxn_distributions as D
import mxn_reference as REF
import mxn_scenario as S
from mxn_texts import de, de_signed, dist_text, factor_text, n_label, p_text, round_sig, sig3, thousands


@dataclass(frozen=True)
class Settings:
    dist: str = C.DEFAULT_DIST
    n: int = C.DEFAULT_N
    R: int = C.DEFAULT_R
    seed: int = C.DEFAULT_SEED
    n_lim: int = C.DEFAULT_N_LIM
    norm: str = C.DEFAULT_NORM
    n_pegel: int = C.DEFAULT_N_PEGEL
    p: float = C.DEFAULT_P
    method: str = C.DEFAULT_METHOD


def label(key: str) -> str:
    return C.DIST_LABELS[key]


# ------------------------------------------------------------------------------------------------------------------------------------------
# Experiment 1: das Größte von n, Simulation gegen F^n
# ------------------------------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class SimAnalysis:
    sim: S.Simulation
    ks: float                  # Kolmogorov-Abstand der Simulation zu F^n
    eps: float                 # DKW-Schranke (99 %) für R Blöcke
    mean_sim: float
    mean_exact: float | None   # None: Mittelwert existiert nicht
    se: float | None           # Standardfehler des Mittels (nur wenn die Varianz existiert)
    median_sim: float
    median_exact: float
    noise: float               # Rauschband 1,628/√R


def variance_exists(key: str) -> bool:
    return not (key == "frechet" or (D.is_pareto(key) and D.pareto_alpha(key) <= 2.0))


def ks_to_exact(sim: S.Simulation) -> float:
    key, n = sim.key, sim.n
    if key == "wuerfel":
        right = lambda k: D.cdf_max("wuerfel", k, n)
        left = lambda k: D.cdf_max("wuerfel", k - 1.0, n)
        return S.ks_discrete(sim.values, right, left)
    return S.ks_statistic(sim.values, lambda x: D.cdf_max(key, x, n))


def analyse_sim(sim: S.Simulation) -> SimAnalysis:
    vals = sim.values
    se = float(np.std(vals, ddof=1) / math.sqrt(sim.R)) if (sim.R > 1 and variance_exists(sim.key)) else None
    return SimAnalysis(
        sim=sim, ks=ks_to_exact(sim), eps=X.dkw_bound(sim.R), mean_sim=float(np.mean(vals)), mean_exact=D.mean_max(sim.key, sim.n), se=se,
        median_sim=float(np.median(vals)), median_exact=D.median_max(sim.key, sim.n), noise=X.noise_band(sim.R),
    )


def exact_curve(key: str, n: int, norm: str = "B", npts: int = 500):
    """Gitter x und F^n(x) für die Darstellung; Bereich von F^n = 0,002 bis 0,995. Würfel: Treppe über 0 bis 6."""
    if key == "wuerfel":
        x = np.linspace(0.0, 6.0, 601)
        return x, D.cdf_max(key, x, n)
    lo = D.quantile_max(key, n, 0.998)
    hi = D.quantile_max(key, n, 0.005)
    x = np.linspace(lo, hi, npts)
    return x, D.cdf_max(key, x, n)


def limit_curve(key: str, n: int, x, norm: str = "B"):
    """Grenzgesetz an den Stellen x (in der Skala der Einzelwerte): G((x − b)/a); None für den Würfel und für Normal mit n = 1."""
    if key == "wuerfel" or (key == "normal" and n < 2):
        return None                    # kein Grenzgesetz bzw. keine Normierung (Normal braucht n >= 2)
    b, a = D.norming(key, n, norm)
    return D.limit_cdf(key, (np.asarray(x, dtype=float) - b) / a)


def verdict_sim(a: SimAnalysis) -> tuple[str, str]:
    """Verdikt zu Experiment 1 (Kernfrage 1): Abstand der Simulation zu F^n gegen die DKW-Schranke, Mittel gegen exakt."""
    s = a.sim
    inside = a.ks <= a.eps
    kopf = (f"{label(s.key)}, n = {s.n}, R = {thousands(s.R)} Blöcke: die simulierten Maxima liegen im Kolmogorov-Abstand {de(a.ks, 4)} zu F(x)ⁿ, "
            f"{'unter' if inside else 'über'} der Schranke {de(a.eps, 4)} (DKW, 99 %). ")
    if a.mean_exact is not None:
        exact_name = f"Hₙ = {de(a.mean_exact, 4)}" if s.key == "exp" else de(a.mean_exact, 4)
        mitte = f"Mittel {de(a.mean_sim, 4)} gegen exakt {exact_name}"
        if a.se is not None:
            mitte += f" (Standardfehler {de(a.se, 4)})."
        else:
            mitte += (f" (die Varianz ist unendlich, ein Standardfehler existiert nicht; Median {de(a.median_sim, 4)} gegen exakt {de(a.median_exact, 4)}).")
    else:
        mitte = f"Die Standard-Fréchet-Verteilung hat keinen Mittelwert: Median {de(a.median_sim, 4)} gegen exakt {de(a.median_exact, 4)}."
    if inside:
        return "success", kopf + mitte
    return "warning", kopf + "Bei 99 % kommt das ohne jeden Fehler in etwa 1 von 100 Läufen vor; ein anderer Zufalls-Seed zeigt, ob es bleibt. " + mitte


# ------------------------------------------------------------------------------------------------------------------------------------------
# Experiment 2: Abstand zum Grenzgesetz (exakt)
# ------------------------------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class LimitAnalysis:
    n: int
    norm: str
    d_common: float
    d_B: float
    d_A: float

    @property
    def d_sel(self) -> float:
        return self.d_B if self.norm == "B" else self.d_A

    def factor(self, norm: str | None = None) -> float:
        d = self.d_B if (norm or self.norm) == "B" else self.d_A
        return d / self.d_common

    @property
    def see_common(self) -> float:
        return X.blocks_to_see(self.d_common)

    @property
    def see_normal(self) -> float:
        return X.blocks_to_see(self.d_sel)


def analyse_limit(n: int, norm: str) -> LimitAnalysis:
    return LimitAnalysis(int(n), norm, X.common_distance(n), X.normal_distance(n, "B"), X.normal_distance(n, "A"))


def verdict_limit(a: LimitAnalysis, ref: dict | None = None) -> tuple[str, str]:
    """Verdikt zu Experiment 2 (Kernfrage 2): Abstand je Verteilung mit genannter Normierung, Schwellen, Sichtbarkeit."""
    ref = ref or REF.load()
    th = ref["thresholds"]
    m = f"{0.01}"
    n_exp = th["exp_std_integer"][m]
    n_nB = th["normal_B"][m]
    n_nA = th["normal_A"][m]
    n_bg_exp = ref["exp_best_gumbel"]["thresholds"][m]
    n_bg_nor = th["normal_best_gumbel"][m]
    text = (f"n = {n_label(a.n)}: Exponential, Gleich und Pareto-Verteilung (Standardnormierung) {dist_text(a.d_common, 5)}, "
            f"Normal (Normierung B) {dist_text(a.d_B, 4)}: Faktor {factor_text(a.factor('B'))}; mit Normierung A {dist_text(a.d_A, 4)} (Faktor {factor_text(a.factor('A'))}). "
            f"Unter 0,01 kommt der Abstand bei Exponential (Standardnormierung) ab n = {n_exp}, bei Normal (Normierung B) ab n ≈ {sig3(n_nB)} "
            f"(Normierung A: {'nicht bis 10³⁰' if n_nA is None else 'ab n ≈ ' + sig3(n_nA)}; mit jeweils bestem Gumbel {n_bg_exp} gegen {sig3(n_bg_nor)}). "
            f"Damit der Abstand {dist_text(a.d_common, 5)} die Rauschschranke 1,628/√R (99 %) erreicht, bräuchte man R ≈ {round_sig(a.see_common)} Blöcke (auf zwei Stellen gerundet; "
            f"in einer Probe lag er bei n = 365 mit diesem R in rund vier von fünf Läufen darüber); er wird deshalb gerechnet, nicht simuliert.")
    return "info", text


def threshold_rows(ref: dict | None = None) -> list[dict]:
    """Zeilen der Tabelle „kleinstes n für Abstand unter einer Marke“ aus der eingefrorenen Referenz (Zähleinheit n)."""
    ref = ref or REF.load()
    th = ref["thresholds"]
    rows = []
    for mark in C.MARKS:
        m = str(mark)
        A_note = ""
        if mark == 0.05:
            A_note = f"erstes n = {th['normal_A']['first_below_0.05']}, dauerhaft n ≥ {th['normal_A']['lasting_below_0.05']}"
        elif th["normal_A"].get(m) is not None:
            A_note = f"n ≥ {sig3(th['normal_A'][m])}"
        elif m in th["normal_A"]:
            A_note = "nicht bis 10³⁰"
        else:
            A_note = "nicht gerechnet"
        nb = th["normal_B"][m]
        bg = th["normal_best_gumbel"].get(m)
        rows.append({
            "Abstand unter": p_text(mark),
            "Exponential, Gleich, Pareto (Standardnormierung)": f"n ≥ {th['exp_std_integer'][m]}",
            "Exponential, bestes Gumbel": f"n ≥ {ref['exp_best_gumbel']['thresholds'][m]}",
            "Normal, Normierung B": f"n ≥ {sig3(nb)}" if nb is not None else "nicht bis 10¹⁴",
            "Normal, Normierung A": A_note,
            "Normal, bestes Gumbel": (f"n ≥ {sig3(bg)}" if bg is not None else "–"),
        })
    return rows


# ------------------------------------------------------------------------------------------------------------------------------------------
# Experiment 3: skalieren und typisieren
# ------------------------------------------------------------------------------------------------------------------------------------------
TYPE_NOTES = {
    "exp": "Exponential: die Skalierung ist eine Verschiebung um ln n bei Skala 1, keine Streckung.",
    "gleich": "Gleichverteilung: das Größte rückt an den oberen Rand 1 heran; die Skala ist 1/n.",
    "pareto": "Pareto-Verteilung: das Größte wächst wie n^(1/α); die Skala ist n^(1/α).",
    "frechet": "Standard-Fréchet: F(n·z)ⁿ = F(z) für jedes n (max-stabil), die Skala ist n.",
    "normal": "Normalverteilung: Gumbel-Typ, aber die Annäherung wird mit wachsendem n am langsamsten der vier (im Abstand ab n = 4 bei Normierung B, ab n = 6 bei Normierung A).",
}


def type_curves(key: str, n_list, norm: str = "B", npts: int = 801):
    """Normierte exakte Verteilungsfunktionen für die Blockgrößen n_list und das Grenzgesetz auf einem z-Gitter (Familie bestimmt den Bereich)."""
    fam = D.type_info(key)["family"]
    if fam is None:
        raise ValueError("kein Grenzgesetz")
    if fam == "weibull":
        z = np.linspace(-6.0, 0.5, npts)
    elif fam == "frechet":
        z = np.linspace(0.05, 6.0, npts)
    else:
        z = np.linspace(-3.0, 8.0, npts)
    curves = {}
    for n in n_list:
        if key == "normal" and n < 2:
            continue
        curves[n] = D.normed_cdf(key, n, z, norm)
    return z, curves, D.limit_cdf(key, z)


def normalisation_text(key: str, norm: str = "B") -> str:
    if key == "exp":
        return "Mₙ − ln n"
    if key == "gleich":
        return "n·(Mₙ − 1)"
    if D.is_pareto(key):
        a = D.pareto_alpha(key)
        return f"Mₙ / n^(1/{p_text(a)})"
    if key == "frechet":
        return "Mₙ / n"
    if key == "normal":
        return "(Mₙ − bₙ)/aₙ (Normierung B)" if norm == "B" else "(Mₙ − dₙ)/cₙ (Normierung A)"
    raise ValueError(key)


def verdict_type(key: str, n: int, norm: str = "B") -> tuple[str, str]:
    """Verdikt zu Experiment 3: Typ, ξ, Normierung und der gerechnete Abstand zum Grenzgesetz bei diesem n."""
    ti = D.type_info(key)
    if ti["family"] is None:
        return "info", "Der Würfel hat kein Grenzgesetz: das Größte von n Würfen liegt mit wachsendem n fast sicher bei 6; eine Normierung gibt es hier nicht."
    if key == "normal" and n < 2:
        return "info", "Die Normierung der Normalverteilung braucht n ≥ 2 (b = Φ⁻¹(1 − 1/n) ist für n = 1 nicht definiert)."
    d = X.distance(key, n, norm)
    xi = ti["xi"]
    xi_text = de(xi, 2) if xi not in (0.0, -1.0, 1.0) else {0.0: "0", -1.0: "−1", 1.0: "1"}[xi]
    kopf = (f"{label(key)}: {ti['typ']} (ξ = {xi_text}); normiert wird mit {normalisation_text(key, norm)}. "
            f"Bei n = {n_label(n)} liegt der Abstand zum Grenzgesetz bei {de(d, 5 if d < 0.001 else 4)}")
    if key == "frechet":
        return "success", kopf + ": die Standard-Fréchet-Verteilung ist für jedes n exakt max-stabil, F(n·z)ⁿ = F(z)."
    if key == "normal":
        return "info", kopf + f" (Abstand·ln n = {de(d * math.log(n), 3)}, Normierung {norm}). Der Typ heißt Gumbel-Typ; die Annäherung wird mit wachsendem n am langsamsten der vier (im Abstand ab n = 4 bei Normierung B, ab n = 6 bei Normierung A)."
    schluss = f" (n·Abstand = {de(n * d, 4)}, Grenzwert 2/e² = {de(2 / math.e ** 2, 4)})."
    if key == "gleich":
        schluss += " „Weibull-Typ“ ist nicht die Weibull-Verteilung der Ausfallzeiten."
    return "info", kopf + schluss


# ------------------------------------------------------------------------------------------------------------------------------------------
# Experiment 4: der Pegel, Normalverteilung
# ------------------------------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class PegelAnalysis:
    n: int
    p: float
    exact: float
    levels: dict            # Verfahren -> Pegel
    true_p: dict            # Verfahren -> wahre Überschreitungswahrscheinlichkeit am Pegel des Verfahrens
    xi_gev: float


def level_true_p(n: int, level: float) -> float:
    """Wahre Überschreitungswahrscheinlichkeit des Blockmaximums am Wert `level`: 1 − Φ(level)ⁿ."""
    sf = float(D.norm_sf(np.array([level]))[0])
    return -math.expm1(n * math.log1p(-sf))


def analyse_pegel(n: int, p: float, ref: dict | None = None) -> PegelAnalysis:
    ref = ref or REF.load()
    exact = D.quantile_max("normal", n, p)
    a_B, b_B = D.normal_constants_B(n)[0], D.normal_constants_B(n)[1]
    c_A, d_A = D.normal_constants_A(n)
    y = D.gumbel_y(p)
    fit = REF.best_fit(n)
    levels = {
        "gumbelA": d_A + c_A * y,
        "gumbelB": b_B + a_B * y,
        "gumbelbest": b_B + a_B * D.gev_quantile(fit["gumbel"]["mu"], fit["gumbel"]["sigma"], 0.0, p),
        "gevbest": b_B + a_B * D.gev_quantile(fit["gev"]["mu"], fit["gev"]["sigma"], fit["gev"]["xi"], p),
    }
    return PegelAnalysis(int(n), p, exact, levels, {k: level_true_p(n, v) for k, v in levels.items()}, fit["gev"]["xi"])


def verdict_pegel(a: PegelAnalysis, method: str) -> tuple[str, str]:
    """Verdikt zu Experiment 4 (Kernfrage 3): Pegel exakt gegen das gewählte Verfahren, wahre Wahrscheinlichkeit am Wert des Verfahrens."""
    lv = a.levels[method]
    diff = lv - a.exact
    richtung = "über" if diff > 0 else "unter"
    festgehalten = method != "gevbest"
    text = (f"Normalverteilung, n = {n_label(a.n)}, p = {p_text(a.p)}: exakt {de(a.exact, 3)}; {C.METHOD_LABELS[method]} {de(lv, 3)} "
            f"({de_signed(diff, 3)} Standardabweichungen der Einzelwerte, also {richtung} dem exakten Pegel); "
            f"die wahre Überschreitungswahrscheinlichkeit am Wert des Verfahrens ist {de(a.true_p[method], 4)} statt {p_text(a.p)}. "
            f"Gumbel B {de_signed(a.levels['gumbelB'] - a.exact, 3)}, Gumbel A {de_signed(a.levels['gumbelA'] - a.exact, 3)}, bestes Gumbel {de_signed(a.levels['gumbelbest'] - a.exact, 3)}, "
            f"beste GEV (ξ̂ = {de_signed(a.xi_gev, 2)}) {de_signed(a.levels['gevbest'] - a.exact, 3)}. "
            f"In allen Einstellungen der App liegt die beste GEV näher am exakten Pegel als die drei Gumbel-Verfahren (ξ = 0 festgehalten), bei kleinem p im Tail aber nicht mehr nah; "
            f"„beste“ heißt: beste Anpassung an die exakte Verteilung, kein Schätzer aus Daten.")
    kind = "success" if method == "gevbest" and abs(diff) < 0.1 else ("warning" if festgehalten and abs(diff) >= 0.05 else "info")
    return kind, text


def pegel_rows(a: PegelAnalysis) -> list[dict]:
    rows = [{"Verfahren": "exakt (aus Φⁿ)", "Pegel": de(a.exact, 4), "Differenz": "–", "wahre Wahrscheinlichkeit am Pegel": de(a.p, 4)}]
    for k, name in C.METHOD_LABELS.items():
        rows.append({"Verfahren": name, "Pegel": de(a.levels[k], 4), "Differenz": de_signed(a.levels[k] - a.exact, 3), "wahre Wahrscheinlichkeit am Pegel": de(a.true_p[k], 5)})
    return rows

"""Kolmogorov-Abstand der exakten Verteilung des Größten von n zum Grenzgesetz (exakt gerechnet, keine Simulation).

Abstand = sup |F^n(normiert) − Grenzgesetz|. Zähleinheit n (Beobachtungen je Block); die Normierung gehört in jeden Satz:
* Exponential, Gleich, Pareto-Verteilung: eine Standardnormierung (Verschiebung ln n, Skalierung n, Skalierung n^(1/α)); der Abstand ist für
  alle drei derselbe Wert sup_u |e^−u − (1 − u/n)^n| (Substitution u = e^−x bzw. −x bzw. x^−α).
* Normalverteilung: Normierung B (b = Φ⁻¹(1 − 1/n), a = 1/(n φ(b))) oder A (Lehrbuch-Konstanten).
Nur numpy und Standardbibliothek.
"""

import math

import numpy as np

import mxn_constants as C
import mxn_distributions as D


# ------------------------------------------------------------------------------------------------------------------------------------------
# Exponential, Gleich, Pareto-Verteilung, Fréchet: sup_u |e^−u − (1 − u/n)^n|
# ------------------------------------------------------------------------------------------------------------------------------------------
def _log_ratio(u, n: float):
    """t(u) = n·ln(1 − u/n) + u (<= 0); für u/n < 0,01 per Reihe −Σ_{k>=2} u^k/(k n^(k−1)), sonst direkt (vermeidet Auslöschung bei großem n)."""
    u = np.asarray(u, dtype=float)
    r = u / n
    small = r < 0.01
    with np.errstate(divide="ignore", invalid="ignore"):
        direct = n * np.log1p(-np.minimum(r, 1.0)) + u
        series = np.zeros_like(u)
        term = u * r          # u²/n
        for k in range(2, 10):
            series = series - term / k
            term = term * r
    return np.where(small, series, direct)


def _diff_u(u, n: float):
    """e^−u − (1 − u/n)^n = −e^−u · expm1(t(u)); nichtnegativ."""
    t = _log_ratio(u, n)
    with np.errstate(over="ignore", invalid="ignore"):
        d = -np.exp(-np.asarray(u, dtype=float)) * np.expm1(t)
    return np.where(np.isfinite(d), d, 0.0)


def _golden_max(f, lo: float, hi: float, iters: int = 80) -> float:
    """Maximum einer unimodalen Funktion auf [lo, hi] (goldener Schnitt)."""
    g = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(iters):
        if fc > fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = f(d)
    return max(fc, fd)


def common_distance(n: float, npts: int = 40001) -> float:
    """Abstand für Exponential, Gleich, Pareto-Verteilung (und Fréchet: dort mit anderer Normierung 0): sup_{0<=u<=n} (e^−u − (1 − u/n)^n); n >= 1 reell."""
    if n < 1.0:
        raise ValueError("n muss mindestens 1 sein")
    umax = min(float(n), 40.0)
    u = np.linspace(0.0, umax, npts)
    d = _diff_u(u, n)
    i = int(np.argmax(d))
    lo, hi = u[max(i - 1, 0)], u[min(i + 1, npts - 1)]
    best = _golden_max(lambda t: float(_diff_u(np.array([t]), n)[0]), lo, hi)
    return max(float(d[i]), best)


# ------------------------------------------------------------------------------------------------------------------------------------------
# Normalverteilung
# ------------------------------------------------------------------------------------------------------------------------------------------
def normal_distance(n: float, norm: str = "B", npts: int = C.GRID_POINTS) -> float:
    """Abstand der exakten Verteilung des Größten von n Normalverteilten zum Gumbel-Gesetz mit Normierung B oder A (Gitter z in [−4, 16])."""
    b, a = D.norming("normal", n, norm)
    z = np.linspace(C.GRID_X[0], C.GRID_X[1], npts)
    exact = D.cdf_max("normal", b + a * z, n)
    return float(np.max(np.abs(exact - D.limit_cdf("normal", z))))


def best_fit_distance(n: float, kind: str, params, npts: int = C.GRID_POINTS) -> float:
    """Abstand der exakten Normal-Maximum-Verteilung (Normierung B, Gitter z in [−4, 16]) zu einer GEV bzw. einem Gumbel mit den Parametern `params`
    (μ, σ[, ξ]) auf der z-Skala der Normierung B. Jede Wahl liefert eine obere Schranke des besten Abstands."""
    b, a = D.norming("normal", n, "B")
    z = np.linspace(C.GRID_X[0], C.GRID_X[1], npts)
    exact = D.cdf_max("normal", b + a * z, n)
    if kind == "gumbel":
        fit = D.gumbel_cdf_ab(z, params[0], params[1])
    else:
        fit = D.gev_cdf(z, params[0], params[1], params[2])
    return float(np.max(np.abs(exact - fit)))


def distance(key: str, n: float, norm: str = "B") -> float:
    """Abstand zum Grenzgesetz je Verteilung: Exponential/Gleich/Pareto-Verteilung der gemeinsame Wert, Fréchet 0 (max-stabil), Normal je Normierung."""
    if key == "normal":
        return normal_distance(n, norm)
    if key == "frechet":
        return 0.0
    if key == "wuerfel":
        raise ValueError("der Würfel hat kein Grenzgesetz")
    return common_distance(n)


def blocks_to_see(d: float) -> float:
    """So viele simulierte Blöcke braucht man, damit der Abstand d das Rauschen 1,628/√R (Kolmogorov, 99 %) erreicht: R ≈ (1,628/d)²."""
    if d <= 0.0:
        return math.inf
    return (C.KS_999 / d) ** 2


def noise_band(R: int) -> float:
    """Rauschen der empirischen Verteilungsfunktion bei R Blöcken (Kolmogorov, 99 %): 1,628/√R."""
    return C.KS_999 / math.sqrt(R)


def dkw_bound(R: int, alpha: float = C.DKW_ALPHA) -> float:
    """DKW-Schranke: P(sup|F_R − F| > ε) <= 2·exp(−2 R ε²) = alpha, also ε = √(ln(2/alpha)/(2R))."""
    return math.sqrt(math.log(2.0 / alpha) / (2.0 * R))


# ------------------------------------------------------------------------------------------------------------------------------------------
# Kleinstes n unter einer Marke (für Werkzeug und Tests; die App zeigt die eingefrorene Tabelle)
# ------------------------------------------------------------------------------------------------------------------------------------------
def first_log_cross(f, lo: float, hi: float, thr: float, step: float = 0.25, iters: int = 40):
    """Kleinstes L in [lo, hi] mit f(L) < thr (f fallend in L = log10 n): grob in Schritten, dann Bisektion; None, wenn nie."""
    prev = lo
    L = lo
    while L <= hi:
        if f(L) < thr:
            a, b = prev, L
            if a == L:
                return L
            for _ in range(iters):
                m = 0.5 * (a + b)
                if f(m) < thr:
                    b = m
                else:
                    a = m
            return b
        prev = L
        L += step
    return None


def smallest_integer_n(mark: float, nmax: int = 2000):
    """Kleinstes ganzzahliges n, ab dem der Abstand (Exponential/Gleich/Pareto) bis nmax dauerhaft unter `mark` liegt; None, wenn nmax nicht reicht."""
    vals = [common_distance(n, 4001) for n in range(1, nmax + 1)]
    for n in range(1, nmax + 1):
        if all(v < mark for v in vals[n - 1:]):
            return n
    return None

"""Exakte Verteilung des Größten von n (F^n), Umkehrungen, Normierungen und Grenzgesetze. Nur numpy und Standardbibliothek (scipy nur in den Tests).

Modell: n unabhängige, gleich verteilte Beobachtungen mit stetiger Verteilungsfunktion F (Würfel: diskret); ihr Größtes M_n hat P(M_n <= x) = F(x)^n.
Vorzeichen von ξ (einmal festgelegt): ξ > 0 Fréchet-Typ, ξ = 0 Gumbel-Typ, ξ < 0 Weibull-Typ. `scipy.stats.genextreme` hat c = −ξ,
`scipy.stats.genpareto` hat c = ξ; beides steht nur als Referenz in den Tests.
"""

import math
from fractions import Fraction
from statistics import NormalDist

import numpy as np

import mxn_constants as C

_ND = NormalDist()
_SQRT2 = math.sqrt(2.0)
_SQRT2PI = math.sqrt(2.0 * math.pi)
EULER_GAMMA = 0.5772156649015329
_V_MAX = 1.0 - 2.0 ** -53   # größtes v, das die Umkehrung der Normalverteilung noch annimmt (Grenzfall v = 1: Block mit u = 0)


# ------------------------------------------------------------------------------------------------------------------------------------------
# Normalverteilung ohne scipy: Φ über math.erfc, Umkehrung über statistics.NormalDist
# ------------------------------------------------------------------------------------------------------------------------------------------
def erfc_array(a) -> np.ndarray:
    """math.erfc elementweise (numpy hat keine erfc); `math.erfc` ist auf jeder Plattform bis auf das letzte Bit gleich genau."""
    a = np.asarray(a, dtype=float)
    return np.fromiter((math.erfc(v) for v in a.ravel().tolist()), dtype=float, count=a.size).reshape(a.shape)


def norm_sf(y) -> np.ndarray:
    """1 − Φ(y), auch im oberen Tail mit voller relativer Genauigkeit."""
    return 0.5 * erfc_array(np.asarray(y, dtype=float) / _SQRT2)


def norm_pdf(y):
    y = np.asarray(y, dtype=float)
    return np.exp(-0.5 * y * y) / _SQRT2PI


def norm_isf(s: float) -> float:
    """y mit 1 − Φ(y) = s, für 0 < s < 1."""
    if not 0.0 < s < 1.0:
        raise ValueError("s muss zwischen 0 und 1 liegen")
    return -_ND.inv_cdf(s)


# ------------------------------------------------------------------------------------------------------------------------------------------
# Einzelverteilungen: log F, Umkehrung, Typ
# ------------------------------------------------------------------------------------------------------------------------------------------
def pareto_alpha(key: str) -> float:
    return C.PARETO_ALPHA[key]


def is_pareto(key: str) -> bool:
    return key in C.PARETO_ALPHA


def log_cdf(key: str, x) -> np.ndarray:
    """log F(x) elementweise (−∞ links des Trägers)."""
    x = np.asarray(x, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        if key == "exp":
            return np.where(x > 0, np.log1p(-np.exp(-np.maximum(x, 1e-300))), -np.inf)
        if key == "gleich":
            return np.where(x <= 0, -np.inf, np.log(np.minimum(x, 1.0)))
        if is_pareto(key):
            a = pareto_alpha(key)
            return np.where(x > 1, np.log1p(-np.maximum(x, 1.0) ** (-a)), -np.inf)
        if key == "frechet":
            return np.where(x > 0, -1.0 / np.maximum(x, 1e-300), -np.inf)
        if key == "normal":
            return np.log1p(-norm_sf(x))
        if key == "wuerfel":
            k = np.clip(np.floor(x), 0, 6)
            return np.log(k / 6.0)
    raise ValueError(f"unbekannte Verteilung {key!r}")


def cdf_max(key: str, x, n: float) -> np.ndarray:
    """Exakte Verteilungsfunktion des Größten von n: F(x)^n (n reell erlaubt, n >= 1)."""
    with np.errstate(invalid="ignore", over="ignore"):
        return np.exp(n * log_cdf(key, x))


def inv_from_v(key: str, v) -> np.ndarray:
    """Wert x mit 1 − F(x) = v für v in (0, 1] (v = Überschreitungswahrscheinlichkeit der Einzelverteilung).

    Abkürzung für das Größte von n: das Maximum von n Umkehrungen ist die Umkehrung des kleinsten v, weil die Umkehrung monoton fällt."""
    v = np.asarray(v, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        if key == "exp":
            return -np.log(v)
        if key == "gleich":
            return 1.0 - v
        if is_pareto(key):
            return v ** (-1.0 / pareto_alpha(key))
        if key == "frechet":
            return -1.0 / np.log1p(-np.minimum(v, 1.0 - 2.0 ** -53))
        if key == "normal":
            vc = np.minimum(v, _V_MAX)
            return -np.fromiter((_ND.inv_cdf(t) for t in vc.ravel().tolist()), dtype=float, count=vc.size).reshape(vc.shape)
        if key == "wuerfel":
            return np.floor(6.0 * (1.0 - v)) + 1.0
    raise ValueError(f"unbekannte Verteilung {key!r}")


def quantile_max(key: str, n: float, p: float) -> float:
    """Wert x mit P(M_n > x) = p (Pegel zur Überschreitungswahrscheinlichkeit p je Block): 1 − F(x) = 1 − (1 − p)^(1/n)."""
    if not 0.0 < p < 1.0:
        raise ValueError("p muss zwischen 0 und 1 liegen")
    if key == "wuerfel":
        raise ValueError("für den Würfel gibt es keinen Pegel (diskret)")
    s = -math.expm1(math.log1p(-p) / n)
    return float(inv_from_v(key, np.array([s]))[0])


def median_max(key: str, n: float) -> float:
    if key == "wuerfel":
        target = 0.5
        for k in range(1, 7):
            if (k / 6.0) ** n >= target:
                return float(k)
    return quantile_max(key, n, 0.5)


def mean_max(key: str, n: float):
    """E[M_n] exakt bzw. per Quadratur; None, wenn der Mittelwert nicht existiert (Pareto-Verteilung α <= 1, Fréchet mit α = 1)."""
    if key == "wuerfel":
        return float(sum(Fraction(k) * (Fraction(k, 6) ** int(n) - Fraction(k - 1, 6) ** int(n)) for k in range(1, 7)))
    if key == "exp":
        if not float(n).is_integer():
            raise ValueError("E[M_n] = H_n gilt für ganzzahliges n")
        return harmonic(int(n))
    if key == "gleich":
        return n / (n + 1.0)
    if is_pareto(key):
        a = pareto_alpha(key)
        s = -1.0 / a
        if a <= 1.0:
            return None
        # V = kleinstes v von n Gleichverteilten ~ Beta(1, n); M = V^(−1/α); E[V^s] = Γ(1+s)Γ(n+1)/Γ(n+1+s)
        return math.exp(math.lgamma(1.0 + s) + math.lgamma(n + 1.0) - math.lgamma(n + 1.0 + s))
    if key == "frechet":
        return None
    if key == "normal":
        return normal_mean_max(n)
    raise ValueError(key)


def harmonic(n: int) -> float:
    return float(sum(Fraction(1, k) for k in range(1, n + 1))) if n <= 400 else math.fsum(1.0 / k for k in range(1, n + 1))


def normal_mean_max(n: float, npts: int = 40001) -> float:
    """E[M_n] der Normalverteilung per Quadratur (Trapezregel) über die Dichte n·φ(x)·Φ(x)^(n−1)."""
    x = np.linspace(-9.0, 14.0, npts)
    with np.errstate(invalid="ignore"):
        dens = n * norm_pdf(x) * np.exp((n - 1.0) * log_cdf("normal", x))
    dens = np.nan_to_num(dens)
    h = x[1] - x[0]
    return float(h * (np.sum(x * dens) - 0.5 * (x[0] * dens[0] + x[-1] * dens[-1])))


# ------------------------------------------------------------------------------------------------------------------------------------------
# Typ, Normierung, Grenzgesetz
# ------------------------------------------------------------------------------------------------------------------------------------------
def type_info(key: str) -> dict:
    """Typ des Grenzgesetzes: Name, ξ, Familie. Würfel: kein Grenzgesetz."""
    if key in ("exp", "normal"):
        return {"typ": "Gumbel-Typ", "xi": 0.0, "family": "gumbel"}
    if key == "gleich":
        return {"typ": "Weibull-Typ", "xi": -1.0, "family": "weibull"}
    if is_pareto(key):
        a = pareto_alpha(key)
        return {"typ": "Fréchet-Typ", "xi": 1.0 / a, "family": "frechet", "alpha": a}
    if key == "frechet":
        return {"typ": "Fréchet-Typ", "xi": 1.0, "family": "frechet", "alpha": 1.0}
    return {"typ": "kein Grenzgesetz", "xi": None, "family": None}


def normal_constants_B(n: float):
    """Normierung B: b = Φ⁻¹(1 − 1/n), a = 1/(n·φ(b)). Braucht n > 1."""
    if n <= 1.0:
        raise ValueError("Normierung B braucht n > 1")
    b = norm_isf(1.0 / n)
    a = 1.0 / (n * float(norm_pdf(b)))
    return a, b


def normal_constants_A(n: float):
    """Normierung A (Lehrbuch): c = (2 ln n)^(−1/2), d = √(2 ln n) − (ln ln n + ln 4π)/(2√(2 ln n)). Braucht n > 1."""
    if n <= 1.0:
        raise ValueError("Normierung A braucht n > 1")
    L = math.log(n)
    c = (2.0 * L) ** -0.5
    d = math.sqrt(2.0 * L) - (math.log(L) + math.log(4.0 * math.pi)) / (2.0 * math.sqrt(2.0 * L))
    return c, d


def norming(key: str, n: float, norm: str = "B"):
    """(b, a): z = (x − b)/a ist die normierte Größe, deren Grenzgesetz `limit_cdf` ist (Exponential: Verschiebung ln n, Skala 1)."""
    if key == "exp":
        return math.log(n), 1.0
    if key == "gleich":
        return 1.0, 1.0 / n
    if is_pareto(key):
        return 0.0, n ** (1.0 / pareto_alpha(key))
    if key == "frechet":
        return 0.0, float(n)
    if key == "normal":
        if norm == "A":
            c, d = normal_constants_A(n)
            return d, c
        a, b = normal_constants_B(n)
        return b, a
    raise ValueError(f"keine Normierung für {key!r}")


def limit_cdf(key: str, z) -> np.ndarray:
    """Grenzgesetz der normierten Größe z: Gumbel exp(−e^−z), Weibull-Typ exp(z) für z <= 0 (ξ = −1), Fréchet exp(−z^−α) für z > 0."""
    z = np.asarray(z, dtype=float)
    fam = type_info(key)["family"]
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        if fam == "gumbel":
            return np.exp(-np.exp(-z))
        if fam == "weibull":
            return np.where(z <= 0, np.exp(np.minimum(z, 0.0)), 1.0)
        if fam == "frechet":
            a = type_info(key)["alpha"]
            return np.where(z > 0, np.exp(-np.maximum(z, 1e-300) ** (-a)), 0.0)
    raise ValueError(f"kein Grenzgesetz für {key!r}")


def normed_cdf(key: str, n: float, z, norm: str = "B") -> np.ndarray:
    """Exakte Verteilung der normierten Größe (M_n − b)/a an den Stellen z."""
    b, a = norming(key, n, norm)
    return cdf_max(key, b + a * np.asarray(z, dtype=float), n)


# ------------------------------------------------------------------------------------------------------------------------------------------
# Pegel der Gumbel-Näherungen und der GEV (ξ-Vorzeichen: ξ > 0 Fréchet-Typ)
# ------------------------------------------------------------------------------------------------------------------------------------------
def gumbel_y(p: float) -> float:
    """y = −ln(−ln(1 − p)): Gumbel-Quantil der Überschreitungswahrscheinlichkeit p je Block."""
    return -math.log(-math.log1p(-p))


def gev_quantile(mu: float, sigma: float, xi: float, p: float) -> float:
    """Wert mit Überschreitungswahrscheinlichkeit p je Block: μ − (σ/ξ)(1 − y^(−ξ)) mit y = −ln(1 − p); ξ = 0: μ − σ ln y."""
    y = -math.log1p(-p)
    if abs(xi) < 1e-12:
        return mu - sigma * math.log(y)
    return mu - sigma / xi * (1.0 - y ** (-xi))


def gev_cdf(x, mu: float, sigma: float, xi: float) -> np.ndarray:
    """GEV-Verteilungsfunktion (Vorzeichen: ξ > 0 Fréchet-Typ, so wie in der Linie; scipy `genextreme` hat c = −ξ)."""
    z = (np.asarray(x, dtype=float) - mu) / sigma
    if abs(xi) < 1e-12:
        with np.errstate(over="ignore"):
            return np.exp(-np.exp(-z))
    t = 1.0 + xi * z
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        inner = np.exp(-np.power(np.maximum(t, 1e-300), -1.0 / xi))
    return np.where(t > 0, inner, 0.0 if xi > 0 else 1.0)


def gumbel_cdf_ab(x, mu: float, sigma: float) -> np.ndarray:
    return gev_cdf(x, mu, sigma, 0.0)

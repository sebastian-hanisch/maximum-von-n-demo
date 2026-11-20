"""Unabhängige Referenzen für die Tests: anderer Rechenweg als die App (scipy, math.erfc-Bisektion, Aufzählung, Gitter direkt an F^n).

Nichts hier ruft Code der App auf. scipy steht nur hier und in den Tests; ξ-Vorzeichen: `genextreme` hat c = −ξ, `genpareto` hat c = ξ."""

import math

import numpy as np
from scipy import stats

EULER_GAMMA = 0.5772156649015329
SQRT2 = math.sqrt(2.0)


def Phi(x: float) -> float:
    return 0.5 * math.erfc(-x / SQRT2)


def phi(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def bisect(f, lo: float, hi: float, it: int = 200) -> float:
    """Nullstelle von f in [lo, hi] (Vorzeichenwechsel vorausgesetzt)."""
    flo = f(lo)
    for _ in range(it):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if flo * fm <= 0:
            hi = mid
        else:
            lo, flo = mid, fm
    return 0.5 * (lo + hi)


def normal_pegel_bisection(n: float, p: float) -> float:
    """Exakter Pegel: Φ(x)^n = 1 − p, gelöst nur mit math (Bisektion über log), unabhängig von NormalDist.inv_cdf."""
    return bisect(lambda x: n * math.log(Phi(x)) - math.log1p(-p), -2.0, 12.0)


def gumbel_B_constants(n: float):
    """b = Φ⁻¹(1 − 1/n) durch Bisektion, a = 1/(n φ(b))."""
    b = bisect(lambda x: (1.0 - Phi(x)) - 1.0 / n, -1.0, 12.0)
    return 1.0 / (n * phi(b)), b


def gumbel_A_constants(n: float):
    L = math.log(n)
    c = (2 * L) ** -0.5
    d = math.sqrt(2 * L) - (math.log(L) + math.log(4 * math.pi)) / (2 * math.sqrt(2 * L))
    return c, d


def ks_exp_direct(n: float, npts: int = 2_000_001) -> float:
    """Abstand Exponential an F^n selbst (nicht über u): sup_x |F^n(x + ln n) − exp(−e^−x)| mit scipy.stats.expon."""
    x = np.linspace(-math.log(n) + 1e-12, 30.0, npts)
    fn = stats.expon.cdf(x + math.log(n)) ** n
    return float(np.max(np.abs(fn - np.exp(-np.exp(-x)))))


def ks_gleich_direct(n: float, npts: int = 2_000_001) -> float:
    """Abstand Gleich an F^n: x = 1 + z/n, z in [−n, 0]; Grenzgesetz e^z (Weibull-Typ, ξ = −1)."""
    z = np.linspace(-min(n, 40.0), 0.0, npts)
    fn = stats.uniform.cdf(1.0 + z / n) ** n
    return float(np.max(np.abs(fn - np.exp(z))))


def ks_pareto_direct(n: float, alpha: float, npts: int = 2_000_001) -> float:
    """Abstand Pareto-Verteilung an F^n: x = n^(1/α)·z, Grenzgesetz exp(−z^−α) (Fréchet-Typ); scipy `pareto` hat b = α."""
    z = np.linspace(n ** (-1.0 / alpha) * (1 + 1e-12), 40.0, npts)
    fn = stats.pareto.cdf(n ** (1.0 / alpha) * z, b=alpha) ** n
    return float(np.max(np.abs(fn - np.exp(-z ** (-alpha)))))


def ks_normal_grid(n: float, c_or_a: float, d_or_b: float, npts: int) -> float:
    """Normal-Abstand an F^n = scipy norm.cdf^n gegen Gumbel, Konstanten (Skala, Lage) gegeben, Gitter z in [−4, 16]."""
    z = np.linspace(-4.0, 16.0, npts)
    fn = np.exp(n * np.log1p(-stats.norm.sf(c_or_a * z + d_or_b)))
    return float(np.max(np.abs(fn - np.exp(-np.exp(-z)))))


def genextreme_cdf(x, mu, sigma, xi):
    """GEV mit scipy: c = −ξ."""
    return stats.genextreme.cdf(x, c=-xi, loc=mu, scale=sigma)

"""Orakel-Tests (anderer Rechenweg als die App): scipy.stats, Aufzählung, Bisektion mit math.erfc, Gitter direkt an F^n, Brüche.

(a) F^n der App gegen scipy.stats hoch n; (b) Abstand Exponential/Gleich/Pareto gegen drei unabhängige Gitter; (c) Normal-Abstand mit zwei Gitterauflösungen
und gegen scipy-Konstanten; (d) Schwellen-n; (e) Pegel gegen scipy `isf` und Bisektion, Gumbel A und B gegen die ausgeschriebenen Formeln;
(f) Momente als Brüche gegen Quadratur; (h) Simulation gegen F^n im Band. Die Zufallsergebnisse sind Bänder, nie ein Test auf das letzte Bit."""

import math
from fractions import Fraction

import numpy as np
import pytest
from helpers_ref import (
    gumbel_A_constants,
    gumbel_B_constants,
    ks_exp_direct,
    ks_gleich_direct,
    ks_normal_grid,
    ks_pareto_direct,
    normal_pegel_bisection,
)
from scipy import integrate, stats

import mxn_constants as C
import mxn_distance as X
import mxn_distributions as D
import mxn_evaluation as E
import mxn_scenario as S

XGRID = np.concatenate([np.linspace(-3, 12, 301), np.linspace(0.0001, 3, 200), np.linspace(1, 80, 400)])


# --- (a) F^n gegen scipy.stats -------------------------------------------------------------------------------------------------------------------
SCIPY_CDF = {
    "exp": lambda x: stats.expon.cdf(x),
    "gleich": lambda x: stats.uniform.cdf(x),
    "pareto15": lambda x: stats.pareto.cdf(x, b=1.5),
    "pareto2": lambda x: stats.pareto.cdf(x, b=2.0),
    "pareto4": lambda x: stats.pareto.cdf(x, b=4.0),
    "frechet": lambda x: stats.invweibull.cdf(x, c=1.0),           # Standard-Fréchet exp(−1/x) = Weibull-Inverse mit c = 1
    "normal": lambda x: stats.norm.cdf(x),
}


@pytest.mark.parametrize("key", list(SCIPY_CDF))
@pytest.mark.parametrize("n", [1, 2, 7, 20, 365])
def test_fn_der_app_gleich_scipy_hoch_n(key, n):
    x = XGRID
    assert D.cdf_max(key, x, n) == pytest.approx(SCIPY_CDF[key](x) ** n, abs=1e-12)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_wuerfel_fn_gegen_aufzaehlung(n):
    import itertools
    outcomes = list(itertools.product(range(1, 7), repeat=n))
    for k in range(0, 8):
        cnt = sum(1 for o in outcomes if max(o) <= k)
        assert float(D.cdf_max("wuerfel", float(k), n)) == pytest.approx(cnt / 6 ** n, abs=1e-15)


# --- (b) Abstand Exponential/Gleich/Pareto gegen drei unabhängige Gitter ----------------------------------------------------------------------
@pytest.mark.parametrize("n", [1, 2, 3, 10, 50, 365, 1000])
def test_abstand_drei_verteilungen_gleich_auf_1e7(n):
    d = X.common_distance(n)
    assert ks_exp_direct(n, 800_001) == pytest.approx(d, abs=1e-7)
    assert ks_gleich_direct(n, 800_001) == pytest.approx(d, abs=1e-7)
    assert ks_pareto_direct(n, 2.0, 800_001) == pytest.approx(d, abs=1e-7)


@pytest.mark.parametrize("alpha", [1.5, 4.0])
def test_abstand_pareto_haengt_nicht_an_alpha(alpha):
    assert ks_pareto_direct(20, alpha, 800_001) == pytest.approx(X.common_distance(20), abs=1e-7)


def test_abstand_n1_ist_e_minus_1_und_n2_ist_016190256():
    assert X.common_distance(1) == pytest.approx(math.exp(-1), abs=1e-12)
    assert X.common_distance(2) == pytest.approx(0.16190256, abs=1e-8)


# --- (c) Normal-Abstand: zwei Gitterauflösungen und scipy-Konstanten --------------------------------------------------------------------------
@pytest.mark.parametrize("n", [5, 20, 365, 10_000])
def test_normal_abstand_zwei_gitter_gleich_auf_1e7(n):
    for norm in ("B", "A"):
        assert X.normal_distance(n, norm, 200_001) == pytest.approx(X.normal_distance(n, norm, 800_001), abs=1e-7)


@pytest.mark.parametrize("n", [2, 20, 365, 10_000, 10 ** 6])
def test_normal_abstand_gegen_scipy_konstanten(n):
    b = float(stats.norm.isf(1.0 / n))
    a = 1.0 / (n * float(stats.norm.pdf(b)))
    assert X.normal_distance(n, "B") == pytest.approx(ks_normal_grid(n, a, b, 400_001), abs=1e-7)
    L = math.log(n)
    c = (2 * L) ** -0.5
    d = math.sqrt(2 * L) - (math.log(L) + math.log(4 * math.pi)) / (2 * math.sqrt(2 * L))
    assert X.normal_distance(n, "A") == pytest.approx(ks_normal_grid(n, c, d, 400_001), abs=1e-7)


@pytest.mark.parametrize("n", [3, 100, 5000])
def test_normierung_b_konstanten_gegen_bisektion_und_scipy(n):
    a, b = D.normal_constants_B(n)
    a_ref, b_ref = gumbel_B_constants(n)
    assert b == pytest.approx(b_ref, abs=1e-9) and a == pytest.approx(a_ref, rel=1e-8)
    assert b == pytest.approx(float(stats.norm.isf(1.0 / n)), abs=1e-12)
    c, d = D.normal_constants_A(n)
    c_ref, d_ref = gumbel_A_constants(n)
    assert (c, d) == pytest.approx((c_ref, d_ref), rel=1e-14)


# --- (d) Schwellen-n ---------------------------------------------------------------------------------------------------------------------------
def _u_grid_distance(n: int) -> float:
    """Unabhängig von mxn_distance: max_u (e^−u − (1 − u/n)^n) auf einem festen Gitter mit 400 001 Punkten."""
    u = np.linspace(0.0, min(float(n), 40.0), 400_001)
    with np.errstate(divide="ignore"):
        return float(np.max(np.exp(-u) - np.exp(n * np.log1p(-u / n))))


@pytest.mark.parametrize("mark,n_expected", [(0.05, 6), (0.02, 14), (0.01, 28), (0.005, 55), (0.001, 272)])
def test_schwellen_n_exponential_ganzzahlig_exakt(ref, mark, n_expected):
    assert ref["thresholds"]["exp_std_integer"][str(mark)] == n_expected
    assert _u_grid_distance(n_expected - 1) >= mark > _u_grid_distance(n_expected)         # n − 1 liegt noch darüber, n darunter
    assert all(_u_grid_distance(n) < mark for n in range(n_expected, n_expected + 60))      # und dauerhaft (Abstand fällt streng, Gitter)


def test_abstand_exponential_faellt_streng_n_1_bis_2000_auf_dem_gitter():
    vals = [X.common_distance(n, 4001) for n in range(1, 2001)]
    assert all(vals[i + 1] < vals[i] for i in range(len(vals) - 1))


@pytest.mark.parametrize("mark,n_expected", [(0.05, 11.2), (0.02, 544.0), (0.01, 2.86e5)])
def test_schwellen_n_normal_b_drei_stellen_und_abstand_am_ergebnis(ref, mark, n_expected):
    n_thr = ref["thresholds"]["normal_B"][str(mark)]
    assert float(f"{n_thr:.3g}") == n_expected                                              # auf 3 Stellen gerundet
    b = float(stats.norm.isf(1.0 / n_thr))
    a = 1.0 / (n_thr * float(stats.norm.pdf(b)))
    assert ks_normal_grid(n_thr, a, b, 400_001) == pytest.approx(mark, abs=1e-6)           # der Abstand am Ergebnis ist die Marke
    below = ks_normal_grid(n_thr * 0.97, *_b_a(n_thr * 0.97), 400_001)
    assert below > mark                                                                      # etwas kleineres n liegt darüber


def _b_a(n):
    b = float(stats.norm.isf(1.0 / n))
    return 1.0 / (n * float(stats.norm.pdf(b))), b


# --- (e) Pegel ----------------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("n", [2, 24, 365, 8760])
@pytest.mark.parametrize("p", [0.1, 0.01, 0.001, 0.0001])
def test_pegel_exakt_gegen_scipy_isf_und_bisektion(n, p):
    sf = -math.expm1(math.log1p(-p) / n)
    assert D.quantile_max("normal", n, p) == pytest.approx(float(stats.norm.isf(sf)), abs=1e-9)
    assert D.quantile_max("normal", n, p) == pytest.approx(normal_pegel_bisection(n, p), abs=1e-6)


@pytest.mark.parametrize("n", [24, 365, 8760])
@pytest.mark.parametrize("p", [0.01, 0.001])
def test_gumbel_a_und_b_gegen_die_ausgeschriebenen_formeln(n, p):
    a_ref, b_ref = gumbel_B_constants(n)
    c_ref, d_ref = gumbel_A_constants(n)
    y = -math.log(-math.log(1 - p))
    pa = E.analyse_pegel(n, p)
    assert pa.levels["gumbelB"] == pytest.approx(b_ref + a_ref * y, abs=1e-7)
    assert pa.levels["gumbelA"] == pytest.approx(d_ref + c_ref * y, abs=1e-9)
    assert pa.true_p["gumbelB"] == pytest.approx(float(-np.expm1(n * np.log1p(-stats.norm.sf(pa.levels["gumbelB"])))), rel=1e-6)


@pytest.mark.parametrize("key,mu,sigma,xi", [("normal", 0.0, 1.0, 0.0)])
def test_gev_quantil_gegen_scipy_vorzeichen(key, mu, sigma, xi):
    # ξ-Vorzeichen: scipy genextreme hat c = −ξ. Die Gegenprobe mit c = +ξ liefert bei ξ ≠ 0 einen anderen Wert.
    for xi in (-0.3, 0.2):
        for p in (0.1, 0.01):
            mine = D.gev_quantile(1.5, 0.8, xi, p)
            assert mine == pytest.approx(float(stats.genextreme.isf(p, c=-xi, loc=1.5, scale=0.8)), abs=1e-9)
            assert abs(mine - float(stats.genextreme.isf(p, c=+xi, loc=1.5, scale=0.8))) > 1e-3
            x = np.array([mine])
            assert float(D.gev_cdf(x, 1.5, 0.8, xi)[0]) == pytest.approx(1 - p, abs=1e-9)
    # ξ → 0 stetig ins Gumbel-Quantil
    assert D.gev_quantile(1.5, 0.8, 1e-9, 0.01) == pytest.approx(D.gev_quantile(1.5, 0.8, 0.0, 0.01), abs=1e-6)


def test_gpd_vorzeichen_genpareto_c_gleich_xi():
    # genpareto: c = ξ (die Ränder der Linie): ξ > 0 schwerer Rand, ξ < 0 endlicher Endpunkt −σ/ξ
    assert float(stats.genpareto.cdf(1e6, c=0.5)) < 1.0
    assert float(stats.genpareto.cdf(2.0 + 1e-9, c=-0.5)) == pytest.approx(1.0)       # Endpunkt σ/|ξ| = 2


# --- (f) Momente ---------------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("n", [1, 2, 3, 10, 20, 365])
def test_exponential_mittel_h_n_als_bruch_gegen_quadratur(n):
    H = sum(Fraction(1, k) for k in range(1, n + 1))
    q, _ = integrate.quad(lambda t: 1 - (1 - math.exp(-t)) ** n, 0, 80, limit=300)
    assert D.mean_max("exp", n) == pytest.approx(float(H), rel=1e-13)
    assert float(H) == pytest.approx(q, abs=1e-8)


@pytest.mark.parametrize("key", ["pareto15", "pareto2", "pareto4"])
@pytest.mark.parametrize("n", [1, 2, 20, 365])
def test_pareto_mittel_geschlossene_formel_gegen_quadratur(key, n):
    alpha = D.pareto_alpha(key)
    q, _ = integrate.quad(lambda x: 1 - (1 - x ** -alpha) ** n, 1, np.inf, limit=500)
    assert D.mean_max(key, n) == pytest.approx(1 + q, rel=1e-6)                          # E[M] = 1 + ∫₁^∞ (1 − F^n) dx


@pytest.mark.parametrize("n", [1, 2, 5, 20, 365])
def test_normal_mittel_gegen_scipy_quadratur(n):
    q, _ = integrate.quad(lambda t: t * n * stats.norm.pdf(t) * stats.norm.cdf(t) ** (n - 1), -12, 14, limit=500)
    assert D.mean_max("normal", n) == pytest.approx(q, abs=1e-9)


# --- Abkürzung gegen Brute Force ---------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("key", list(C.DIST_KEYS))
def test_abkuerzung_gleich_brute_force_n_umkehrungen_dann_maximum(key):
    sim = S.simulate_blocks(key, 20, 300, 4242)
    brute = S.brute_force_block_maxima(key, 20, 300, 4242)
    assert np.max(np.abs(sim.values - brute) / np.maximum(1.0, np.abs(brute))) < 1e-12


# --- (h) Simulation gegen F^n im Band ----------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("seed", [101, 202, 303])
def test_simulation_ks_unter_1_2_dkw_fuer_vier_verteilungen(seed):
    R, n = 20000, 20
    v = S.block_vmin(n, R, seed)                                                          # ein Zufallsstrom, vier Umkehrungen
    eps = X.dkw_bound(R)
    for key in ("exp", "gleich", "pareto2", "normal"):
        values = D.inv_from_v(key, v)
        ks = S.ks_statistic(values, lambda x, key=key: D.cdf_max(key, x, n))
        assert ks < 1.2 * eps, (key, seed, ks, eps)
        mean_exact = D.mean_max(key, n)
        if key != "pareto2":                                                              # α = 2: Varianz unendlich, Standardfehler gibt es nicht
            assert abs(values.mean() - mean_exact) < 4 * values.std(ddof=1) / math.sqrt(R), (key, seed)

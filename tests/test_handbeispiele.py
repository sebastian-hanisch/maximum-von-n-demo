"""Handbeispiele (Brüche, Ganzzahlen, Tafelwerte) gegen die Funktionen der App.

Die Erwartungen stehen als Rechnung im Test (Aufzählung aller Ausgänge, Brüche, Handformeln), nicht als abgeschriebene Ausgabe der App.
Von Hand nachrechenbar (Papier, Taschenrechner ohne Wurzelsuche): Würfel, Gleich, Normal n = 2 und 3 in geschlossener Form, Hₙ, Bootstrap;
u* = 1,59362, die Pegel-Bisektion bei n = 365 und die GEV-Zeugen brauchen Wurzelsuche oder Gitter."""

import itertools
import math
from fractions import Fraction

import numpy as np
import pytest
from helpers_ref import (
    Phi,
    bisect,
    ks_exp_direct,
    ks_gleich_direct,
    ks_pareto_direct,
    normal_pegel_bisection,
)
from scipy import optimize

import mxn_distance as X
import mxn_distributions as D
import mxn_evaluation as E


# --- 1. Würfel -------------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("n", [2, 3])
def test_wuerfel_verteilung_des_groessten_durch_aufzaehlung(n):
    outcomes = list(itertools.product(range(1, 7), repeat=n))
    assert len(outcomes) == 6 ** n
    for k in range(1, 7):
        cnt = sum(1 for o in outcomes if max(o) <= k)
        assert Fraction(cnt, 6 ** n) == Fraction(k, 6) ** n                       # P(M_n <= k) = (k/6)^n
        assert D.cdf_max("wuerfel", float(k), n) == pytest.approx(cnt / 6 ** n, rel=1e-14)
    p6 = Fraction(sum(1 for o in outcomes if max(o) == 6), 6 ** n)
    assert p6 == 1 - Fraction(5, 6) ** n
    assert 1.0 - float(D.cdf_max("wuerfel", 5.0, n)) == pytest.approx(float(p6), rel=1e-14)


def test_wuerfel_erwartungswerte_161_36_und_119_24():
    assert Fraction(sum(max(o) for o in itertools.product(range(1, 7), repeat=2)), 36) == Fraction(161, 36)
    assert Fraction(sum(max(o) for o in itertools.product(range(1, 7), repeat=3)), 216) == Fraction(119, 24)
    assert D.mean_max("wuerfel", 2) == pytest.approx(161 / 36, rel=1e-14)
    assert D.mean_max("wuerfel", 3) == pytest.approx(119 / 24, rel=1e-14)
    assert Fraction(11, 36) == 1 - Fraction(25, 36)
    assert D.mean_max("wuerfel", 2) > 3.5                                           # größer als ein einzelner Wurf (7/2)


# --- 2. Gleichverteilung ----------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("n", [1, 2, 5, 10])
def test_gleich_mittel_n_durch_n_plus_1_gegen_quadratur(n):
    from scipy import integrate
    q, _ = integrate.quad(lambda x: 1 - x ** n, 0, 1)
    assert D.mean_max("gleich", n) == pytest.approx(n / (n + 1), rel=1e-14)
    assert D.mean_max("gleich", n) == pytest.approx(q, abs=1e-12)


def test_gleich_n10_z_minus_1():
    assert float(D.normed_cdf("gleich", 10, -1.0)) == pytest.approx(0.9 ** 10, rel=1e-13)       # (1 + z/n)^n = 0,9^10 = 0,34868
    assert 0.9 ** 10 == pytest.approx(0.3486784401, abs=1e-12)
    assert math.exp(-1) - 0.9 ** 10 == pytest.approx(0.019201, abs=1e-6)                       # Differenz zu e^-1
    assert float(D.limit_cdf("gleich", -1.0)) == pytest.approx(math.exp(-1), rel=1e-14)


# --- 3. Exponential: Abstand von Hand ---------------------------------------------------------------------------------------------------------
def test_exponential_n1_abstand_ist_e_hoch_minus_1():
    assert X.common_distance(1) == pytest.approx(math.exp(-1), abs=1e-12)


def test_exponential_n2_u_stern_und_abstand_016190256():
    us = optimize.brentq(lambda t: math.exp(-t) - (1 - t / 2), 0.5, 1.99)               # f' = 0  <=>  e^-u = 1 - u/2
    assert us == pytest.approx(1.5936242600, abs=1e-9)
    f_star = math.exp(-us) - (1 - us / 2) ** 2
    assert f_star == pytest.approx(math.exp(-us) * (1 - math.exp(-us)), abs=1e-14)       # bei u*: f = e^-u (1 − e^-u)
    assert f_star == pytest.approx(0.16190256, abs=1e-8)
    assert math.exp(-2) < f_star                                                       # der Randwert u = 2 (0,1353) ist kleiner
    assert X.common_distance(2) == pytest.approx(f_star, abs=1e-9)
    # zweiter Weg: direkt an F² mit feinem Gitter in x (scipy), nicht über u
    assert ks_exp_direct(2, 1_000_001) == pytest.approx(f_star, abs=2e-8)


def test_abstand_an_einer_stelle_liegt_unter_dem_supremum():
    stelle = math.exp(-1) - 0.25                                                       # x = 0: F²(ln 2) = 1/4 gegen e^-1
    assert float(D.normed_cdf("exp", 2, 0.0)) == pytest.approx(0.25, abs=1e-15)
    assert stelle == pytest.approx(0.117879, abs=1e-6) and stelle < X.common_distance(2)


def test_n_mal_abstand_geht_gegen_2_durch_e_quadrat():
    assert 2 / math.e ** 2 == pytest.approx(0.27067057, abs=1e-8)
    us = optimize.minimize_scalar(lambda t: -(t * t * math.exp(-t)), bounds=(0.5, 6), method="bounded", options=dict(xatol=1e-10)).x
    assert us == pytest.approx(2.0, abs=1e-6)                                          # Maximum von u² e^-u bei u = 2
    assert 10_000 * X.common_distance(10_000) == pytest.approx(2 / math.e ** 2, abs=2e-4)
    assert 10 ** 6 * X.common_distance(10 ** 6) == pytest.approx(2 / math.e ** 2, abs=2e-6)


def test_exponential_momente_renyi_gegen_quadratur():
    from scipy import integrate
    for n in (1, 2, 3, 10):
        H = sum(Fraction(1, k) for k in range(1, n + 1))
        V = sum(Fraction(1, k * k) for k in range(1, n + 1))
        m1, _ = integrate.quad(lambda t: 1 - (1 - math.exp(-t)) ** n, 0, 60, limit=200)
        m2, _ = integrate.quad(lambda t: 2 * t * (1 - (1 - math.exp(-t)) ** n), 0, 60, limit=200)
        assert float(H) == pytest.approx(m1, abs=1e-9)
        assert float(V) == pytest.approx(m2 - m1 ** 2, abs=1e-8)
        assert D.mean_max("exp", n) == pytest.approx(float(H), rel=1e-14)
    assert sum(Fraction(1, k) for k in range(1, 11)) == Fraction(7381, 2520)
    assert float(Fraction(7381, 2520)) - (math.log(10) + D.EULER_GAMMA) == pytest.approx(0.04917, abs=1e-4)
    assert D.harmonic(20) == pytest.approx(3.597739657143682, rel=1e-14)
    assert sum(Fraction(1, k) for k in range(1, 21)) == Fraction(55835135, 15519504)


# --- 4. Normal --------------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("n", [1, 2, 3, 5])
def test_normal_p_maximum_unter_null_ist_zwei_hoch_minus_n(n):
    assert Fraction(1, 2) ** n == Fraction(1, 2 ** n)
    assert float(D.cdf_max("normal", 0.0, n)) == pytest.approx(2.0 ** -n, rel=1e-14)


def _moment_from_cdf(n, power):
    """E[M^power] aus der Verteilungsfunktion der App per Trapezregel (Dichte als Differenz), unabhängig von normal_mean_max."""
    x = np.linspace(-10, 12, 400001)
    F = D.cdf_max("normal", x, n)
    dens = np.diff(F) / np.diff(x)
    mid = 0.5 * (x[1:] + x[:-1])
    return float(np.sum(mid ** power * dens * np.diff(x)))


def test_normal_n2_und_n3_in_geschlossener_form():
    m1 = _moment_from_cdf(2, 1)
    assert m1 == pytest.approx(1 / math.sqrt(math.pi), abs=1e-7)                      # E[M_2] = 1/√π
    assert _moment_from_cdf(2, 2) - m1 ** 2 == pytest.approx(1 - 1 / math.pi, abs=1e-7)   # Var[M_2] = 1 − 1/π
    assert _moment_from_cdf(3, 1) == pytest.approx(3 / (2 * math.sqrt(math.pi)), abs=1e-7)  # E[M_3] = 3/(2√π)
    assert D.mean_max("normal", 2) == pytest.approx(1 / math.sqrt(math.pi), abs=1e-9)
    assert D.mean_max("normal", 3) == pytest.approx(3 / (2 * math.sqrt(math.pi)), abs=1e-9)


def test_pegel_n365_exakt_und_gumbel_a_und_b_von_hand():
    n, p = 365, 0.01
    x_exact = normal_pegel_bisection(n, p)                                             # nur math: Φ(x)^365 = 0,99
    assert x_exact == pytest.approx(4.0330, abs=5e-4)
    a = E.analyse_pegel(n, p)
    assert a.exact == pytest.approx(x_exact, abs=1e-6)
    b_n = bisect(lambda x: 1 - Phi(x) - 1 / n, 1.0, 8.0)                               # b = Φ⁻¹(1 − 1/n)
    a_n = 1 / (n * math.exp(-b_n ** 2 / 2) / math.sqrt(2 * math.pi))
    y = -math.log(-math.log(0.99))
    assert b_n == pytest.approx(2.7774, abs=5e-4) and a_n == pytest.approx(0.3250, abs=5e-4)
    assert b_n + a_n * y == pytest.approx(4.2724, abs=5e-4) and a.levels["gumbelB"] == pytest.approx(b_n + a_n * y, abs=1e-7)
    L = math.log(n)
    c = (2 * L) ** -0.5
    d = math.sqrt(2 * L) - (math.log(L) + math.log(4 * math.pi)) / (2 * math.sqrt(2 * L))
    assert d + c * y == pytest.approx(4.1475, abs=5e-4) and a.levels["gumbelA"] == pytest.approx(d + c * y, abs=1e-7)
    assert a.exact < a.levels["gumbelA"] < a.levels["gumbelB"]
    p_true = 1 - math.exp(n * math.log(Phi(b_n + a_n * y)))
    assert p_true == pytest.approx(0.00352, abs=5e-5) and a.true_p["gumbelB"] == pytest.approx(p_true, abs=1e-8)


def test_normierung_a_ist_nicht_monoton_n4_gegen_n16():
    k = {n: X.normal_distance(n, "A", 100_001) for n in (4, 16, 17, 100, 621, 622)}
    assert k[4] == pytest.approx(0.04871, abs=2e-5)
    assert k[16] == pytest.approx(0.06574, abs=2e-5)
    assert k[4] < k[16] and k[100] < k[16]                                             # steigt von n = 4 an, fällt später
    assert k[17] >= k[16]                                                              # das Maximum liegt bei n = 17
    assert k[621] >= 0.05 > k[622]                                                     # dauerhaft unter 0,05 erst ab n = 622


def test_bestes_gev_zeugen_n2_und_n3_unter_dem_lokalen_optimum():
    # Parameter aus der Suche (Differential Evolution) auf der Skala der Einzelwerte, hier nur eingesetzt: jede GEV ist eine obere Schranke des besten Abstands
    x = np.linspace(-8, 12, 400001)
    ks2 = float(np.max(np.abs(D.cdf_max("normal", x, 2) - D.gev_cdf(x, 0.262616, 0.794478, -0.242457))))
    ks3 = float(np.max(np.abs(D.cdf_max("normal", x, 3) - D.gev_cdf(x, 0.566287, 0.71081, -0.222299))))
    assert ks2 < 0.0037 < 0.01411
    assert ks3 < 0.0034 < 0.00368
    assert 0.0036 < ks2 and 0.0033 < ks3


# --- 5. Fréchet und Pareto-Verteilung ---------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("n,z", [(1, 0.7), (2, 1.3), (7, 0.4), (365, 3.0)])
def test_standard_frechet_ist_max_stabil(n, z):
    F = lambda t: math.exp(-1 / t)
    assert F(n * z) ** n == pytest.approx(F(z), rel=1e-12)
    assert float(D.normed_cdf("frechet", n, z)) == pytest.approx(F(z), rel=1e-12)
    assert F(6.0) ** 3 == pytest.approx(F(2.0), rel=1e-14)


def test_pareto_n2_z1_ist_ein_viertel_und_abstand_wie_beim_exponential():
    assert (1 - 0.5) ** 2 == 0.25
    assert ks_pareto_direct(2, 2.0, 1_000_001) == pytest.approx(X.common_distance(2), abs=5e-8)


@pytest.mark.parametrize("alpha", [1.5, 2.0, 4.0])
def test_xi_ist_ein_durch_alpha_ueber_die_gev_formel(alpha):
    zg = np.linspace(0.2, 30.0, 5000)
    xi = 1.0 / alpha
    gev = D.gev_cdf(zg, 1.0, xi, xi)                                                    # μ = 1, σ = ξ = 1/α: 1 + ξ(z − 1)/σ = z
    assert float(np.max(np.abs(gev - np.exp(-zg ** -alpha)))) < 1e-13
    assert D.type_info("pareto2" if alpha == 2.0 else ("pareto15" if alpha == 1.5 else "pareto4"))["xi"] == pytest.approx(xi)


@pytest.mark.parametrize("key,alpha", [("pareto15", 1.5), ("pareto2", 2.0), ("pareto4", 4.0)])
def test_pareto_n_eine_million_abstand_hoechstens_027_durch_n(key, alpha):
    n = 10 ** 6
    z = np.linspace(0.5, 20.0, 2000)
    b, a = D.norming(key, n)
    fn = D.cdf_max(key, a * z, n)
    assert float(np.max(np.abs(fn - np.exp(-z ** (-alpha))))) < 0.2707 / n * 1.01


# --- 6. Verweis Bootstrap ---------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("n,bruch", [(3, Fraction(19, 27)), (4, Fraction(175, 256))])
def test_bootstrap_anteil_gleiches_maximum_durch_aufzaehlung(n, bruch):
    draws = list(itertools.product(range(n), repeat=n))
    assert len(draws) == n ** n
    cnt = sum(1 for d in draws if (n - 1) in d)                                        # enthält die Stelle des größten Wertes
    assert Fraction(cnt, n ** n) == bruch == 1 - Fraction(n - 1, n) ** n


def test_bootstrap_n50_ist_636_prozent_und_wiederkehrperiode_nahe_632():
    assert 1 - 0.98 ** 50 == pytest.approx(0.6358, abs=5e-5)
    assert 1 - (1 - 1 / 365) ** 365 == pytest.approx(1 - math.exp(-1), abs=2e-3)


# --- Gegenprobe zu den Gittern: Exponential, Gleich, Pareto-Verteilung haben denselben Abstand -------------------------------------------------
@pytest.mark.parametrize("n", [1, 2, 5, 20, 365])
def test_drei_verteilungen_gleicher_abstand_an_f_hoch_n_direkt(n):
    d = X.common_distance(n)
    assert ks_exp_direct(n, 1_000_001) == pytest.approx(d, abs=1e-7)
    assert ks_gleich_direct(n, 1_000_001) == pytest.approx(d, abs=1e-7)
    assert ks_pareto_direct(n, 2.0, 1_000_001) == pytest.approx(d, abs=1e-7)

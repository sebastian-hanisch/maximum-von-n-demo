"""Literaturtest: Beispiele 3.1 bis 3.3 bei Coles (2001), nur die Befunde, nichts wörtlich.

Fundstelle: Coles, S. (2001), An Introduction to Statistical Modeling of Extreme Values, Springer, Abschnitt 3.1, Beispiele 3.1 bis 3.3, S. 45 bis 52.
Stand der Einsicht: Quellenstufe W (gelesen anhand einer Kopie von einer Mitarbeiter-Seite, Weitergaberechte unklar, deshalb hier keine Textstellen), eingetragen am 2026-11-19.
Nicht getestet: Fisher/Tippett 1928 (nur Abstract), Hall 1979 (nur Zusammenfassung), Gnedenko 1943, Jenkinson 1955 (nicht gelesen). Die Normal-Konstanten der Normierung A
werden nicht aus dem Buch übernommen (Errata, de Haan/Ferreira), sondern in `test_oracle_maximum.py` gegen Φⁿ selbst geprüft."""

import math

import numpy as np
import pytest

import mxn_distributions as D


@pytest.mark.parametrize("n", [1, 2, 10, 100, 1000])
def test_beispiel_3_1_exponential_aN_gleich_1_bN_gleich_ln_n(n):
    b, a = D.norming("exp", n)
    assert (b, a) == (math.log(n), 1.0)                                          # Verschiebung ln n, Skala 1
    z = np.linspace(-2.0, 6.0, 81)
    exakt = D.normed_cdf("exp", n, z)
    grenze = D.limit_cdf("exp", z)
    assert np.max(np.abs(exakt - grenze)) <= 0.3679 / n * 1.0001 + 1e-12      # Abstand höchstens e⁻¹/n bei n = 1 und kleiner danach: nur die Größenordnung 1/n


def test_beispiel_3_1_exponential_exakt_formel_1_minus_e_hoch_minus_z_durch_n_hoch_n():
    n, z = 50, 1.3
    assert float(D.normed_cdf("exp", n, z)) == pytest.approx((1 - math.exp(-z) / n) ** n, rel=1e-13)


@pytest.mark.parametrize("n,z", [(1, 0.7), (2, 1.3), (10, 0.4), (365, 3.0)])
def test_beispiel_3_2_standard_frechet_ist_fuer_jedes_n_exakt(n, z):
    assert float(D.normed_cdf("frechet", n, z)) == pytest.approx(math.exp(-1 / z), rel=1e-12)
    b, a = D.norming("frechet", n)
    assert (b, a) == (0.0, float(n))                                             # a_n = n, b_n = 0
    assert D.type_info("frechet")["family"] == "frechet"


@pytest.mark.parametrize("n", [2, 10, 100])
def test_beispiel_3_3_gleich_hat_xi_minus_1_und_skala_1_durch_n(n):
    assert D.type_info("gleich") == {"typ": "Weibull-Typ", "xi": -1.0, "family": "weibull"}
    b, a = D.norming("gleich", n)
    assert (b, a) == (1.0, 1.0 / n)
    z = np.linspace(-min(5.0, float(n)), 0.0, 51)                          # z >= −n: links davon ist der Träger zu Ende
    assert D.normed_cdf("gleich", n, z) == pytest.approx((1 + z / n) ** n, rel=1e-12)
    assert D.limit_cdf("gleich", z) == pytest.approx(np.exp(z), rel=1e-14)       # Grenzgesetz e^z für z ≤ 0

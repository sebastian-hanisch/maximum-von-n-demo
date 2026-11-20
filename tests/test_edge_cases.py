"""Randfälle: n = 1, n = 2, n = 10¹², R = 1 und R = 0, Pareto-Verteilung und Fréchet ohne Mittelwert, p = 0 oder 1, p = 10⁻¹⁰, n nicht ganzzahlig, Eingaben außerhalb der Grenzen."""

import math

import numpy as np
import pytest
from scipy import stats

import mxn_constants as C
import mxn_distance as X
import mxn_distributions as D
import mxn_evaluation as E
import mxn_presets as P
import mxn_reference as REF
import mxn_scenario as S


def test_n_gleich_1_das_maximum_ist_der_einzelwert():
    x = np.array([0.3, 1.0, 2.5])
    for key in ("exp", "gleich", "pareto2", "frechet", "normal"):
        assert D.cdf_max(key, x, 1) == pytest.approx(np.exp(D.log_cdf(key, x)), rel=1e-14)
    assert X.common_distance(1) == pytest.approx(math.exp(-1), abs=1e-12)
    sim = S.simulate_blocks("exp", 1, 2000, 9)
    assert S.ks_statistic(sim.values, lambda t: D.cdf_max("exp", t, 1)) < 0.05


def test_n_gleich_2_exakte_werte():
    assert X.common_distance(2) == pytest.approx(0.16190256, abs=1e-8)
    assert float(D.cdf_max("gleich", 0.5, 2)) == 0.25
    assert D.mean_max("gleich", 2) == pytest.approx(2 / 3, rel=1e-14)


def test_n_zehn_hoch_12_normal_exakt_und_abstand_endlich():
    n = 10 ** 12
    assert D.quantile_max("normal", n, 0.01) == pytest.approx(float(stats.norm.isf(-math.expm1(math.log1p(-0.01) / n))), abs=1e-9)
    d = X.normal_distance(n, "B")
    assert 0.0 < d < 0.01 and math.isfinite(d)
    assert X.common_distance(n) * n == pytest.approx(2 / math.e ** 2, rel=1e-5)


def test_r_gleich_1_und_r_gleich_0():
    sim = S.simulate_blocks("exp", 20, 1, 3)
    assert sim.R == 1 and sim.values.shape == (1,)
    a = E.analyse_sim(sim)
    assert a.se is None and math.isfinite(a.ks)
    with pytest.raises(ValueError, match="R muss"):
        S.simulate_blocks("exp", 20, 0, 3)
    with pytest.raises(ValueError):
        S.block_vmin(20, 0, 3)


def test_budget_begrenzt_r_bei_n_365_auf_5479():
    assert S.effective_R(365, 20000) == 5479
    assert S.effective_R(20, 20000) == 20000
    assert S.effective_R(1, 50000) == 50000
    sim = S.simulate_blocks("exp", 365, 20000, 1)
    assert sim.R == 5479 and sim.R_wish == 20000 and sim.clipped


def test_n_nicht_ganzzahlig_oder_kleiner_1_meldet_fehler():
    for bad in (0, 0.5, 2.5, -3):
        with pytest.raises(ValueError):
            S.effective_R(bad, 1000)
    with pytest.raises(ValueError):
        X.common_distance(0.5)
    with pytest.raises(ValueError):
        D.normal_constants_B(1)
    with pytest.raises(ValueError):
        D.normal_constants_A(1)
    with pytest.raises(ValueError):
        D.mean_max("exp", 2.5)


def test_fraechet_hat_keinen_mittelwert_und_das_verdikt_zeigt_den_median():
    assert D.mean_max("frechet", 20) is None
    sim = S.simulate_blocks("frechet", 20, 2000, 4)
    kind, text = E.verdict_sim(E.analyse_sim(sim))
    assert "keinen Mittelwert" in text and "Median" in text


def test_pareto_alpha_1_5_hat_mittelwert_aber_keine_varianz_und_kein_standardfehler():
    a = E.analyse_sim(S.simulate_blocks("pareto15", 20, 2000, 4))
    assert a.mean_exact is not None and a.se is None
    kind, text = E.verdict_sim(a)
    assert "Varianz ist unendlich" in text and "Median" in text


def test_pegel_p_null_und_eins_melden_fehler_und_p_1e_minus_10_ist_endlich():
    for p in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            D.quantile_max("normal", 365, p)
        with pytest.raises(ValueError):
            E.analyse_pegel(365, p)
    x = D.quantile_max("normal", 365, 1e-10)
    assert math.isfinite(x) and x > D.quantile_max("normal", 365, 1e-4)
    assert E.level_true_p(365, x) == pytest.approx(1e-10, rel=1e-4)


def test_wuerfel_hat_keinen_pegel_und_kein_grenzgesetz():
    with pytest.raises(ValueError):
        D.quantile_max("wuerfel", 3, 0.1)
    with pytest.raises(ValueError):
        X.distance("wuerfel", 10)
    kind, text = E.verdict_type("wuerfel", 10)
    assert "kein Grenzgesetz" in text


def test_normal_n_gleich_1_im_experiment_3_meldet_statt_zu_rechnen():
    kind, text = E.verdict_type("normal", 1)
    assert "n ≥ 2" in text


def test_eingaben_ausserhalb_der_reglergrenzen_werden_begrenzt_oder_verworfen():
    spec = P.SETTING_SPECS["n_slider"]
    assert P.limit_value(spec, 0) == C.N_MIN and P.limit_value(spec, 10 ** 9) == C.N_MAX
    assert P.limit_value(P.SETTING_SPECS["R_slider"], 5) == C.R_MIN
    assert P.limit_value(P.SETTING_SPECS["dist"], "unsinn") is None
    assert P.limit_value(P.SETTING_SPECS["n_lim"], 12345) is None
    assert P.limit_value(P.SETTING_SPECS["p_exc"], 0.5) is None
    assert P.limit_value(P.SETTING_SPECS["p_exc"], 0.01) == 0.01


def test_nicht_eingefrorenes_n_fuer_den_pegel_ist_ein_fehler():
    with pytest.raises(KeyError):
        REF.best_fit(1234)
    for n in C.N_PEGEL_OPTIONS:
        assert REF.best_fit(n)["gev"]["xi"] < 0


def test_ks_statistik_bei_einem_wert_und_diskret():
    assert S.ks_statistic([0.5], lambda x: np.asarray(x) * 1.0) == pytest.approx(0.5)
    sim = S.simulate_blocks("wuerfel", 1, 6000, 8)
    assert E.ks_to_exact(sim) < 0.05


def test_normal_n_gleich_1_hat_keine_grenzkurve_in_experiment_1():
    """Regression: die Grenzkurve der Normalverteilung braucht Normierung B oder A und damit n >= 2; bei n = 1 wird sie weggelassen statt abzustürzen."""
    x = np.linspace(-3, 3, 11)
    assert E.limit_curve("normal", 1, x) is None and E.limit_curve("wuerfel", 5, x) is None
    assert E.limit_curve("normal", 2, x) is not None

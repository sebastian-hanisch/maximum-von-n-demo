"""Verdikte (reine Funktionen in mxn_evaluation.py) an handgesetzten Werten: Schwellen und jede Zahl im Text."""

import numpy as np
import pytest

import mxn_distance as X
import mxn_evaluation as E
import mxn_reference as REF
import mxn_scenario as S


def _sim_analysis(key="exp", n=20, R=20000, ks=0.0049, eps=0.0115, mean_sim=3.5889, mean_exact=3.5977, se=0.0089, median_sim=2.9, median_exact=2.95):
    sim = S.Simulation(key, n, R, R, 35, np.array([1.0, 2.0]), 0.1)
    return E.SimAnalysis(sim, ks, eps, mean_sim, mean_exact, se, median_sim, median_exact, 0.0115)


def test_verdikt_simulation_unter_der_schranke_ist_success_mit_allen_zahlen():
    kind, text = E.verdict_sim(_sim_analysis())
    assert kind == "success"
    for z in ("Exponential(1)", "n = 20", "R = 20.000", "0,0049", "unter der Schranke 0,0115", "Mittel 3,5889", "Hₙ = 3,5977", "Standardfehler 0,0089"):
        assert z in text, z


def test_verdikt_simulation_ueber_der_schranke_ist_warning_und_sagt_es():
    kind, text = E.verdict_sim(_sim_analysis(ks=0.0135))
    assert kind == "warning" and "über der Schranke 0,0115" in text and "1 von 100" in text


def test_verdikt_schwelle_genau_auf_der_schranke_gilt_als_darunter():
    kind, text = E.verdict_sim(_sim_analysis(ks=0.0115, eps=0.0115))
    assert kind == "success" and "unter der Schranke" in text


def test_verdikt_ohne_mittelwert_zeigt_den_median():
    kind, text = E.verdict_sim(_sim_analysis(key="frechet", mean_exact=None, se=None, median_sim=28.8002, median_exact=28.8539))
    assert "keinen Mittelwert" in text and "Median 28,8002 gegen exakt 28,8539" in text and "Mittel " not in text.replace("Mittelwert", "")


def test_verdikt_ohne_standardfehler_nennt_die_unendliche_varianz():
    kind, text = E.verdict_sim(_sim_analysis(key="pareto2", se=None, mean_exact=7.9763, mean_sim=7.8778, median_sim=6.0, median_exact=6.1))
    assert "Varianz ist unendlich" in text and "Mittel 7,8778 gegen exakt 7,9763" in text and "Median 6,0000 gegen exakt 6,1000" in text


def test_verdikt_grenzgesetz_nennt_normierung_faktor_und_schwellen():
    a = E.LimitAnalysis(365, "B", 0.000742241, 0.021334324, 0.052394610)
    kind, text = E.verdict_limit(a)
    assert kind == "info"
    for z in ("n = 365", "0,00074", "Normierung B) 0,0213", "Faktor 28,7", "Normierung A 0,0524 (Faktor 70,6)", "ab n = 28", "2,86·10⁵", "nicht bis 10³⁰", "6 gegen 897", "R ≈ 4.800.000 Blöcke", "Rauschschranke 1,628/√R (99 %)"):
        assert z in text, z
    assert a.factor("B") == pytest.approx(28.74, abs=0.01) and a.factor("A") == pytest.approx(70.59, abs=0.01)


def test_verdikt_grenzgesetz_bei_n_eine_million_ohne_scheingenaue_stellen():
    """Bei n = 10⁶ ist der gemeinsame Abstand 2,7·10⁻⁷: keine Anzeige „0,00000“, keine 14-stellige Blockzahl, kein Faktor mit Nachkommastelle ab 1000."""
    kind, text = E.verdict_limit(E.analyse_limit(10 ** 6, "B"))
    assert "0,00000" not in text and "2,7·10⁻⁷" in text
    assert "R ≈ 3,6·10¹³ Blöcke" in text and "Faktor 3,36·10⁴" in text
    assert text.count("Rauschschranke") == 1


def test_grenzgesetz_faktor_folgt_der_gewaehlten_normierung():
    a = E.LimitAnalysis(365, "A", 0.000742241, 0.021334324, 0.052394610)
    assert a.d_sel == 0.052394610 and a.factor() == pytest.approx(70.59, abs=0.01)
    assert a.see_normal < a.see_common


def test_verdikt_pegel_n365_p001_gumbel_b_und_gev():
    a = E.analyse_pegel(365, 0.01)
    kind, text = E.verdict_pegel(a, "gumbelB")
    assert kind == "warning"
    for z in ("n = 365", "p = 0,01", "exakt 4,033", "4,272", "+0,239", "über dem exakten Pegel", "0,0035 statt 0,01", "Gumbel A +0,114", "ξ̂ = −0,09"):
        assert z in text, z
    kind, text = E.verdict_pegel(a, "gevbest")
    assert kind == "success" and "−0,020" in text and "näher am exakten Pegel als die drei Gumbel-Verfahren" in text and "kein Schätzer aus Daten" in text
    assert "Langsamkeit" not in text


def test_verdikt_pegel_unter_dem_exakten_wert_sagt_unter():
    a = E.analyse_pegel(365, 0.01)
    fake = E.PegelAnalysis(a.n, a.p, a.exact, {**a.levels, "gumbelA": a.exact - 0.2}, a.true_p, a.xi_gev)
    kind, text = E.verdict_pegel(fake, "gumbelA")
    assert "unter dem exakten Pegel" in text and "−0,200" in text


def test_verdikt_typ_je_verteilung():
    kind, text = E.verdict_type("exp", 365)
    assert "Gumbel-Typ (ξ = 0)" in text and "Mₙ − ln n" in text and "n·Abstand = 0,2709" in text
    kind, text = E.verdict_type("gleich", 20)
    assert "Weibull-Typ (ξ = −1)" in text and "nicht die Weibull-Verteilung" in text
    kind, text = E.verdict_type("pareto4", 20)
    assert "Fréchet-Typ (ξ = 0,25)" in text and "n^(1/4)" in text
    kind, text = E.verdict_type("frechet", 50)
    assert kind == "success" and "max-stabil" in text and "0,0000" in text
    kind, text = E.verdict_type("normal", 365, "A")
    assert "Normierung A" in text and "Abstand·ln n = 0,309" in text
    assert "wird mit wachsendem n am langsamsten der vier (im Abstand ab n = 4 bei Normierung B, ab n = 6 bei Normierung A)" in text and "die langsamste der vier" not in text


def test_normal_ist_im_abstand_erst_ab_n4_bzw_n6_weiter_weg_als_die_gemeinsame_kurve():
    """Belegt den Zusatz „im Abstand ab n = 4 (B) bzw. n = 6 (A)“ im Verdikt und in TYPE_NOTES (Gitter 20.001, n bis 400)."""
    gem = {n: X.common_distance(n) for n in range(2, 401)}
    assert [n for n in range(2, 401) if X.normal_distance(n, "B", 20001) <= gem[n]] == [2, 3]
    assert [n for n in range(2, 401) if X.normal_distance(n, "A", 20001) <= gem[n]] == [2, 3, 4, 5]
    assert "ab n = 4 bei Normierung B, ab n = 6 bei Normierung A" in E.TYPE_NOTES["normal"]


def test_pegel_tabelle_hat_fuenf_zeilen_mit_verfahren_und_wahrer_wahrscheinlichkeit():
    rows = E.pegel_rows(E.analyse_pegel(365, 0.01))
    assert [r["Verfahren"] for r in rows][0] == "exakt (aus Φⁿ)" and len(rows) == 5
    assert rows[2]["Differenz"] == "+0,239" and rows[2]["wahre Wahrscheinlichkeit am Pegel"] == "0,00352"


def test_schwellentabelle_zeilen_aus_der_referenz():
    rows = {r["Abstand unter"]: r for r in E.threshold_rows(REF.load())}
    assert rows["0,01"]["Exponential, Gleich, Pareto (Standardnormierung)"] == "n ≥ 28"
    assert rows["0,01"]["Normal, Normierung B"] == "n ≥ 2,86·10⁵"
    assert rows["0,01"]["Normal, Normierung A"] == "nicht bis 10³⁰"
    assert rows["0,05"]["Normal, Normierung A"] == "erstes n = 4, dauerhaft n ≥ 622"
    assert rows["0,001"]["Normal, Normierung B"] == "nicht bis 10¹⁴"
    assert rows["0,01"]["Normal, bestes Gumbel"] == "n ≥ 897"

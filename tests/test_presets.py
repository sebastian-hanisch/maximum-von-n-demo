"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen, die Aussagen der Hilfetexte und je Preset die erwartete Verdikt-Zeile."""

from fractions import Fraction
from functools import lru_cache

import pytest

import mxn_constants as C
import mxn_distributions as D
import mxn_evaluation as E
import mxn_presets as P
import mxn_scenario as S


@lru_cache(maxsize=None)
def _analysis(name):
    s = _settings(name)
    return E.analyse_sim(S.simulate_blocks(s.dist, s.n, s.R, s.seed))


def _settings(name) -> E.Settings:
    p = C.PRESETS[name]
    return E.Settings(p["dist"], p["n"], p["R"], p["seed"], p["n_lim"], p["norm"], p["n_pegel"], p["p"], p["method"])


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_slider_bounds_and_choices():
    for name, preset in C.PRESETS.items():
        for key, state_key in P.PRESET_KEYS.items():
            assert P.limit_value(P.SETTING_SPECS[state_key], preset[key]) == preset[key], (name, key)


def test_default_preset_equals_the_default_settings():
    assert _settings(C.STANDARD_PRESET) == E.Settings()
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_presets_unterscheiden_sich_paarweise():
    settings = [tuple(sorted(p.items())) for p in C.PRESETS.values()]
    assert len(set(settings)) == len(settings)


def test_preset_namen_passen_zu_den_werten():
    assert C.PRESETS["Würfel: das Größte von 2"]["dist"] == "wuerfel" and C.PRESETS["Würfel: das Größte von 2"]["n"] == 2
    assert C.PRESETS["Gleich, n = 20 (Weibull-Typ)"]["dist"] == "gleich" and D.type_info("gleich")["typ"] == "Weibull-Typ"
    assert C.PRESETS["Pareto-Verteilung, n = 20 (Fréchet-Typ)"]["dist"] == "pareto2" and D.type_info("pareto2")["typ"] == "Fréchet-Typ"
    assert C.PRESETS["Exponential, n = 20"]["n"] == 20 and C.PRESETS["Normal, n = 365: 28,7-mal größerer Abstand"]["dist"] == "normal"
    for p in C.PRESETS.values():
        assert p["n"] <= C.N_MAX and p["R"] * p["n"] <= C.SIM_BUDGET        # kein Preset wird durch die Begrenzung n·R verändert (n = 365: R = 5 000)


def test_hilfetexte_der_presets_stimmen_mit_der_rechnung():
    assert Fraction(161, 36) == sum(Fraction(max(a, b)) for a in range(1, 7) for b in range(1, 7)) / 36 and 161 / 36 == pytest.approx(4.47, abs=0.005)
    lim = E.analyse_limit(365, "B")
    assert round(lim.d_B, 4) == 0.0213 and round(lim.d_common, 5) == 0.00074 and round(lim.factor("B"), 1) == 28.7
    assert round(E.analyse_limit(365, "A").factor("A"), 1) == 70.6
    assert "0,0213" in C.PRESET_HELP["Normal, n = 365: 28,7-mal größerer Abstand"] and "28,7" in C.PRESET_HELP["Normal, n = 365: 28,7-mal größerer Abstand"]
    assert "70,6" in C.PRESET_HELP["Normal gegen Exponential, gleiche n"]
    assert "ξ = 1/2" in C.PRESET_HELP["Pareto-Verteilung, n = 20 (Fréchet-Typ)"] and D.type_info("pareto2")["xi"] == 0.5


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_je_preset_die_erwartete_verdikt_zeile_der_simulation(name):
    s = _settings(name)
    a = _analysis(name)
    kind, text = E.verdict_sim(a)
    assert kind == "success", text                                              # das Preset meldet „unter der Schranke“, und das trifft zu
    assert E.label(s.dist) in text and f"n = {s.n}" in text and "unter der Schranke" in text
    assert a.ks <= a.eps
    exakt = D.mean_max(s.dist, s.n)
    assert exakt is not None
    if a.se is not None:
        assert abs(a.mean_sim - exakt) < 4 * a.se                                # Mittel innerhalb von 4 Standardfehlern
    else:
        assert abs(a.median_sim - a.median_exact) < 0.05 * a.median_exact          # Varianz unendlich: der Median ist das Maß


def test_preset_wuerfel_nennt_den_mittelwert_4_47():
    a = _analysis("Würfel: das Größte von 2")
    kind, text = E.verdict_sim(a)
    assert "gegen exakt 4,4722" in text and a.mean_exact == pytest.approx(161 / 36)


def test_preset_pareto_und_normal_365_zeilen():
    s = _settings("Normal, n = 365: 28,7-mal größerer Abstand")
    assert E.verdict_sim(_analysis("Normal, n = 365: 28,7-mal größerer Abstand"))[1].count("R = 5.000") == 1       # R = 5 000 liegt unter der Grenze 5 479
    kind, text = E.verdict_limit(E.analyse_limit(s.n_lim, s.norm))
    assert "Faktor 28,7" in text
    kind, text = E.verdict_pegel(E.analyse_pegel(s.n_pegel, s.p), s.method)
    assert kind == "warning" and "+0,239" in text and "0,0035 statt 0,01" in text
    s2 = _settings("Normal gegen Exponential, gleiche n")
    kind, text = E.verdict_pegel(E.analyse_pegel(s2.n_pegel, s2.p), s2.method)
    assert kind == "success" and "beste GEV" in text


def test_kein_preset_ohne_wirkung_meldet_eine_wirkung():
    """Presets mit gleicher Verteilung und gleichem n unterscheiden sich im Verdikt-Text (R, Zufalls-Seed oder Normierung)."""
    texts = set()
    for name in C.PRESETS:
        s = _settings(name)
        lim = E.verdict_limit(E.analyse_limit(s.n_lim, s.norm))[1]
        texts.add((E.verdict_sim(_analysis(name))[1], lim))
    assert len(texts) == len(C.PRESETS)

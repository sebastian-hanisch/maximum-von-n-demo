"""Eingefrorene Tabellen `mxn_reference.json`: Aufbau (Quelle und Stand je Abschnitt), Gleichheit mit der Neu-Rechnung, Zeugen für „bestes Gumbel/GEV“."""

import json

import mxn_make_reference as TOOL
import pytest

import mxn_constants as C
import mxn_distance as X
import mxn_reference as REF


def test_jeder_abschnitt_nennt_quelle_und_stand(ref):
    for name in ("distance_tables", "curve", "thresholds", "exp_best_gumbel", "best_fit", "bands"):
        assert ref[name].get("quelle") and ref[name].get("stand"), name
    assert ref["meta"]["stand"] and ref["meta"]["python"] and ref["meta"]["scipy"]


def test_json_ist_gueltig_und_ohne_nan():
    text = REF.PATH.read_text(encoding="utf-8")
    assert "NaN" not in text and "Infinity" not in text
    assert json.loads(text)["meta"]["stand"] == "2026-11-20"


def test_neu_rechnen_der_schnellen_teile_stimmt_mit_der_datei_ueberein():
    """`tools/mxn_make_reference.py --check`: n <= 365, ganzzahlige Schwellen, Normal-B-Schwellen, gespeicherte Parameter (relative Toleranz, kein Test auf das letzte Bit)."""
    assert TOOL.check() == 0


def test_tabelle_normal_b_und_a_gegen_live_rechnung(ref):
    nt = ref["distance_tables"]["normal"]
    for i, n in enumerate(nt["n"]):
        if n <= 1000:
            assert X.normal_distance(n, "B") == pytest.approx(nt["B"][i], rel=1e-6)
            assert X.normal_distance(n, "A") == pytest.approx(nt["A"][i], rel=1e-6)
            assert X.common_distance(n) == pytest.approx(nt["common"][i], rel=1e-6)


def test_kurve_ist_nach_n_sortiert_und_nutzt_die_gleichen_werte(ref):
    cur = ref["curve"]
    assert cur["n"] == sorted(cur["n"]) and len(cur["n"]) == len(cur["common"]) == len(cur["B"]) == len(cur["A"])
    assert cur["n"][-1] == pytest.approx(1e12, rel=1e-9)
    for k in (0, 5, 20):
        n = cur["n"][k]
        assert X.common_distance(n) == pytest.approx(cur["common"][k], rel=1e-6)
        assert X.normal_distance(n, "B", 50001) == pytest.approx(cur["B"][k], rel=1e-6)
    assert all(b < a for a, b in zip(cur["B"], cur["B"][1:]))                    # Normierung B fällt auf dem Gitter streng
    assert all(b < a for a, b in zip(cur["common"], cur["common"][1:]))


def test_beste_anpassung_ist_obere_schranke_und_besser_als_normierung_b(ref):
    table = ref["best_fit"]["table"]
    assert set(table) == {str(n) for n in C.N_PEGEL_OPTIONS}
    last = None
    for n_s in sorted(table, key=int):
        n = int(n_s)
        fit = table[n_s]
        gb = X.best_fit_distance(n, "gumbel", (fit["gumbel"]["mu"], fit["gumbel"]["sigma"]))
        gv = X.best_fit_distance(n, "gev", (fit["gev"]["mu"], fit["gev"]["sigma"], fit["gev"]["xi"]))
        assert gb == pytest.approx(fit["gumbel"]["ks"], rel=1e-9) and gv == pytest.approx(fit["gev"]["ks"], rel=1e-9)
        assert gb <= X.normal_distance(n, "B") + 1e-9                           # B ist ein Gumbel: das beste Gumbel ist höchstens so schlecht
        assert gv < gb and gv < 0.004                                           # die GEV mit freiem ξ liegt schon ab n = 2 unter 0,004
        assert fit["gev"]["xi"] < 0
        assert fit["gev"]["ks_other_run"] == pytest.approx(fit["gev"]["ks"], rel=1e-6)    # beide Zufalls-Seeds der Suche enden am selben Optimum
        if last is not None:
            assert gv < last                                                    # fällt in n
        last = gv


def test_bestes_gev_ist_besser_als_das_lokale_optimum_von_nelder_mead_allein(ref):
    assert ref["best_fit"]["table"]["2"]["gev"]["ks"] < 0.0037 < 0.01411
    assert ref["best_fit"]["table"]["3"]["gev"]["ks"] < 0.0034 < 0.00368


def test_schwellen_exponential_bestes_gumbel_und_normal_bestes_gumbel(ref):
    assert ref["exp_best_gumbel"]["thresholds"] == {"0.05": 2, "0.02": 3, "0.01": 6, "0.005": 11, "0.001": 52}
    assert ref["exp_best_gumbel"]["monotone_1_to_400"] is True
    bg = ref["thresholds"]["normal_best_gumbel"]
    assert float(f"{bg['0.02']:.3g}") == 18.8 and round(bg["0.01"]) == 897 and float(f"{bg['0.005']:.3g}") == 8.28e5
    # unabhängig: bei n = 897 liegt das beste Gumbel (obere Schranke aus der Tabelle) schon unter 0,01, bei n = 365 noch darüber
    t = ref["best_fit"]["table"]
    assert t["897"]["gumbel"]["ks"] < 0.01
    assert X.best_fit_distance(365, "gumbel", (t["365"]["gumbel"]["mu"], t["365"]["gumbel"]["sigma"])) > 0.01


def test_normierung_a_nicht_monoton_in_der_datei(ref):
    a = ref["thresholds"]["normal_A"]
    assert a["first_below_0.05"] == 4 and a["lasting_below_0.05"] == 622 and a["rises_count"] == 13 and a["rises_from_to"] == [4, 17]
    assert a["0.01"] is None and a["0.02"] == pytest.approx(1.07e12, rel=5e-3)


def test_baender_haben_die_erwarteten_zaehlungen(ref):
    b = ref["bands"]
    assert b["n20_R20000"]["seeds"] == 100 and b["pareto_alpha_n20_R20000"]["seeds"] == 30 and b["visibility_n365_R20000"]["seeds"] == 30
    assert b["exp_moments_R20000"]["seeds"] == 40
    for key in ("exp", "gleich", "pareto2", "normal"):
        band = b["n20_R20000"][key]
        assert band["over_dkw"] >= band["over_1.2dkw"] >= 0
        assert 0.005 < band["ks_Fn_mean"] < 0.007                               # Erwartung des Rauschens 0,8687/√R = 0,0061 bei R = 20 000
    assert b["n20_R20000"]["dkw"] == pytest.approx(X.dkw_bound(20000), rel=1e-12)

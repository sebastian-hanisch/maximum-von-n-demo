"""Jede Zahl der README: Zeichenkette der README gegen die Rechnung bzw. die eingefrorene Messung (Zahl → Test-Tabelle der README).

Die Zeichenketten werden hier aus Rechnung oder Referenz gebildet und müssen wörtlich in der README stehen; ändert sich eine Rechnung, bricht der Test, und die README ist nachzuziehen."""

import math
import re
from pathlib import Path

import pytest

import mxn_distance as X
import mxn_distributions as D
import mxn_evaluation as E
import mxn_reference as REF
from mxn_texts import de, de_signed, round_sig, sig3, thousands

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")


README_FLAT = re.sub(r"\s+", " ", README)


def in_readme(*parts):
    """Jede Zeichenkette muss (bei vereinheitlichten Leerzeichen und Zeilenumbrüchen) in der README stehen."""
    fehlt = [p for p in parts if re.sub(r"\s+", " ", p) not in README_FLAT]
    assert not fehlt, f"in der README nicht gefunden: {fehlt}"


@pytest.fixture(scope="module")
def ref():
    return REF.load()


# --- Befund 1 und 2: Bänder -------------------------------------------------------------------------------------------------------------------
def test_claims_baender(ref):
    b = ref["bands"]["n20_R20000"]
    keys = ("exp", "gleich", "pareto2", "normal")
    in_readme(" / ".join(de(b[k]["ks_Fn_mean"], 5) for k in keys), " / ".join(de(b[k]["ks_Fn_max"], 5) for k in keys))
    assert b["seeds"] == 100 and b["R"] == 20000 and b["n"] == 20
    assert [b[k]["over_dkw"] for k in keys] == [0, 1, 2, 0] and sum(b[k]["over_dkw"] for k in keys) == 3
    assert sum(b[k]["over_1.2dkw"] for k in keys) == 1 and b["gleich"]["over_1.2dkw"] == 1
    in_readme("0, 1, 2 und 0 von je 100 Läufen (3 von 400)", "über 1,2 × Schranke = 0,0138 liegt 1 von 400 (Gleich, 0,01597)")
    assert de(b["dkw"], 4) == "0,0115" and de(1.2 * b["dkw"], 4) == "0,0138"
    in_readme("0,0115 (99 %)", "0,8687/√R = 0,00614")
    assert 0.8687 / math.sqrt(20000) == pytest.approx(0.00614, abs=5e-6)
    p = ref["bands"]["pareto_alpha_n20_R20000"]
    assert p["seeds"] == 30
    in_readme(f"mittlerer Abstand {de(p['pareto15']['ks_Fn_mean'], 5)} und {de(p['pareto4']['ks_Fn_mean'], 5)}", "über der Schranke 1 und 0 von 30 Läufen, über 1,2 × Schranke 0 von 30")
    assert (p["pareto15"]["over_dkw"], p["pareto4"]["over_dkw"]) == (1, 0) and p["pareto15"]["over_1.2dkw"] == p["pareto4"]["over_1.2dkw"] == 0
    # Fehlalarmrate der Regel „unter 1,2 × Schranke“: 2·exp(−2·(1,2·1,628)²) rund 0,1 % je Lauf, erwartet 0,4 von 400
    rate = 2 * math.exp(-2 * (1.2 * 1.628) ** 2)
    assert rate == pytest.approx(0.001, abs=3e-4) and 400 * rate == pytest.approx(0.4, abs=0.12)
    in_readme("Wahrscheinlichkeit 0,1 % je Lauf vereinbar (erwartet 0,4 von 400)", "3 von 400 Läufen statt 4 von 400")
    m = ref["bands"]["exp_moments_R20000"]
    assert m["seeds"] == 40 and m["20"]["share_abs_z_over_2"] == 0.125 and m["20"]["z_of_mean"] == pytest.approx(-2.75, abs=0.005)
    in_readme("in 5 von 40 Läufen (12,5 %)", "2,75 Standardfehler unter 0")
    assert m["365"]["z_of_mean"] == pytest.approx(0.116, abs=0.001)
    in_readme(f"bei n = 365 liegt das Mittel der 40 z-Werte bei {de_signed(m['365']['z_of_mean'], 2)} Standardfehlern")
    c = ref["bands"]["exp_moments_control_n20"]
    assert c["seeds"] == 300
    in_readme(f"Mittel {de_signed(c['z_mean'], 2)} und Standardabweichung {de(c['z_sd'], 2)}", f"Anteil |z| > 2 von {de(100 * c['share_abs_z_over_2'], 1)} %")
    assert abs(c["z_mean"]) < 0.2 and 0.85 < c["z_sd"] < 1.15                # die Kontrolle bestätigt N(0, 1) (Standardfehler des Mittels 1/√300 = 0,058)
    in_readme("(erwartet 4,6 %)")
    assert 100 * 2 * (1 - 0.5 * math.erfc(-2 / math.sqrt(2))) == pytest.approx(4.6, abs=0.06)


# --- Befunde 3, 4, 6: Abstände ----------------------------------------------------------------------------------------------------------------
def test_claims_abstaende(ref):
    in_readme(de(5 * X.common_distance(5), 4) + " (n = 5)", de(20 * X.common_distance(20), 4) + " (n = 20)", de(365 * X.common_distance(365), 4) + " (n = 365)",
              de(10 ** 6 * X.common_distance(10 ** 6), 5) + " (n = 10⁶)", "Grenzwert 2/e² = " + de(2 / math.e ** 2, 5))
    in_readme("e⁻¹ = " + de(math.exp(-1), 4), "n = 2 gibt " + de(X.common_distance(2), 8))
    d_c, d_B, d_A = X.common_distance(365), X.normal_distance(365, "B"), X.normal_distance(365, "A")
    in_readme(f"Exponential (Standardnormierung) {de(d_c, 5)}, Normal (Normierung B) {de(d_B, 4)}, **Faktor {de(d_B / d_c, 1)}**", f"(Normierung A) {de(d_A, 4)}, Faktor {de(d_A / d_c, 1)}")
    bg = ref["best_fit"]["table"]["365"]["gumbel"]["ks"]
    eg = ref["exp_best_gumbel"]["distance"]["365"]
    in_readme(f"({de(eg, 6)} und {de(bg, 5)})", f"ist es der Faktor {de(bg / eg, 1)}")
    assert eg == pytest.approx(1.40e-4, rel=5e-3) and bg == pytest.approx(0.01144, rel=5e-3)
    assert bg / eg == pytest.approx(81.5, abs=0.1)
    # Rate der Annäherung (README „Ehrliche Grenzen“): Abstand·ln n ist bei Normierung B flach, bei A wächst es von n = 365 bis 10³⁰
    prodB = [X.normal_distance(n, "B") * math.log(n) for n in (20, 100, 365, 1000, 10 ** 4)]
    assert 0.122 < min(prodB) and max(prodB) < 0.127
    prodA = [X.normal_distance(n, "A") * math.log(n) for n in (365, 10 ** 4, 10 ** 12, 10 ** 20, 10 ** 30)]
    assert all(a < b for a, b in zip(prodA, prodA[1:])) and prodA[0] == pytest.approx(0.309, abs=0.001) and prodA[-1] == pytest.approx(0.743, abs=0.001)
    in_readme("bei Normierung A steigt es langsam weiter (n = 365 bis 10³⁰)")


def test_claims_schwellen(ref):
    th = ref["thresholds"]
    in_readme(f"**n ≥ {th['exp_std_integer']['0.01']}**", f"n ≥ {ref['exp_best_gumbel']['thresholds']['0.01']};", f"**n ≥ {sig3(th['normal_B']['0.01'])}**", f"n ≥ {sig3(th['normal_best_gumbel']['0.01'])};")
    assert th["normal_A"]["0.01"] is None
    in_readme("Normierung A erreicht 0,01 bis n = 10³⁰ nicht", f"erstmals bei n = {th['normal_A']['first_below_0.05']}, dauerhaft ab n = {th['normal_A']['lasting_below_0.05']}")
    # Normierung A nicht monoton (eigene Rechnung, Gitter 20.001): Werte bei n = 4 und 17, 13 Anstiege
    k = {n: X.normal_distance(n, "A", 20001) for n in range(2, 3001)}
    ups = [n for n in range(2, 3000) if k[n + 1] >= k[n]]
    assert ups == list(range(4, 17)) and len(ups) == 13 and max(k[n] for n in range(5, 3001)) == k[17]
    in_readme(f"von n = 4 ({de(k[4], 4)}) bis n = 17 ({de(k[17], 4)})", "13 Anstiege im Gitter n = 2 bis 3.000")
    assert k[4] < k[17] and all(k[n] > k[n + 1] for n in range(17, 400))
    assert th["normal_A"]["rises_from_to"] == [4, 17] and th["normal_A"]["rises_count"] == 13


# --- Befund 7: Pegel --------------------------------------------------------------------------------------------------------------------------
def test_claims_pegel(ref):
    a = E.analyse_pegel(365, 0.01)
    d = lambda k: de_signed(a.levels[k] - a.exact, 3)
    in_readme(f"exakt {de(a.exact, 3)}", f"Gumbel A {de(a.levels['gumbelA'], 3)} ({d('gumbelA')})", f"{de(a.levels['gumbelB'], 3)} ({d('gumbelB')})",
              f"ist {de(a.true_p['gumbelB'], 4)} statt 0,01", f"(ξ̂ = {de_signed(a.xi_gev, 3)}) liegt bei n = 365, p = 0,01 um {d('gevbest')} daneben")
    in_readme(f"Gumbel A bei n = 365, p = 0,01 ist {de(a.levels['gumbelA'], 3)}** ({de(a.levels['gumbelA'], 6)})")
    b24 = E.analyse_pegel(24, 0.001)
    in_readme(f"Gumbel B {de_signed(b24.levels['gumbelB'] - b24.exact, 3)}")
    lo_A = lo_B = 9.0
    hi_A = hi_B = -9.0
    for n in (24, 365, 8760):
        p = E.analyse_pegel(n, 0.01)
        dA, dB = p.levels["gumbelA"] - p.exact, p.levels["gumbelB"] - p.exact
        lo_A, hi_A, lo_B, hi_B = min(lo_A, dA), max(hi_A, dA), min(lo_B, dB), max(hi_B, dB)
    in_readme(f"Gumbel A zwischen {de_signed(lo_A, 3)} und {de_signed(hi_A, 3)}", f"Gumbel B zwischen {de_signed(lo_B, 3)} und {de_signed(hi_B, 3)} (n = 24, 365, 8.760)")
    g2, g365 = ref["best_fit"]["table"]["2"]["gev"], ref["best_fit"]["table"]["365"]["gev"]
    in_readme(f"bei n = 2 einen Abstand von {de(g2['ks'], 5)} (ξ̂ = {de_signed(g2['xi'], 3)}) und bei n = 365 von {de(g365['ks'], 5)}")
    assert g2["ks"] < 0.004 and g365["ks"] < 0.001
    # beste GEV gegen die drei Gumbel-Verfahren: alle n und p der App (Aussage in README Befund 7 und im Verdikt)
    import mxn_constants as C
    gev_p001, gev_tail = [], []
    for n in C.N_PEGEL_OPTIONS:
        for p in C.P_OPTIONS:
            q = E.analyse_pegel(n, p)
            abw = {k: abs(v - q.exact) for k, v in q.levels.items()}
            assert abw["gevbest"] < min(abw["gumbelA"], abw["gumbelB"], abw["gumbelbest"]), (n, p)
            if n >= 20 and p == 0.01:
                gev_p001.append(abw["gevbest"])
            if n >= 20 and p == 0.0001:
                gev_tail.append(abw["gevbest"])
    assert max(gev_p001) < 0.06 and 0.05 < max(gev_p001)
    assert min(gev_tail) == pytest.approx(0.075, abs=0.0005) and max(gev_tail) == pytest.approx(0.375, abs=0.0005)
    in_readme("höchstens 0,06 daneben, im Tail (p = 0,0001) um 0,075 bis 0,375 (n ≥ 20)")


# --- Befund 8: Sichtbarkeit -------------------------------------------------------------------------------------------------------------------
def test_claims_sichtbarkeit(ref):
    see_e, see_n = X.blocks_to_see(X.common_distance(365)), X.blocks_to_see(X.normal_distance(365, "B"))
    assert see_e == pytest.approx(4.81e6, rel=2e-3)
    assert thousands(see_n) == "5.823" and round_sig(see_n) == "5.800" and round_sig(see_e) == "4.800.000"
    in_readme("rund 4,8 Millionen Blöcke", f"Normal (Normierung B) rund {round_sig(see_n)}", "5.800 (gerundet aus 5.823)")
    v = ref["bands"]["visibility_n365_R20000"]
    assert v["seeds"] == 30 and v["R"] == 20000 and v["n"] == 365
    in_readme(f"mittlerer Abstand zum Gumbel-Gesetz {de(v['exp']['ks_limit_mean'], 5)} (Rauschen), über der Schranke\n   in {v['exp']['over_dkw']} von 30 Läufen; Normal {de(v['normal']['ks_limit_mean'], 5)}, über der Schranke in {v['normal']['over_dkw']} von 30")
    assert (v["exp"]["over_dkw"], v["normal"]["over_dkw"]) == (0, 30)
    # Gegenprobe: bei bloßem Rauschen wären 0,3 von 30 Läufen über der Schranke zu erwarten (1 %)
    assert 30 * 0.01 == pytest.approx(0.3)
    assert X.noise_band(20000) == pytest.approx(1.628 / math.sqrt(20000), rel=1e-14)


# --- Befund 9: Momente ------------------------------------------------------------------------------------------------------------------------
def test_claims_momente():
    from fractions import Fraction
    H = sum(Fraction(1, k) for k in range(1, 21))
    V = sum(Fraction(1, k * k) for k in range(1, 21))
    assert H == Fraction(55835135, 15519504)
    in_readme(f"Hₙ = {de(float(H), 5)} (n = 20, Bruch {thousands(H.numerator)}/{thousands(H.denominator)})", f"ln n + γ = {de(math.log(20) + D.EULER_GAMMA, 5)}", f"Σ 1/k² = {de(float(V), 5)}", f"π²/6 = {de(math.pi ** 2 / 6, 5)}")
    in_readme("E[M₂] = 161/36 = 4,47")
    assert 161 / 36 == pytest.approx(4.4722, abs=1e-4)
    in_readme("19/27 (n = 3) und 175/256 (n = 4)", "n = 50: " + de(1 - 0.98 ** 50, 4))


# --- die Test-Tabelle der README nennt nur vorhandene Tests -----------------------------------------------------------------------------------
def test_alle_in_der_readme_genannten_tests_gibt_es():
    gefunden = 0
    for datei, name in re.findall(r"`(test_[a-z_]+)::(test_[a-z0-9_]+)`", README):
        text = (ROOT / "tests" / f"{datei}.py").read_text(encoding="utf-8")
        assert f"def {name}(" in text, (datei, name)
        gefunden += 1
    for name in re.findall(r"`(test_claims_[a-z]+)`", README):
        assert f"def {name}(" in Path(__file__).read_text(encoding="utf-8"), name
        gefunden += 1
    assert gefunden >= 12


def test_readme_nennt_die_normierung_und_die_zufalls_seeds_richtig_geschrieben():
    assert "Zufalls-Seed" in README and "Saatwert" not in README and "Schwanz" not in README
    assert "trägt" not in README and "tragen" not in README.replace("beitragen", "")
    assert "— " not in README and " – " in README                             # kein Geviertstrich im Fließtext (Gedankenstrich nur im Titel und im Footer)

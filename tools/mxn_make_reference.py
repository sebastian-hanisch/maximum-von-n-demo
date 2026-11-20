"""Erzeugt und prüft `mxn_reference.json`: die eingefrorenen Tabellen der Demo (Abstand zum Grenzgesetz, Schwellen-n, bestes Gumbel/GEV, Simulationsbänder).

Aufruf (Test-venv, ein Thread, im Ordner der Demo):
    OMP_NUM_THREADS=1 python tools/mxn_make_reference.py --check          # rechnet n <= 365, die ganzzahligen Schwellen und die Normal-B-Schwellen neu (Sekunden)
    OMP_NUM_THREADS=1 python tools/mxn_make_reference.py --write           # erzeugt die schnellen Teile neu, übernimmt die langsamen aus der Datei
    OMP_NUM_THREADS=1 python tools/mxn_make_reference.py --write --lang    # zusätzlich: Schwellen für Normierung A und bestes Gumbel, Exponential mit bestem Gumbel (n = 1 bis 400), rund eine Minute
    OMP_NUM_THREADS=1 python tools/mxn_make_reference.py --write --bands   # zusätzlich: Simulationsbänder (SplitMix64 als numpy-Vektor, Messhilfe), einige Minuten

Zufall: SplitMix64 als numpy-Vektor in der Zähler-Form (der k-te Ausgang ist mix(Zufalls-Seed + k·γ)); bitgleich zur Vorlage `mxn_rng.py`
(Test `tests/test_rng_vector.py`). Nur Messhilfe für die großen Bänder; die App rechnet mit der reinen Vorlage.
scipy (Differential Evolution, Nelder-Mead, `ndtri`) steht nur hier und in den Tests, nie in der App.
"""

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import mxn_constants as C  # noqa: E402
import mxn_distance as X  # noqa: E402
import mxn_distributions as D  # noqa: E402

REF = ROOT / "mxn_reference.json"
STAND = "2026-11-20"
N_TABLE = (2, 3, 5, 7, 10, 16, 20, 50, 100, 365, 1000, 10000)
N_COMMON = (1, 2, 3, 5, 10, 20, 28, 50, 100, 365, 1000, 10000, 1_000_000)
CURVE_N = [2.0] + [10.0 ** (k / 4.0) for k in range(2, 49)]          # n = 3,16 bis 10^12
THRESH_EXP_MARKS = C.MARKS


def log(*a):
    print(*a, flush=True)


# ------------------------------------------------------------------------------------------------------------------------------------------
# SplitMix64 als numpy-Vektor (Messhilfe)
# ------------------------------------------------------------------------------------------------------------------------------------------
_G = np.uint64(0x9E3779B97F4A7C15)
_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)


def sm64_u64(seed: int, n: int, start: int = 0) -> np.ndarray:
    """n Ausgänge ab dem (start + 1)-ten, wie `SplitMix64(seed)` nach `start` Aufrufen von `next_u64`."""
    with np.errstate(over="ignore"):
        k = np.arange(start + 1, start + n + 1, dtype=np.uint64)
        z = np.uint64(seed & 0xFFFFFFFFFFFFFFFF) + k * _G
        z = (z ^ (z >> np.uint64(30))) * _M1
        z = (z ^ (z >> np.uint64(27))) * _M2
        return z ^ (z >> np.uint64(31))


def sm64_unif(seed: int, n: int, start: int = 0) -> np.ndarray:
    """Gleichverteilt in [0, 1) mit 53 Bit, wie `SplitMix64.random()`."""
    return (sm64_u64(seed, n, start) >> np.uint64(11)).astype(np.float64) / float(1 << 53)


def block_vmin_np(seed: int, R: int, n: int) -> np.ndarray:
    """Dieselben Blöcke wie `mxn_scenario.block_vmin` (kleinstes v = 1 − u_max je Block), vektorisiert."""
    return 1.0 - sm64_unif(seed, R * n).reshape(R, n).max(axis=1)


def values_np(key: str, v: np.ndarray) -> np.ndarray:
    """Umkehrung mit scipy `ndtri` für die Normalverteilung (nur Messhilfe)."""
    if key == "normal":
        from scipy import special
        return -special.ndtri(np.minimum(v, D._V_MAX))
    return D.inv_from_v(key, v)


# ------------------------------------------------------------------------------------------------------------------------------------------
# Tabellen
# ------------------------------------------------------------------------------------------------------------------------------------------
def distance_tables() -> dict:
    return {
        "common": {"n": list(N_COMMON), "d": [X.common_distance(n) for n in N_COMMON]},
        "normal": {"n": list(N_TABLE), "B": [X.normal_distance(n, "B") for n in N_TABLE], "A": [X.normal_distance(n, "A") for n in N_TABLE],
                   "common": [X.common_distance(n) for n in N_TABLE]},
    }


def curve_table() -> dict:
    return {"n": CURVE_N, "common": [X.common_distance(n) for n in CURVE_N],
            "B": [X.normal_distance(n, "B", 50001) for n in CURVE_N], "A": [X.normal_distance(n, "A", 50001) for n in CURVE_N]}


def exp_integer_thresholds(nmax: int = 2000) -> dict:
    vals = [X.common_distance(n, 4001) for n in range(1, nmax + 1)]
    out = {}
    for mark in THRESH_EXP_MARKS:
        out[str(mark)] = next((n for n in range(1, nmax + 1) if all(v < mark for v in vals[n - 1:])), None)
    return out


def normal_B_thresholds(marks=THRESH_EXP_MARKS, hi=14.0) -> dict:
    f = lambda L: X.normal_distance(10.0 ** L, "B")
    out = {}
    for mark in marks:
        L = X.first_log_cross(f, 0.7, hi, mark, step=0.1)
        out[str(mark)] = None if L is None else 10.0 ** L
    return out


def normal_A_thresholds() -> dict:
    f = lambda L: X.normal_distance(10.0 ** L, "A")
    out = {}
    for mark in (0.05, 0.02, 0.01):
        L = X.first_log_cross(f, 0.7, 30.0, mark, step=0.25)
        out[str(mark)] = None if L is None else 10.0 ** L
    # Normierung A ist nicht monoton: erstes n unter 0,05 und dauerhaftes n (ganzzahlig, Gitter 20 001)
    kA = {n: X.normal_distance(n, "A", 20001) for n in range(2, 3001)}
    first = next(n for n in range(2, 3001) if kA[n] < 0.05)
    last = max(n for n in range(2, 3001) if kA[n] >= 0.05)
    ups = [n for n in range(2, 3000) if kA[n + 1] >= kA[n]]
    out["first_below_0.05"] = first
    out["lasting_below_0.05"] = last + 1
    out["rises_from_to"] = [ups[0], ups[-1] + 1] if ups else None
    out["rises_count"] = len(ups)
    return out


# --- bestes Gumbel / beste GEV (scipy nur hier) ---------------------------------------------------------------------------------------------
def _exact_on_grid(n: float, npts: int):
    b, a = D.norming("normal", n, "B")
    z = np.linspace(C.GRID_X[0], C.GRID_X[1], npts)
    return z, D.cdf_max("normal", b + a * z, n)


def best_gumbel(n: float) -> dict:
    from scipy import optimize
    z, ex = _exact_on_grid(n, 4001)

    def f(p):
        return float(np.max(np.abs(ex - D.gumbel_cdf_ab(z, p[0], math.exp(p[1])))))
    fits = []
    for sd in (0, 1):
        de = optimize.differential_evolution(f, [(-3, 3), (-3, 3)], seed=sd, tol=1e-12, maxiter=400, popsize=30, polish=False)
        fits.append(optimize.minimize(f, de.x, method="Nelder-Mead", options=dict(xatol=1e-10, fatol=1e-13, maxiter=4000)))
    vals = [X.best_fit_distance(n, "gumbel", (r.x[0], math.exp(r.x[1]))) for r in fits]
    i = int(np.argmin(vals))
    return {"mu": float(fits[i].x[0]), "sigma": float(math.exp(fits[i].x[1])), "ks": vals[i], "ks_other_run": vals[1 - i]}


def best_gev(n: float) -> dict:
    from scipy import optimize
    z, ex = _exact_on_grid(n, 4001)

    def f(p):
        return float(np.max(np.abs(ex - D.gev_cdf(z, p[0], math.exp(p[1]), p[2]))))
    fits = []
    for sd in (0, 1):
        de = optimize.differential_evolution(f, [(-3, 3), (-3, 3), (-0.9, 0.9)], seed=sd, tol=1e-12, maxiter=600, popsize=30, polish=False)
        fits.append(optimize.minimize(f, de.x, method="Nelder-Mead", options=dict(xatol=1e-9, fatol=1e-12, maxiter=6000)))
    vals = [X.best_fit_distance(n, "gev", (r.x[0], math.exp(r.x[1]), r.x[2])) for r in fits]
    i = int(np.argmin(vals))
    return {"mu": float(fits[i].x[0]), "sigma": float(math.exp(fits[i].x[1])), "xi": float(fits[i].x[2]), "ks": vals[i], "ks_other_run": vals[1 - i]}


def best_fit_table() -> dict:
    out = {}
    for n in C.N_PEGEL_OPTIONS:
        t0 = time.perf_counter()
        out[str(n)] = {"gumbel": best_gumbel(n), "gev": best_gev(n)}
        log(f"  beste Anpassung n = {n}: Gumbel {out[str(n)]['gumbel']['ks']:.5f}, GEV {out[str(n)]['gev']['ks']:.5f} (xi {out[str(n)]['gev']['xi']:+.3f}) [{time.perf_counter() - t0:.0f} s]")
    return out


def best_gumbel_thresholds() -> dict:
    from scipy import optimize

    def f(L):
        n = 10.0 ** L
        z, ex = _exact_on_grid(n, 4001)
        g = lambda p: float(np.max(np.abs(ex - D.gumbel_cdf_ab(z, p[0], math.exp(p[1])))))
        r = optimize.minimize(g, [0.0, 0.0], method="Nelder-Mead", options=dict(xatol=1e-8, fatol=1e-11, maxiter=2000))
        return X.best_fit_distance(n, "gumbel", (r.x[0], math.exp(r.x[1])), 50001)
    out = {}
    for mark in (0.02, 0.01, 0.005):
        L = X.first_log_cross(f, 0.7, 12.0, mark, step=0.5, iters=14)
        out[str(mark)] = None if L is None else 10.0 ** L
    return out


def exp_best_gumbel(n: int) -> float:
    from scipy import optimize
    x = np.linspace(-3.0, 40.0, 8001) + math.log(n)
    with np.errstate(all="ignore"):
        e = np.where(x > 0, np.exp(n * np.log1p(-np.exp(-np.maximum(x, 1e-300)))), 0.0)
    f = lambda p: float(np.max(np.abs(e - np.exp(-np.exp(-(x - p[0]) / math.exp(p[1]))))))
    best = None
    for st in ([math.log(n), 0.0], [math.log(n) - 0.3, 0.1], [math.log(n) + 0.3, -0.1]):
        r = optimize.minimize(f, st, method="Nelder-Mead", options=dict(xatol=1e-9, fatol=1e-13, maxiter=4000))
        if best is None or r.fun < best.fun:
            best = r
    xf = np.linspace(-3.0, 40.0, 400001) + math.log(n)
    with np.errstate(all="ignore"):
        ef = np.where(xf > 0, np.exp(n * np.log1p(-np.exp(-np.maximum(xf, 1e-300)))), 0.0)
    return float(np.max(np.abs(ef - np.exp(-np.exp(-(xf - best.x[0]) / math.exp(best.x[1]))))))


def exp_best_gumbel_table(nmax: int = 400) -> dict:
    vals = [exp_best_gumbel(n) for n in range(1, nmax + 1)]
    thr = {}
    for mark in THRESH_EXP_MARKS:
        thr[str(mark)] = next((n for n in range(1, nmax + 1) if all(v < mark for v in vals[n - 1:])), None)
    pick = {str(n): vals[n - 1] for n in (1, 2, 5, 10, 20, 100, 365, 400)}
    return {"thresholds": thr, "distance": pick, "monotone_1_to_400": all(vals[i + 1] < vals[i] for i in range(nmax - 1))}


# ------------------------------------------------------------------------------------------------------------------------------------------
# Simulationsbänder (Messhilfe, numpy-SplitMix64; Zufalls-Seeds anders als in der Vormessung V10)
# ------------------------------------------------------------------------------------------------------------------------------------------
KEY_ID = {"exp": 1, "gleich": 2, "pareto2": 3, "normal": 4, "pareto15": 5, "pareto4": 6, "frechet": 7}


def _ks(sample, cdf):
    s = np.sort(sample)
    m = s.size
    f = cdf(s)
    i = np.arange(1, m + 1)
    return float(max(np.max(i / m - f), np.max(f - (i - 1) / m)))


def band_ks_vs_Fn(keys, n, R, S, base) -> dict:
    eps = X.dkw_bound(R)
    out = {"n": n, "R": R, "seeds": S, "dkw": eps}
    for key in keys:
        k1, k2 = [], []
        for s in range(S):
            v = block_vmin_np(base + 1000 * s + KEY_ID[key], R, n)
            M = values_np(key, v)
            k1.append(_ks(M, lambda x: D.cdf_max(key, x, n)))
            if D.type_info(key)["family"]:
                b, a = D.norming(key, n, "B")
                k2.append(_ks(M, lambda x: D.limit_cdf(key, (x - b) / a)))
        k1 = np.array(k1)
        out[key] = {"ks_Fn_mean": float(k1.mean()), "ks_Fn_max": float(k1.max()), "over_dkw": int(np.sum(k1 > eps)), "over_1.2dkw": int(np.sum(k1 > 1.2 * eps)),
                    "ks_limit_mean": float(np.mean(k2)) if k2 else None, "ks_limit_sd": float(np.std(k2, ddof=1)) if k2 else None}
        log(f"  Band n={n} R={R} S={S} {key}: KS(Fn) mean {k1.mean():.5f} max {k1.max():.5f} über DKW {int(np.sum(k1 > eps))}, über 1,2·DKW {int(np.sum(k1 > 1.2 * eps))}")
    return out


def band_visibility(n, R, S, base) -> dict:
    eps = X.dkw_bound(R)
    out = {"n": n, "R": R, "seeds": S, "dkw": eps}
    for key in ("exp", "normal"):
        lim, ex = [], []
        for s in range(S):
            M = values_np(key, block_vmin_np(base + 31 * s + KEY_ID[key], R, n))
            b, a = D.norming(key, n, "B")
            lim.append(_ks(M, lambda x: D.limit_cdf(key, (x - b) / a)))
            ex.append(_ks(M, lambda x: D.cdf_max(key, x, n)))
        lim, ex = np.array(lim), np.array(ex)
        out[key] = {"ks_limit_mean": float(lim.mean()), "ks_limit_min": float(lim.min()), "ks_limit_max": float(lim.max()), "over_dkw": int(np.sum(lim > eps)),
                    "ks_Fn_max": float(ex.max()), "exact_distance": X.distance(key, n, "B")}
        log(f"  Sichtbarkeit n={n} R={R} {key}: Abstand {out[key]['exact_distance']:.5f}, Mittel KS(Grenze) {lim.mean():.5f}, über DKW in {int(np.sum(lim > eps))} von {S}")
    return out


def band_exp_moments(ns, R, S, base) -> dict:
    out = {"R": R, "seeds": S}
    for n in ns:
        H = D.harmonic(n)
        V = math.fsum(1.0 / (k * k) for k in range(1, n + 1))
        zs = []
        for s in range(S):
            M = values_np("exp", block_vmin_np(base + 17 * s + n, R, n))
            zs.append((M.mean() - H) / math.sqrt(V / R))
        zs = np.array(zs)
        out[str(n)] = {"H_n": H, "sum_inv_sq": V, "max_abs_z": float(np.abs(zs).max()), "share_abs_z_over_2": float(np.mean(np.abs(zs) > 2.0)), "z_of_mean": float(zs.mean() * math.sqrt(S))}
        log(f"  Exp-Momente n={n}: max|z| {np.abs(zs).max():.2f}, Anteil |z|>2 {np.mean(np.abs(zs) > 2):.3f}, z des Gesamtmittels {zs.mean() * math.sqrt(S):+.2f}")
    return out


def band_exp_control(n, R, S, base) -> dict:
    """Kontrolle zu den Exponential-Momenten: viele Zufalls-Seeds, Verteilung der z-Werte des Mittels (erwartet: Mittel 0, Standardabweichung 1, Anteil |z| > 2 rund 4,6 %)."""
    H = D.harmonic(n)
    V = math.fsum(1.0 / (k * k) for k in range(1, n + 1))
    zs = np.array([(values_np("exp", block_vmin_np(base + 7919 * s, R, n)).mean() - H) / math.sqrt(V / R) for s in range(S)])
    out = {"n": n, "R": R, "seeds": S, "z_mean": float(zs.mean()), "z_sd": float(zs.std(ddof=1)), "share_abs_z_over_2": float(np.mean(np.abs(zs) > 2.0)), "max_abs_z": float(np.abs(zs).max())}
    log(f"  Kontrolle Exp-Momente n={n}, {S} Zufalls-Seeds: Mittel z {out['z_mean']:+.3f}, SD {out['z_sd']:.3f}, Anteil |z|>2 {out['share_abs_z_over_2']:.3f}")
    return out


def bands() -> dict:
    log("Simulationsbänder (numpy-SplitMix64; Zufalls-Seeds ab 910 000):")
    return {
        "n20_R20000": band_ks_vs_Fn(("exp", "gleich", "pareto2", "normal"), 20, 20000, 100, 910_000),
        "pareto_alpha_n20_R20000": band_ks_vs_Fn(("pareto15", "pareto4"), 20, 20000, 30, 930_000),
        "visibility_n365_R20000": band_visibility(365, 20000, 30, 950_000),
        "exp_moments_R20000": band_exp_moments((20, 365), 20000, 40, 970_000),
        "exp_moments_control_n20": band_exp_control(20, 20000, 300, 1_234_567),
    }


# ------------------------------------------------------------------------------------------------------------------------------------------
def meta() -> dict:
    import scipy
    return {
        "stand": STAND, "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "plattform": platform.system(),
        "quelle": "tools/mxn_make_reference.py (Abstände: mxn_distance.py; Optimierung: scipy Differential Evolution zwei Zufalls-Seeds plus Nelder-Mead; Bänder: SplitMix64 als numpy-Vektor)",
        "hinweis": "Abstände sind exakt gerechnet (Gitter 200 001 Punkte, Auflösung des Supremums rund 1e-8), keine Simulation; Bänder sind Zufallsergebnisse (Zahl der Zufalls-Seeds je Eintrag).",
    }


def build(args) -> dict:
    old = json.loads(REF.read_text(encoding="utf-8")) if REF.exists() else {}
    out = {"meta": meta()}
    t0 = time.perf_counter()
    out["distance_tables"] = {"quelle": "mxn_distance.common_distance / normal_distance", "stand": STAND, **distance_tables()}
    log(f"Abstandstabellen [{time.perf_counter() - t0:.0f} s]")
    out["curve"] = {"quelle": "mxn_distance (Gitter 50 001 Punkte)", "stand": STAND, **curve_table()}
    log(f"Kurve [{time.perf_counter() - t0:.0f} s]")
    thr = {"quelle": "mxn_distance.first_log_cross bzw. kleinstes dauerhaftes ganzzahliges n", "stand": STAND, "marks": list(C.MARKS)}
    thr["exp_std_integer"] = exp_integer_thresholds()
    log(f"Schwellen Exponential [{time.perf_counter() - t0:.0f} s]")
    thr["normal_B"] = normal_B_thresholds()
    log(f"Schwellen Normal B [{time.perf_counter() - t0:.0f} s]")
    if args.lang or "normal_A" not in old.get("thresholds", {}):
        thr["normal_A"] = normal_A_thresholds()
        thr["normal_best_gumbel"] = best_gumbel_thresholds()
        out["exp_best_gumbel"] = {"quelle": "scipy Nelder-Mead, drei Startwerte, Gitter 400 001", "stand": STAND, **exp_best_gumbel_table()}
        log(f"Schwellen A und bestes Gumbel, Exponential bestes Gumbel [{time.perf_counter() - t0:.0f} s]")
    else:
        thr["normal_A"] = old["thresholds"]["normal_A"]
        thr["normal_best_gumbel"] = old["thresholds"]["normal_best_gumbel"]
        out["exp_best_gumbel"] = old["exp_best_gumbel"]
    out["thresholds"] = thr
    if args.lang or "best_fit" not in old:
        out["best_fit"] = {"quelle": "scipy Differential Evolution (zwei Zufalls-Seeds) plus Nelder-Mead; Parameter auf der z-Skala der Normierung B", "stand": STAND, "n": list(C.N_PEGEL_OPTIONS), **{"table": best_fit_table()}}
    else:
        out["best_fit"] = old["best_fit"]
    if args.bands or "bands" not in old:
        out["bands"] = {"quelle": "tools/mxn_make_reference.py --bands", "stand": STAND, **bands()}
    else:
        out["bands"] = old["bands"]
    return out


def check() -> int:
    """Rechnet die schnellen Teile neu und vergleicht sie mit der Datei (relative Toleranz 1e-6, wegen der letzten Gleitkommastelle je Plattform)."""
    ref = json.loads(REF.read_text(encoding="utf-8"))
    bad = []

    def cmp(name, a, b, tol=1e-6):
        if a is None or b is None:
            if a != b:
                bad.append(f"{name}: {a} gegen {b}")
        elif abs(a - b) > tol * max(abs(b), 1e-300):
            bad.append(f"{name}: neu {a} gegen Datei {b}")

    for n, d in zip(ref["distance_tables"]["common"]["n"], ref["distance_tables"]["common"]["d"]):
        cmp(f"common n={n}", X.common_distance(n), d)
    nt = ref["distance_tables"]["normal"]
    for i, n in enumerate(nt["n"]):
        if n <= 365:
            cmp(f"normal B n={n}", X.normal_distance(n, "B"), nt["B"][i])
            cmp(f"normal A n={n}", X.normal_distance(n, "A"), nt["A"][i])
    th = ref["thresholds"]
    got = exp_integer_thresholds()
    for k, v in got.items():
        if v != th["exp_std_integer"][k]:
            bad.append(f"Exponential Schwelle {k}: neu {v} gegen Datei {th['exp_std_integer'][k]}")
    gotB = normal_B_thresholds(marks=(0.05, 0.02, 0.01))
    for k, v in gotB.items():
        cmp(f"Normal B Schwelle {k}", v, th["normal_B"][k], 1e-3)
    for n_s, fit in ref["best_fit"]["table"].items():
        n = int(n_s)
        cmp(f"Gumbel-Anpassung n={n}", X.best_fit_distance(n, "gumbel", (fit["gumbel"]["mu"], fit["gumbel"]["sigma"])), fit["gumbel"]["ks"])
        cmp(f"GEV-Anpassung n={n}", X.best_fit_distance(n, "gev", (fit["gev"]["mu"], fit["gev"]["sigma"], fit["gev"]["xi"])), fit["gev"]["ks"])
    for line in bad:
        log("ABWEICHUNG", line)
    log("mxn_reference.json: " + ("in Ordnung" if not bad else f"{len(bad)} Abweichungen"))
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--lang", action="store_true")
    ap.add_argument("--bands", action="store_true")
    args = ap.parse_args()
    if args.check:
        return check()
    if args.write:
        data = build(args)
        REF.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        log(f"geschrieben: {REF}")
        return 0
    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

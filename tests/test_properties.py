"""Eigenschaften: je 20 bis 50 Zufallsinstanzen mit festem Zufalls-Seed (hier numpy-Generator nur für die Prüfpunkte, nicht für die App)."""

import random

import numpy as np
import pytest

import mxn_distance as X
import mxn_distributions as D
import mxn_scenario as S
from mxn_rng import SplitMix64

CONTINUOUS = ("exp", "gleich", "pareto15", "pareto2", "pareto4", "frechet", "normal")


def _points(key, rng, m=40):
    if key == "normal":
        return np.sort(np.array([rng.uniform(-3, 5) for _ in range(m)]))
    if key == "gleich":
        return np.sort(np.array([rng.uniform(0.01, 1.0) for _ in range(m)]))
    return np.sort(np.array([rng.uniform(0.01, 30.0) for _ in range(m)]))


@pytest.mark.parametrize("key", CONTINUOUS)
def test_fn_waechst_in_x_und_faellt_in_n(key):
    rng = random.Random(7)
    for _ in range(25):
        x = _points(key, rng)
        n = rng.randint(1, 300)
        f1 = D.cdf_max(key, x, n)
        assert np.all(np.diff(f1) >= -1e-15) and np.all((f1 >= 0) & (f1 <= 1))
        assert np.all(D.cdf_max(key, x, n + 1) <= f1 + 1e-15)


@pytest.mark.parametrize("key", CONTINUOUS)
def test_verschachtelung_der_bloecke_f_hoch_n_mal_m(key):
    rng = random.Random(11)
    for _ in range(25):
        x = _points(key, rng)
        n, m = rng.randint(1, 30), rng.randint(1, 12)
        assert D.cdf_max(key, x, n * m) == pytest.approx(D.cdf_max(key, x, n) ** m, rel=1e-10, abs=1e-15)


def test_lage_skalen_aenderung_aendert_den_normierten_abstand_nicht():
    """Normal(μ, σ): Gitter in der Skala der Einzelwerte verändern (x = μ + σ·y), Normierung und Grenzgesetz mitführen: dasselbe Supremum."""
    rng = random.Random(3)
    for _ in range(20):
        n = rng.randint(2, 2000)
        mu, sigma = rng.uniform(-50, 50), rng.uniform(0.1, 20)
        a, b = D.normal_constants_B(n)
        z = np.linspace(-4, 16, 20001)
        base = np.max(np.abs(D.cdf_max("normal", b + a * z, n) - D.limit_cdf("normal", z)))
        x = mu + sigma * (b + a * z)                                      # Einzelwerte der Normal(μ, σ), normiert mit μ + σ·b und σ·a
        aff = D.cdf_max("normal", (x - mu) / sigma, n)
        z2 = ((x - mu) / sigma - b) / a
        assert np.max(np.abs(aff - D.limit_cdf("normal", z2))) == pytest.approx(base, abs=1e-12)


def test_abstand_haengt_nicht_an_alpha_und_ist_fuer_drei_verteilungen_gleich():
    for n in (1, 2, 5, 20, 100, 365, 1000):
        d = X.distance("exp", n)
        for key in ("gleich", "pareto15", "pareto2", "pareto4"):
            assert X.distance(key, n) == d
    assert X.distance("frechet", 20) == 0.0


def test_abstand_exponential_faellt_streng_und_normal_b_auch_normal_a_nicht():
    ns = list(range(1, 400))
    vals = [X.common_distance(n, 4001) for n in ns]
    assert all(b < a for a, b in zip(vals, vals[1:]))
    grid = [10.0 ** (0.7 + 0.1 * i) for i in range(0, 114)]
    vb = [X.normal_distance(n, "B", 20001) for n in grid]
    assert all(b < a for a, b in zip(vb, vb[1:]))
    va = [X.normal_distance(n, "A", 20001) for n in range(2, 30)]
    assert any(b >= a for a, b in zip(va, va[1:]))                           # Normierung A steigt zwischen n = 4 und 17


def test_gleicher_zufalls_seed_gleiche_ausgabe_und_anderer_seed_andere():
    a = S.simulate_blocks("exp", 10, 500, 123)
    b = S.simulate_blocks("exp", 10, 500, 123)
    c = S.simulate_blocks("exp", 10, 500, 124)
    assert np.array_equal(a.values, b.values)
    assert not np.array_equal(a.values, c.values)


def test_reihenfolge_der_bloecke_aendert_ks_und_mittel_nicht():
    sim = S.simulate_blocks("normal", 20, 1000, 5)
    perm = np.random.default_rng(0).permutation(sim.values)
    cdf = lambda x: D.cdf_max("normal", x, 20)
    assert S.ks_statistic(sim.values, cdf) == S.ks_statistic(perm, cdf)
    assert sim.values.mean() == pytest.approx(perm.mean(), rel=1e-12)


def test_gleicher_seed_gleiche_ks_fuer_alle_stetigen_verteilungen():
    """Die Umkehrung ist monoton: bei gleichem Zufalls-Seed liegen die Block-Maxima aller stetigen Verteilungen in derselben Rangordnung, KS gegen F^n ist deshalb gleich."""
    ks = []
    for key in ("exp", "gleich", "pareto2", "normal", "frechet"):
        sim = S.simulate_blocks(key, 15, 2000, 77)
        ks.append(S.ks_statistic(sim.values, lambda x, key=key: D.cdf_max(key, x, 15)))
    assert max(ks) - min(ks) < 1e-9


@pytest.mark.parametrize("key", CONTINUOUS)
def test_das_maximum_einer_teilmenge_ist_hoechstens_das_maximum_der_menge(key):
    rng = SplitMix64(99)
    for _ in range(30):
        u = [rng.random() for _ in range(20)]
        vals = D.inv_from_v(key, 1.0 - np.array(u))
        assert np.max(vals[:10]) <= np.max(vals) + 1e-15
        assert D.inv_from_v(key, np.array([1.0 - max(u)]))[0] == pytest.approx(np.max(vals), rel=1e-12)


@pytest.mark.parametrize("key", ("exp", "gleich", "pareto2", "frechet", "normal"))
def test_quantil_kehrt_f_hoch_n_um(key):
    rng = random.Random(5)
    for _ in range(20):
        n = rng.randint(1, 500)
        p = rng.choice([0.5, 0.1, 0.01, 0.001, 1e-5])
        x = D.quantile_max(key, n, p)
        assert float(D.cdf_max(key, np.array([x]), n)[0]) == pytest.approx(1 - p, rel=1e-8, abs=1e-12)


def test_grenzgesetz_laeuft_mit_n_auf_die_kurve_zu_exponential_gleich_pareto():
    z = {"exp": np.linspace(-1, 6, 200), "gleich": np.linspace(-6, 0, 200), "pareto2": np.linspace(0.3, 8, 200)}
    for key, zz in z.items():
        d = [float(np.max(np.abs(D.normed_cdf(key, n, zz) - D.limit_cdf(key, zz)))) for n in (5, 50, 500, 5000)]
        assert all(b < a for a, b in zip(d, d[1:]))

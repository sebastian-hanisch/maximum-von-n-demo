"""Die numpy-Messhilfe (SplitMix64 als Vektor in der Zähler-Form, `tools/mxn_make_reference.py`) ist bitgleich zur Vorlage `mxn_rng.py`."""

import mxn_make_reference as TOOL
import numpy as np

import mxn_distributions as D
import mxn_scenario as S
from mxn_rng import SplitMix64


def test_u64_5000_ausgaenge_wert_fuer_wert():
    r = SplitMix64(13)
    assert [r.next_u64() for _ in range(5000)] == TOOL.sm64_u64(13, 5000).tolist()


def test_random_3000_werte_exakt():
    r = SplitMix64(7)
    assert [r.random() for _ in range(3000)] == TOOL.sm64_unif(7, 3000).tolist()


def test_strom_aus_zwei_stuecken_gleich_dem_strom_am_stueck():
    assert np.array_equal(np.concatenate([TOOL.sm64_u64(5, 100), TOOL.sm64_u64(5, 50, 100)]), TOOL.sm64_u64(5, 150))


def test_bloecke_der_reinen_vorlage_und_des_vektors_bitgleich():
    v_pure = S.block_vmin(20, 300, 4242)
    v_vec = TOOL.block_vmin_np(4242, 300, 20)
    assert np.array_equal(v_pure, v_vec)


def test_normal_mit_stdlib_inv_cdf_gleich_scipy_ndtri():
    from scipy import special
    v = S.block_vmin(20, 300, 4242)
    mine = D.inv_from_v("normal", v)
    assert np.max(np.abs(mine - TOOL.values_np("normal", v))) < 1e-12
    grid = np.array([10.0 ** (-k / 2.0) for k in range(1, 41)])                   # bis p = 1e-20
    ref = -special.ndtri(grid)
    assert np.max(np.abs(D.inv_from_v("normal", grid) - ref) / np.maximum(1.0, np.abs(ref))) < 1e-12


def test_gleichverteilung_der_vorlage_mittel_und_varianz():
    u = TOOL.sm64_unif(99, 200000)
    assert abs(u.mean() - 0.5) < 4 * (1 / 12 / 200000) ** 0.5 and abs(u.var() - 1 / 12) < 4e-3

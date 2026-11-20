"""Szenario: Block-Maxima aus der reinen Zufalls-Vorlage (SplitMix64, `mxn_rng.py`, wörtliche Kopie).

Ein Block besteht aus n gleichverteilten Zahlen u_1 … u_n; das Größte der n Einzelwerte x_i = F⁻¹(u_i) ist F⁻¹(u_max) (die Umkehrung ist monoton). Ein Block
braucht also n gleichverteilte Zahlen und EINE Umkehrung; Normal und Pareto-Verteilung brauchen keine Box-Muller-Paare. Die Probe „n Umkehrungen, dann Maximum“
steht als Test (`brute_force_block_maxima`)."""

import time
from dataclasses import dataclass

import numpy as np

import mxn_constants as C
import mxn_distributions as D
from mxn_rng import SplitMix64


@dataclass(frozen=True)
class Simulation:
    key: str
    n: int
    R: int                 # tatsächlich simulierte Blöcke (nach Begrenzung durch n·R <= SIM_BUDGET)
    R_wish: int            # gewünschte Blöcke
    seed: int
    values: np.ndarray     # die R Block-Maxima, in der Reihenfolge der Blöcke
    seconds: float

    @property
    def clipped(self) -> bool:
        return self.R < self.R_wish


def effective_R(n: int, R: int, budget: int = C.SIM_BUDGET) -> int:
    """Begrenzung n·R <= budget (bei n = 365 also R <= 5 479); mindestens ein Block."""
    if n < 1 or float(n) != int(n):
        raise ValueError("n muss eine ganze Zahl von mindestens 1 sein")
    if R < 1 or float(R) != int(R):
        raise ValueError("R muss eine ganze Zahl von mindestens 1 sein")
    return max(1, min(int(R), budget // int(n)))


def block_vmin(n: int, R: int, seed: int) -> np.ndarray:
    """Kleinstes v = 1 − u_max je Block (v in (0, 1]), aus n·R Aufrufen von `random()` der reinen Vorlage."""
    if n < 1 or R < 1:
        raise ValueError("n und R müssen mindestens 1 sein")
    rng = SplitMix64(int(seed))
    rand = rng.random
    steps = range(int(n))
    umax = [max([rand() for _ in steps]) for _ in range(int(R))]
    return 1.0 - np.array(umax, dtype=float)


def simulate_blocks(key: str, n: int, R: int, seed: int) -> Simulation:
    """R Block-Maxima von je n unabhängigen Werten der Verteilung `key` (R wird auf SIM_BUDGET // n begrenzt)."""
    t0 = time.perf_counter()
    R_eff = effective_R(n, R)
    v = block_vmin(n, R_eff, seed)
    values = D.inv_from_v(key, v)
    return Simulation(key, int(n), R_eff, int(R), int(seed), values, time.perf_counter() - t0)


def brute_force_block_maxima(key: str, n: int, R: int, seed: int) -> np.ndarray:
    """Probe (Abkürzung gegen Brute Force): n Umkehrungen je Block, dann das Maximum; dieselben Zufallszahlen wie `simulate_blocks`."""
    rng = SplitMix64(int(seed))
    out = []
    for _ in range(int(R)):
        v = 1.0 - np.array([rng.random() for _ in range(int(n))], dtype=float)
        out.append(float(np.max(D.inv_from_v(key, v))))
    return np.array(out)


def ks_statistic(sample, cdf) -> float:
    """Kolmogorov-Abstand der empirischen Verteilungsfunktion der Stichprobe zur Verteilungsfunktion `cdf` (vektorisiert)."""
    s = np.sort(np.asarray(sample, dtype=float))
    m = s.size
    f = np.asarray(cdf(s), dtype=float)
    i = np.arange(1, m + 1)
    return float(max(np.max(i / m - f), np.max(f - (i - 1) / m)))


def ks_discrete(sample, cdf_right, cdf_left) -> float:
    """Kolmogorov-Abstand für eine diskrete Verteilung (Würfel): Vergleich an den Sprungstellen von rechts und links."""
    s = np.sort(np.asarray(sample, dtype=float))
    m = s.size
    vals, counts = np.unique(s, return_counts=True)
    emp_right = np.cumsum(counts) / m
    emp_left = emp_right - counts / m
    return float(max(np.max(np.abs(emp_right - cdf_right(vals))), np.max(np.abs(emp_left - cdf_left(vals)))))

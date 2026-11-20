"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import numpy as np
import plotly.graph_objects as go

import mxn_constants as C
import mxn_distributions as D
import mxn_evaluation as E
import mxn_reference as REF
from mxn_texts import de, n_label, p_text, thousands


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _layout(fig, title: str, xtitle: str, ytitle: str, height: int = 420):
    fig.update_layout(title=title, xaxis_title=xtitle, yaxis_title=ytitle, height=height, margin=dict(l=10, r=10, t=60, b=10),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
    return lock_axes(fig)


# ------------------------------------------------------------------------------------------------------------------------------------------
def build_sim(a: E.SimAnalysis, norm: str = "B") -> go.Figure:
    """Experiment 1: simulierte Verteilungsfunktion der R Block-Maxima gegen F(x)^n und das Grenzgesetz."""
    s = a.sim
    x, fn = E.exact_curve(s.key, s.n, norm)
    srt = np.sort(s.values)
    emp = np.searchsorted(srt, x, side="right") / srt.size
    step = "hv" if s.key == "wuerfel" else "linear"
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.round(x, 6), y=np.round(emp, 5), mode="lines", name=f"Simulation (R = {thousands(s.R)} Blöcke)",
                             line=dict(color=C.COLOR_SIM, width=3, shape=step)))
    fig.add_trace(go.Scatter(x=np.round(x, 6), y=np.round(fn, 6), mode="lines", name="exakt: F(x)ⁿ", line=dict(color=C.COLOR_EXACT, width=2, dash="dot", shape=step)))
    lim = E.limit_curve(s.key, s.n, x, norm)
    if lim is not None:
        fig.add_trace(go.Scatter(x=np.round(x, 6), y=np.round(lim, 6), mode="lines", name="Grenzgesetz (normiert)", line=dict(color=C.COLOR_LIMIT, width=2, dash="dash")))
    return _layout(fig, f"Das Größte von n = {n_label(s.n)}: {E.label(s.key)}", "x (Wert des Größten im Block)", "P(Mₙ ≤ x)")


def build_limit(a: E.LimitAnalysis, ref: dict) -> go.Figure:
    """Experiment 2: Abstand zum Grenzgesetz über n (doppelt-logarithmisch), eingefrorene Kurve plus der live gerechnete Punkt bei n."""
    cur = ref["curve"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cur["n"], y=cur["common"], mode="lines", name="Exponential, Gleich, Pareto (Standardnormierung)", line=dict(color=C.COLOR_EXACT, width=3)))
    fig.add_trace(go.Scatter(x=cur["n"], y=cur["B"], mode="lines", name="Normal, Normierung B", line=dict(color=C.COLOR_B, width=3)))
    fig.add_trace(go.Scatter(x=cur["n"], y=cur["A"], mode="lines", name="Normal, Normierung A", line=dict(color=C.COLOR_LIMIT, width=2, dash="dash")))
    fig.add_hline(y=C.KS_999 / np.sqrt(C.DEFAULT_R), line=dict(color=C.COLOR_OTHER, width=1, dash="dot"),
                  annotation_text="Rauschen bei R = 20.000 Blöcken", annotation_position="bottom right")
    pts = [("Standard", a.d_common, C.COLOR_EXACT), (f"Normal {a.norm}", a.d_sel, C.COLOR_B if a.norm == "B" else C.COLOR_LIMIT)]
    for name, d, col in pts:
        fig.add_trace(go.Scatter(x=[a.n], y=[d], mode="markers", name=f"n = {n_label(a.n)}: {name} {de(d, 5 if d < 0.001 else 4)}", marker=dict(size=12, color=col, line=dict(width=2, color="black"))))
    fig.update_xaxes(type="log")
    fig.update_yaxes(type="log")
    return _layout(fig, "Abstand zum Grenzgesetz (Kolmogorov) über n, exakt gerechnet", "n (Beobachtungen je Block)", "Kolmogorov-Abstand", height=460)


def build_type(key: str, n_sel: int, norm: str = "B") -> go.Figure:
    """Experiment 3: normiertes Maximum läuft mit wachsendem n auf die Grenzkurve zu."""
    ns = sorted({2, 5, 20, 100, 365} | {int(n_sel)})
    z, curves, lim = E.type_curves(key, ns, norm)
    fig = go.Figure()
    shades = np.linspace(0.25, 0.85, len(ns))
    for shade, n in zip(shades, ns):
        if n not in curves:
            continue
        sel = n == n_sel
        fig.add_trace(go.Scatter(x=np.round(z, 4), y=np.round(curves[n], 5), mode="lines", name=f"n = {n_label(n)}",
                                 line=dict(color=f"rgba(47,107,101,{shade:.2f})" if not sel else "rgba(76,120,168,1)", width=4 if sel else 2)))
    fig.add_trace(go.Scatter(x=np.round(z, 4), y=np.round(lim, 5), mode="lines", name="Grenzgesetz", line=dict(color=C.COLOR_LIMIT, width=3, dash="dash")))
    return _layout(fig, f"{E.label(key)}: normiertes Maximum {E.normalisation_text(key, norm)}", "z (normierte Größe)", "P(normiertes Maximum ≤ z)")


def method_cdf(n: int, method: str, x, ref: dict | None = None):
    """Verteilungsfunktion des gewählten Verfahrens an den Stellen x (Skala der Einzelwerte)."""
    x = np.asarray(x, dtype=float)
    if method == "gumbelA":
        c, d = D.normal_constants_A(n)
        return D.limit_cdf("normal", (x - d) / c)
    a, b = D.normal_constants_B(n)
    z = (x - b) / a
    if method == "gumbelB":
        return D.limit_cdf("normal", z)
    fit = REF.best_fit(n)
    if method == "gumbelbest":
        return D.gumbel_cdf_ab(z, fit["gumbel"]["mu"], fit["gumbel"]["sigma"])
    return D.gev_cdf(z, fit["gev"]["mu"], fit["gev"]["sigma"], fit["gev"]["xi"])


def build_pegel(a: E.PegelAnalysis, method: str) -> go.Figure:
    """Experiment 4: Überschreitungswahrscheinlichkeit je Block (logarithmisch) exakt gegen das gewählte Verfahren; der Pegel steht bei p."""
    lo = D.quantile_max("normal", a.n, 0.5)
    hi = D.quantile_max("normal", a.n, a.p / 30.0)
    x = np.linspace(lo, hi, 400)
    exact = -np.expm1(a.n * D.log_cdf("normal", x))
    approx = 1.0 - method_cdf(a.n, method, x)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.round(x, 5), y=exact, mode="lines", name="exakt: 1 − Φ(x)ⁿ", line=dict(color=C.COLOR_EXACT, width=3)))
    fig.add_trace(go.Scatter(x=np.round(x, 5), y=np.maximum(approx, 1e-12), mode="lines", name=C.METHOD_LABELS[method], line=dict(color=C.COLOR_LIMIT, width=2, dash="dash")))
    fig.add_hline(y=a.p, line=dict(color=C.COLOR_OTHER, width=1, dash="dot"), annotation_text=f"p = {p_text(a.p)}", annotation_position="top right")
    fig.add_trace(go.Scatter(x=[a.exact], y=[a.p], mode="markers", name=f"exakter Pegel {de(a.exact, 3)}", marker=dict(size=11, color=C.COLOR_EXACT, line=dict(width=2, color="black"))))
    lv = a.levels[method]
    fig.add_trace(go.Scatter(x=[lv], y=[a.p], mode="markers", name=f"Pegel des Verfahrens {de(lv, 3)}", marker=dict(size=11, symbol="diamond", color=C.COLOR_LIMIT, line=dict(width=2, color="black"))))
    fig.update_yaxes(type="log")
    return _layout(fig, f"Normalverteilung, n = {n_label(a.n)}: Überschreitungswahrscheinlichkeit des Block-Maximums", "x (in Standardabweichungen der Einzelwerte)", "P(Mₙ > x), logarithmisch")

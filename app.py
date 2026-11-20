"""Das Größte von n: wie sich ein Maximum verteilt - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Das Größte von n unabhängigen Werten hat die exakte Verteilung F(x)^n; ihr Grenzgesetz (Gumbel-, Weibull- oder Fréchet-Typ) ist nur eine Näherung,
und wie gut sie ist, unterscheidet sich zwischen den Einzelverteilungen (als Erklärung gilt der Rand). Vier Experimente: Simulation gegen F^n, Abstand zum Grenzgesetz (exakt gerechnet),
Skalieren und Typisieren, Pegel der Normalverteilung. Aufbau nach DEMO-PLAYBOOK §2.

Lauffähig mit: streamlit run app.py
"""

import pandas as pd
import streamlit as st

import mxn_constants as C
import mxn_distributions as D
import mxn_evaluation as E
import mxn_reference as REF
import mxn_scenario as S
import mxn_texts as T
from mxn_layout import dunkles_design, kopf_css
from mxn_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from mxn_texts import de, n_label, p_text, thousands
from mxn_visualization import build_limit, build_pegel, build_sim, build_type

st.set_page_config(page_title="Das Größte von n – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _simulate(key, n, R, seed):
    return S.simulate_blocks(key, n, R, seed)


@st.cache_data(show_spinner=False)
def _limit(n, norm):
    return E.analyse_limit(n, norm)


@st.cache_data(show_spinner=False)
def _pegel(n, p):
    return E.analyse_pegel(n, p)


def _d(x, nd):
    """Delta-Text mit ASCII-Vorzeichen: Streamlit liest nur '+' und '-' als Richtung (das echte Minuszeichen − färbt falsch)."""
    return f"{x:+.{nd}f}".replace(".", ",")


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in mxn_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("📈 Das Größte von n: wie sich ein Maximum verteilt")
st.markdown(T.INTRO)

with st.expander("So funktioniert das Maximum von n", expanded=True):
    st.markdown(T.HOW_IT_WORKS)

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Experiment 1 und 3: Verteilung und Block**")
    dist = st.selectbox("Einzelverteilung", C.DIST_KEYS, key="dist", format_func=lambda k: C.DIST_LABELS[k],
                        help="Aus welcher Verteilung die n Beobachtungen je Block stammen. Alle sind ausgedacht.")
    n = st.slider("Beobachtungen je Block n", *bounds("n_slider"), key="n_slider", help="Wie viele unabhängige Beobachtungen ein Block hat; das Größte davon ist das Block-Maximum.")
    R = st.slider("Simulierte Blöcke R", *bounds("R_slider"), step=C.R_STEP, key="R_slider",
                  help=f"Wie viele Blöcke simuliert werden. Begrenzung n·R ≤ {thousands(C.SIM_BUDGET)} Zufallszahlen (reine Python-Vorlage).")
    seed = st.number_input("Zufalls-Seed der Simulation", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neuen Zufalls-Seed würfeln", width="stretch", on_click=randomize_seed)
    st.markdown("**Experiment 2 und 3: Grenzgesetz**")
    n_lim = st.select_slider("n für den Abstand zum Grenzgesetz", options=C.N_LIM_OPTIONS, key="n_lim", format_func=n_label,
                             help="Exakt gerechnet, keine Simulation. Reicht bis n = 10¹².")
    norm = st.radio("Normierung der Normalverteilung", list(C.NORM_LABELS), key="norm", format_func=lambda k: C.NORM_LABELS[k],
                    help="Die Normierung gehört in jeden Satz: B ist die aus Φ⁻¹(1 − 1/n) gerechnete Normierung (ab n ≈ 7 näher am Grenzgesetz), A die Lehrbuch-Konstanten.")
    st.markdown("**Experiment 4: Pegel der Normalverteilung**")
    n_pegel = st.selectbox("Beobachtungen je Block n", C.N_PEGEL_OPTIONS, key="n_pegel", format_func=n_label,
                           help="Nur eingefrorene n: die Anpassung von „bestem Gumbel“ und „bester GEV“ ist live nicht zuverlässig.")
    p_exc = st.select_slider("Überschreitungswahrscheinlichkeit p je Block", options=C.P_OPTIONS, key="p_exc", format_func=p_text,
                             help="p = 0,01 ist der „100er-Wert“ eines Jahresmaximums (Lehrbuch-Lesart, ein Block ein Jahr).")
    method = st.selectbox("Verfahren", list(C.METHOD_LABELS), key="method", format_func=lambda k: C.METHOD_LABELS[k],
                          help="Gumbel mit ξ = 0 festgehalten gegen eine GEV mit freiem ξ.")

sync_query_params({"dist": dist, "n_slider": int(n), "R_slider": int(R), "seed_input": int(seed), "n_lim": int(n_lim), "norm": norm,
                   "n_pegel": int(n_pegel), "p_exc": float(p_exc), "method": method})

settings = E.Settings(dist, int(n), int(R), int(seed), int(n_lim), norm, int(n_pegel), float(p_exc), method)
ref = REF.load()

# --- Experiment 1: das Größte von n -----------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Experiment 1: Das Größte von n")
R_eff = S.effective_R(settings.n, settings.R)
laufzeit = settings.n * R_eff * 1.0e-6
if laufzeit >= 0.5:
    with st.spinner(f"Simuliere {thousands(settings.n * R_eff)} Zufallszahlen (rund {de(laufzeit, 1)} s, in der Cloud länger) …"):
        sim = _simulate(settings.dist, settings.n, settings.R, settings.seed)
else:
    sim = _simulate(settings.dist, settings.n, settings.R, settings.seed)
a1 = E.analyse_sim(sim)
if sim.clipped:
    st.info(f"R wird auf {thousands(sim.R)} Blöcke begrenzt: n·R darf höchstens {thousands(C.SIM_BUDGET)} Zufallszahlen betragen (reine Python-Vorlage, Laufzeit).")

m1, m2, m3 = st.columns(3)
# Delta immer als „mein Wert minus Referenz“; die Farbe regelt delta_color="inverse" (Vorzeichen-Falle, DEMO-PLAYBOOK §2).
m1.metric("Abstand Simulation zu F(x)ⁿ (Kolmogorov)", de(a1.ks, 4), delta=f"{_d(a1.ks - a1.eps, 4)} gegen Schranke {de(a1.eps, 4)}", delta_color="inverse")
if a1.mean_exact is not None:
    m2.metric("Mittel des Größten (Simulation)", de(a1.mean_sim, 3), delta=f"{_d(a1.mean_sim - a1.mean_exact, 3)} gegen exakt {de(a1.mean_exact, 3)}", delta_color="off")
else:
    m2.metric("Median des Größten (Simulation)", de(a1.median_sim, 3), delta=f"{_d(a1.median_sim - a1.median_exact, 3)} gegen exakt {de(a1.median_exact, 3)}", delta_color="off")
ti = D.type_info(settings.dist)
m3.metric("Typ des Grenzgesetzes", ti["typ"], delta=("ξ = " + de(ti["xi"], 2)) if ti["xi"] is not None else None, delta_color="off")
st.plotly_chart(build_sim(a1, settings.norm), width="stretch", key="sim_chart")
st.caption(f"Simulation: {thousands(sim.R)} Blöcke zu je n = {settings.n} Zufallszahlen, kleinstes v = 1 − u je Block und eine Umkehrung; Zufall aus SplitMix64. "
           f"Rechenzeit dieses Laufs {de(sim.seconds, 2)} s. Das Grenzgesetz ist normiert (Exponential: Verschiebung ln n; Normal: Normierung {settings.norm}).")

st.markdown("---")

# --- Kernfragen ----------------------------------------------------------------------------------------------------------------------------

lim = _limit(settings.n_lim, settings.norm)
peg = _pegel(settings.n_pegel, settings.p)
st.markdown("## ❓ Kernfragen")
st.markdown("**1. Wie ist das Größte von n verteilt, und stimmt F(x)ⁿ?**")
kind, text = E.verdict_sim(a1)
getattr(st, kind)(text)
st.markdown("**2. Wie nahe liegt das Grenzgesetz, und unterscheidet sich das zwischen den Einzelverteilungen?**")
kind, text = E.verdict_limit(lim, ref)
getattr(st, kind)(text)
st.markdown("**3. Was macht das mit dem Pegel?**")
kind, text = E.verdict_pegel(peg, settings.method)
getattr(st, kind)(text)

st.markdown("---")

# --- Experiment 2 --------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 2: Wie weit reicht das Grenzgesetz?")
st.caption("Regler: n für den Abstand und Normierung (Seitenleiste). Alles exakt gerechnet, keine Simulation; Zähleinheit n, Abstand = Kolmogorov-Abstand.")
st.plotly_chart(build_limit(lim, ref), width="stretch", key="limit_chart")
st.markdown("**Kleinstes n, ab dem der Abstand dauerhaft unter einer Marke liegt** (eingefroren, Zähleinheit n)")
st.dataframe(pd.DataFrame(E.threshold_rows(ref)), hide_index=True, width="stretch")
st.caption(T.THRESHOLD_NOTE)

st.markdown("---")

# --- Experiment 3 --------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 3: Skalieren und typisieren")
st.caption("Regler: Einzelverteilung, n und Normierung (Seitenleiste). Gezeigt wird die exakte Verteilung des normierten Maximums für mehrere n.")
if D.type_info(settings.dist)["family"] is not None and not (settings.dist == "normal" and settings.n < 2):
    st.plotly_chart(build_type(settings.dist, settings.n, settings.norm), width="stretch", key="type_chart")
kind, text = E.verdict_type(settings.dist, settings.n, settings.norm)
getattr(st, kind)(text)
st.markdown("**Landkarte der drei Typen**")
st.dataframe(pd.DataFrame(T.TYPE_TABLE), hide_index=True, width="stretch")

st.markdown("---")

# --- Experiment 4 --------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 4: Der Pegel, Gumbel gegen exakt")
st.caption("Regler: n, p und Verfahren (Seitenleiste); nur Normalverteilung. Einheit: Standardabweichungen der Einzelwerte.")
st.plotly_chart(build_pegel(peg, settings.method), width="stretch", key="pegel_chart")
st.dataframe(pd.DataFrame(E.pegel_rows(peg)), hide_index=True, width="stretch")

st.markdown("---")

with st.expander("Was bleibt offen?"):
    st.markdown(T.open_questions(ref))

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(T.ASSUMPTIONS)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(T.FORMULAS_INTRO)
    st.markdown(T.SYMBOLS)
    for block in T.FORMULA_BLOCKS:
        st.markdown(block)
    st.markdown(T.FORMULAS_OUTRO)

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Extremwerte: Blöcke, Schwelle, Pegel](https://sebastianhanisch.net/konzepte-extremwert.html)."
)

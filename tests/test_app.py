"""End-to-end Smoke-Test via Streamlits offiziellem AppTest-Framework: laedt app.py mit den
Standardeinstellungen, klickt jeden Preset-Button und faehrt jeden Slider an seine Grenzen und prueft, dass
kein Python-Fehler auftritt - insbesondere `streamlit.errors.StreamlitDuplicateElementId` (mehrere
st.plotly_chart-Aufrufe ohne eindeutiges key= koennen zufaellig identischen Inhalt rendern und kollidieren)
und `StreamlitAPIException` bei Slidern mit berechneten Grenzen (min == max)."""

import os

from streamlit.testing.v1 import AppTest

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))  # unter Last (parallele Läufe) reichen 180 s nicht immer; Umgebungsvariable zum Anpassen
AUTOPLAY_MARKER = "\u25b6"  # Auto-Play-Buttons (Animation mit sleep) im Smoke-Test auslassen


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def test_app_loads_without_exception():
    _fresh_app()


def test_every_button_click_does_not_raise():
    labels = [b.label for b in _fresh_app().button]
    assert labels, "keine Buttons gefunden"
    for index, label in enumerate(labels):
        if AUTOPLAY_MARKER in label:
            continue
        at = _fresh_app()
        at.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (label, [str(e) for e in at.exception])


def test_every_slider_at_its_min_and_max_does_not_raise():
    labels = [s.label for s in _fresh_app().slider]
    assert labels, "keine Slider gefunden"
    for index, label in enumerate(labels):
        for edge in ("min", "max"):
            at = _fresh_app()
            slider = at.slider[index]
            value = slider.min if edge == "min" else slider.max
            if isinstance(slider.value, int):
                value = int(value)
            slider.set_value(value).run(timeout=TIMEOUT)
            assert not at.exception, (label, edge, [str(e) for e in at.exception])


# --- Zusätzliche Tests dieser Demo (der Anfang der Datei ist die Vorlage des Portfolios) ----------------------------------------------------
import pytest  # noqa: E402

import mxn_constants as C  # noqa: E402


def _alle_meldungen(at) -> str:
    return " ".join(e.value for kind in (at.success, at.info, at.warning, at.error) for e in kind)


def test_app_zeigt_die_drei_kernfragen():
    at = _fresh_app()
    assert len(at.success) + len(at.info) + len(at.warning) >= 3
    texte = _alle_meldungen(at)
    assert "unter der Schranke" in texte and "Faktor 28,7" in texte and "exakt 4,033" in texte
    assert not at.error


@pytest.mark.parametrize("dist", list(C.DIST_KEYS))
def test_jede_einzelverteilung_laeuft_bei_kleinem_und_grossem_n(dist):
    for n in (C.N_MIN, C.N_MAX):
        at = _fresh_app()
        at.sidebar.selectbox(key="dist").set_value(dist).run(timeout=TIMEOUT)
        at.sidebar.slider(key="n_slider").set_value(n).run(timeout=TIMEOUT)
        assert not at.exception, (dist, n, [str(e) for e in at.exception])


def test_jeder_n_fuer_den_abstand_und_beide_normierungen_laufen():
    for norm in C.NORM_LABELS:
        for n_lim in C.N_LIM_OPTIONS:
            at = _fresh_app()
            at.sidebar.radio(key="norm").set_value(norm)
            at.sidebar.select_slider(key="n_lim").set_value(n_lim).run(timeout=TIMEOUT)
            assert not at.exception, (norm, n_lim, [str(e) for e in at.exception])


def test_jedes_n_jedes_p_und_jedes_verfahren_fuer_den_pegel_laufen():
    for n in C.N_PEGEL_OPTIONS:
        for p in C.P_OPTIONS:
            at = _fresh_app()
            at.sidebar.selectbox(key="n_pegel").set_value(n)
            at.sidebar.select_slider(key="p_exc").set_value(p).run(timeout=TIMEOUT)
            assert not at.exception, (n, p, [str(e) for e in at.exception])
    for method in C.METHOD_LABELS:
        at = _fresh_app()
        at.sidebar.selectbox(key="method").set_value(method).run(timeout=TIMEOUT)
        assert not at.exception, (method, [str(e) for e in at.exception])


def test_alle_minima_und_alle_maxima_zusammen():
    for edge in ("min", "max"):
        at = _fresh_app()
        for s in at.sidebar.slider:
            s.set_value(int(s.min if edge == "min" else s.max))
        at.sidebar.number_input(key="seed_input").set_value(0 if edge == "min" else C.SEED_MAX)
        at.sidebar.selectbox(key="dist").set_value("normal")
        at.sidebar.select_slider(key="n_lim").set_value(C.N_LIM_OPTIONS[0 if edge == "min" else -1])
        at.sidebar.selectbox(key="n_pegel").set_value(C.N_PEGEL_OPTIONS[0 if edge == "min" else -1])
        at.sidebar.select_slider(key="p_exc").set_value(C.P_OPTIONS[0 if edge == "min" else -1])
        at.run(timeout=TIMEOUT)
        assert not at.exception, (edge, [str(e) for e in at.exception])


def test_wuerfel_hat_im_experiment_3_keine_normierung_und_zeigt_das_in_der_meldung():
    at = _fresh_app()
    at.sidebar.selectbox(key="dist").set_value("wuerfel").run(timeout=TIMEOUT)
    assert not at.exception
    assert "Der Würfel hat kein Grenzgesetz" in _alle_meldungen(at)


def test_begrenzung_von_r_wird_gemeldet_bei_n_365_und_r_20000():
    at = _fresh_app()
    at.sidebar.slider(key="n_slider").set_value(365).run(timeout=TIMEOUT)
    assert "R wird auf 5.479 Blöcke begrenzt" in _alle_meldungen(at)


def test_expander_stehen_unter_ihrem_titel_nicht_per_index():
    at = _fresh_app()
    titel = [e.label for e in at.expander]
    assert "So funktioniert das Maximum von n" in titel and "📐 Mathematische Formulierung" in titel and "Was bleibt offen?" in titel
    formel = next(e for e in at.expander if e.label == "📐 Mathematische Formulierung")
    assert "Lesehilfe" in " ".join(m.value for m in formel.markdown)


def test_unterschriften_der_vier_experimente():
    at = _fresh_app()
    kopf = " ".join(h.value for h in at.subheader) + " " + " ".join(m.value for m in at.markdown)
    for z in ("Experiment 1", "Experiment 2", "Experiment 3", "Experiment 4", "Wo die Annahmen enden", "Kernfragen"):
        assert z in kopf, z

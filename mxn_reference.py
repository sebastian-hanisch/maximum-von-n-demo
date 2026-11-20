"""Lädt die eingefrorenen Tabellen `mxn_reference.json` (Quelle und Stand je Abschnitt stehen in der Datei; Erzeuger und Prüfung: `tools/mxn_make_reference.py`)."""

import json
from functools import lru_cache
from pathlib import Path

PATH = Path(__file__).resolve().parent / "mxn_reference.json"


@lru_cache(maxsize=1)
def load() -> dict:
    return json.loads(PATH.read_text(encoding="utf-8"))


def best_fit(n: int) -> dict:
    """Parameter von bestem Gumbel und bester GEV (z-Skala der Normierung B) für ein eingefrorenes n der Normalverteilung."""
    table = load()["best_fit"]["table"]
    if str(int(n)) not in table:
        raise KeyError(f"n = {n} ist nicht eingefroren (nur {', '.join(table)})")
    return table[str(int(n))]


def threshold(section: str, mark: float):
    """Kleinstes n (dauerhaft) für den Abstand unter `mark` aus der eingefrorenen Tabelle; None, wenn nicht erreicht."""
    return load()["thresholds"][section][str(mark)]

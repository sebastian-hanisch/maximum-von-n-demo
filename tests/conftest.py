"""Gemeinsame Einrichtung: Modulpfad, Zufalls-Seeds und die Pfade der Werkzeuge. Die unabhängigen Referenzen stehen in `helpers_ref.py`."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "tools"))


@pytest.fixture(scope="session")
def ref():
    import mxn_reference as REF
    return REF.load()

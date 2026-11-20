"""Die kopierte Datei mxn_rng.py (wörtlich, Quelle im Kopf der Datei): Vergleich mit der Vorlage, solange sie im Portfolio-Ordner liegt; plus feste Anker."""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SOURCES = {"mxn_rng.py": ("sebastianhanisch-website/tools/templates", "rng_splitmix64_exact.py")}


@pytest.mark.parametrize("copy", list(SOURCES))
def test_copy_is_verbatim_apart_from_the_header(copy):
    folder, original = SOURCES[copy]
    source = ROOT.parent / folder / original
    if not source.exists():
        pytest.skip("Original liegt nicht im Portfolio-Ordner")
    mine = (ROOT / copy).read_text(encoding="utf-8").splitlines()
    assert mine[0].startswith(f"# Wörtliche Kopie von {folder}/{original}")
    assert mine[1:] == source.read_text(encoding="utf-8").splitlines(), "die Kopie weicht von der Vorlage ab: Änderungen gehören in die Vorlage, danach neu kopieren"


def test_the_copy_runs_on_its_own_with_fixed_anchors():
    from mxn_rng import SplitMix64
    r = SplitMix64(0)
    assert [r.next_u64() for _ in range(3)] == [0xE220A8397B1DCDAF, 0x6E789E6AA1B965F4, 0x06C45D188009454F]
    r = SplitMix64(42)
    assert [r.random().hex() for _ in range(2)] == ["0x1.7bae644c5fd6dp-1", "0x1.477f199d93378p-3"]

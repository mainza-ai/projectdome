import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from src.animation.viseme_table import VisemeTable, VISEMES

def test_all_visemes_have_coefficients():
    table = VisemeTable()
    for v in VISEMES:
        coeffs = table.get_coefficients(v)
        assert len(coeffs) == 182, f"Viseme '{v}' should have 182-dim coefficients, got {len(coeffs)}"
        assert coeffs.dtype == np.float32
    print(f"  PASS: all_{len(VISEMES)}_visemes_have_coefficients")

def test_idle_is_zero():
    table = VisemeTable()
    idle = table.get_coefficients("IDLE")
    assert np.allclose(idle, np.zeros(182))
    print("  PASS: idle_is_zero")

def test_get_unknown_returns_idle():
    table = VisemeTable()
    unknown = table.get_coefficients("NONEXISTENT")
    assert np.allclose(unknown, np.zeros(182))
    print("  PASS: get_unknown_returns_idle")

def test_default_values():
    table = VisemeTable()
    aa = table.get_coefficients("aa")
    assert abs(aa[0] - 1.2) < 1e-5
    assert abs(aa[1] - (-0.5)) < 1e-5
    ee = table.get_coefficients("EE")
    assert abs(ee[2] - 1.5) < 1e-5
    th = table.get_coefficients("TH")
    assert abs(th[150] - 1.0) < 1e-5
    print("  PASS: default_values")

def test_set_and_get():
    table = VisemeTable()
    test_coeffs = np.random.randn(182).astype(np.float32)
    table.set_coefficients("PP", test_coeffs)
    retrieved = table.get_coefficients("PP")
    assert np.allclose(test_coeffs, retrieved)
    print("  PASS: set_and_get")

if __name__ == "__main__":
    print("VisemeTable tests:")
    test_all_visemes_have_coefficients()
    test_idle_is_zero()
    test_get_unknown_returns_idle()
    test_default_values()
    test_set_and_get()
    print("\nAll viseme table tests passed!")

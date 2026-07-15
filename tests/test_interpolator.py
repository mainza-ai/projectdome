import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from src.animation.viseme_table import VisemeTable
from src.animation.interpolator import VisemeInterpolator
from src.alignment.viseme_mapper import VisemeEvent

def make_timeline():
    return [
        VisemeEvent(name="PP", start_time=0.0, end_time=0.5),
        VisemeEvent(name="aa", start_time=0.5, end_time=1.0),
    ]

def test_before_first_event():
    table = VisemeTable()
    interp = VisemeInterpolator(table)
    timeline = make_timeline()
    coeffs = interp.get_coefficients(-0.1, timeline)
    assert len(coeffs) == 182
    assert np.allclose(coeffs, table.get_coefficients("IDLE"))
    print("  PASS: before_first_event")

def test_after_last_event():
    table = VisemeTable()
    interp = VisemeInterpolator(table)
    timeline = make_timeline()
    coeffs = interp.get_coefficients(2.0, timeline)
    assert len(coeffs) == 182
    assert np.allclose(coeffs, table.get_coefficients("IDLE"))
    print("  PASS: after_last_event")

def test_inside_event():
    table = VisemeTable()
    interp = VisemeInterpolator(table, ramp_duration=0.04)
    timeline = make_timeline()
    coeffs = interp.get_coefficients(0.25, timeline)
    assert np.allclose(coeffs, table.get_coefficients("PP"))
    print("  PASS: inside_event")

def test_empty_timeline():
    table = VisemeTable()
    interp = VisemeInterpolator(table)
    coeffs = interp.get_coefficients(0.5, [])
    assert np.allclose(coeffs, table.get_coefficients("IDLE"))
    print("  PASS: empty_timeline")

def test_gap_interpolation():
    table = VisemeTable()
    interp = VisemeInterpolator(table, ramp_duration=0.04)
    timeline = [
        VisemeEvent(name="PP", start_time=0.0, end_time=0.2),
        VisemeEvent(name="aa", start_time=0.4, end_time=0.6),
    ]
    mid_coeffs = interp.get_coefficients(0.3, timeline)
    pp_coeffs = table.get_coefficients("PP")
    aa_coeffs = table.get_coefficients("aa")
    expected = pp_coeffs + 0.5 * (aa_coeffs - pp_coeffs)
    assert np.allclose(mid_coeffs, expected, atol=1e-5)
    print("  PASS: gap_interpolation")

def test_ramp_transition():
    table = VisemeTable()
    interp = VisemeInterpolator(table, ramp_duration=0.1)
    timeline = [
        VisemeEvent(name="PP", start_time=0.0, end_time=0.5),
        VisemeEvent(name="aa", start_time=0.5, end_time=1.0),
    ]
    ramp_coeffs = interp.get_coefficients(0.47, timeline)
    pp_coeffs = table.get_coefficients("PP")
    aa_coeffs = table.get_coefficients("aa")
    expected = pp_coeffs + 0.3 * (aa_coeffs - pp_coeffs)
    assert np.allclose(ramp_coeffs, expected, atol=1e-5)
    print("  PASS: ramp_transition")

if __name__ == "__main__":
    print("VisemeInterpolator tests:")
    test_before_first_event()
    test_after_last_event()
    test_inside_event()
    test_empty_timeline()
    test_gap_interpolation()
    test_ramp_transition()
    print("\nAll interpolator tests passed!")

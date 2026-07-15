import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.alignment.viseme_mapper import VisemeMapper, VisemeEvent, PHONEME_TO_VISEME
from src.voice.provider import PhonemeEvent

def test_phoneme_to_viseme_mapping():
    mapper = VisemeMapper()
    assert PHONEME_TO_VISEME["P"] == "PP"
    assert PHONEME_TO_VISEME["B"] == "PP"
    assert PHONEME_TO_VISEME["F"] == "FF"
    assert PHONEME_TO_VISEME["AA"] == "aa"
    assert PHONEME_TO_VISEME["S"] == "SS"
    assert PHONEME_TO_VISEME[""] == "IDLE"
    assert PHONEME_TO_VISEME["SIL"] == "IDLE"
    print("  PASS: phoneme_to_viseme_mapping")

def test_viseme_merging():
    mapper = VisemeMapper()
    phonemes = [
        PhonemeEvent(phoneme="P", start_time=0.0, end_time=0.2),
        PhonemeEvent(phoneme="B", start_time=0.2, end_time=0.4),
        PhonemeEvent(phoneme="AA", start_time=0.4, end_time=0.6),
        PhonemeEvent(phoneme="M", start_time=0.6, end_time=0.8),
    ]
    visemes = mapper.map_phonemes(phonemes)
    assert len(visemes) == 2, f"Expected 2 merged visemes, got {len(visemes)}"
    assert visemes[0].name == "PP"
    assert visemes[0].start_time == 0.0
    assert visemes[0].end_time == 0.4
    assert visemes[1].name == "aa"
    assert visemes[1].start_time == 0.4
    assert visemes[1].end_time == 0.8
    print("  PASS: viseme_merging")

def test_empty_input():
    mapper = VisemeMapper()
    visemes = mapper.map_phonemes([])
    assert visemes == []
    print("  PASS: empty_input")

def test_stress_digit_stripping():
    mapper = VisemeMapper()
    phonemes = [
        PhonemeEvent(phoneme="AH0", start_time=0.0, end_time=0.1),
        PhonemeEvent(phoneme="AH1", start_time=0.1, end_time=0.2),
    ]
    visemes = mapper.map_phonemes(phonemes)
    assert len(visemes) == 1
    assert visemes[0].name == "schwa"
    print("  PASS: stress_digit_stripping")

def test_unknown_phoneme():
    mapper = VisemeMapper()
    phonemes = [
        PhonemeEvent(phoneme="ZZZ", start_time=0.0, end_time=0.1),
    ]
    visemes = mapper.map_phonemes(phonemes)
    assert visemes[0].name == "IDLE"
    print("  PASS: unknown_phoneme")

if __name__ == "__main__":
    print("VisemeMapper tests:")
    test_phoneme_to_viseme_mapping()
    test_viseme_merging()
    test_empty_input()
    test_stress_digit_stripping()
    test_unknown_phoneme()
    print("\nAll viseme mapper tests passed!")

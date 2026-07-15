import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from src.animation.emotion_blender import EmotionBlender

def test_neutral_emotion_zeros():
    blender = EmotionBlender()
    coeffs = blender.get_emotion_coefficients(None)
    assert coeffs.shape == (383,)
    assert np.allclose(coeffs, np.zeros(383))
    print("  PASS: neutral_emotion_zeros")

def test_unknown_emotion_zeros():
    blender = EmotionBlender()
    coeffs = blender.get_emotion_coefficients("BOGUS_EMOTION")
    assert coeffs.shape == (383,)
    assert np.allclose(coeffs, np.zeros(383))
    print("  PASS: unknown_emotion_zeros")

def test_happy_emotion():
    blender = EmotionBlender()
    coeffs = blender.get_emotion_coefficients("HAPPY", intensity=1.0)
    assert coeffs.shape == (383,)
    assert not np.allclose(coeffs, np.zeros(383)), "HAPPY expression should produce non-zero coefficients"
    print("  PASS: happy_emotion")

def test_sad_maps_to_corners_down():
    blender = EmotionBlender()
    sad = blender.get_emotion_coefficients("SAD", intensity=1.0)
    corners = blender.get_emotion_coefficients("CORNERS_DOWN", intensity=1.0)
    assert np.allclose(sad, corners), "SAD should map to CORNERS_DOWN"
    print("  PASS: sad_maps_to_corners_down")

def test_blend_dimensions():
    blender = EmotionBlender()
    speech = np.random.randn(182).astype(np.float32)
    emotion = np.random.randn(383).astype(np.float32)
    blended = blender.blend(speech, emotion)
    assert blended.shape == (383,)
    print("  PASS: blend_dimensions")

def test_blend_anatomy():
    blender = EmotionBlender()
    speech = np.ones(182, dtype=np.float32)
    emotion = np.ones(383, dtype=np.float32)
    blended = blender.blend(speech, emotion)
    assert np.allclose(blended[0:200], emotion[0:200]), "Upper face should be full emotion"
    assert np.allclose(blended[350:382], speech[150:182]), "Tongue should be full speech"
    assert np.allclose(blended[382], emotion[382]), "Pupils should be full emotion"
    assert np.allclose(blended[200:350], speech[0:150] + 0.3 * emotion[200:350]), "Lower face should be speech + 0.3*emotion"
    print("  PASS: blend_anatomy")

if __name__ == "__main__":
    print("EmotionBlender tests:")
    test_neutral_emotion_zeros()
    test_unknown_emotion_zeros()
    test_happy_emotion()
    test_sad_maps_to_corners_down()
    test_blend_dimensions()
    test_blend_anatomy()
    print("\nAll emotion blender tests passed!")

import numpy as np
from typing import Dict, Optional
from gnm.shape.semantic_sampler import ExpressionSampler, Expression

class EmotionBlender:
    def __init__(self):
        print("Loading ExpressionSampler model...")
        self.sampler = ExpressionSampler()
        print("ExpressionSampler loaded.")

    def get_emotion_coefficients(self, emotion_name: Optional[str], intensity: float = 1.0) -> np.ndarray:
        """Get 383-dimensional GNM expression coefficients for the given emotion name and intensity."""
        if not emotion_name:
            return np.zeros(383, dtype=np.float32)

        # Convert string to Expression enum
        try:
            enum_val = Expression[emotion_name.upper().strip()]
        except KeyError:
            print(f"Warning: Unknown emotion '{emotion_name}'. Using neutral (all-zeros) pose.")
            return np.zeros(383, dtype=np.float32)

        # sample_expression returns (1, 383)
        raw_coeffs = self.sampler.sample_expression(enum_val).squeeze(0)
        return raw_coeffs * intensity

    def blend(self, speech_coeffs: np.ndarray, emotion_coeffs: np.ndarray) -> np.ndarray:
        """Blend 182-dim speech coefficients with 383-dim emotion coefficients additively.
        
        Speech vector (182-dim):
        - Index 0-149: lower_face_region_000 to lower_face_region_149
        - Index 150-181: tongue_mean + tongue_000 to tongue_030
        
        Emotion vector (383-dim):
        - Index 0-99: left_eye_region_000 to left_eye_region_099
        - Index 100-199: right_eye_region_000 to right_eye_region_099
        - Index 200-349: lower_face_region_000 to lower_face_region_149
        - Index 350-381: tongue
        - Index 382: pupils
        """
        assert len(speech_coeffs) == 182, f"Speech coefficients must be 182-dimensional, got {len(speech_coeffs)}"
        assert len(emotion_coeffs) == 383, f"Emotion coefficients must be 383-dimensional, got {len(emotion_coeffs)}"

        final_coeffs = np.zeros(383, dtype=np.float32)

        # 1. Upper face & Eyes (0 to 199): driven fully by emotion
        final_coeffs[0:200] = emotion_coeffs[0:200]

        # 2. Lower face (200 to 349): driven by speech, with scaled additive emotion overlap (0.3 weight)
        final_coeffs[200:350] = speech_coeffs[0:150] + 0.3 * emotion_coeffs[200:350]

        # 3. Tongue (350 to 381): driven fully by speech
        final_coeffs[350:382] = speech_coeffs[150:182]

        # 4. Pupils (382): driven fully by emotion
        final_coeffs[382] = emotion_coeffs[382]

        return final_coeffs

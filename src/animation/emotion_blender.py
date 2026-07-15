import os
import warnings
import numpy as np
from typing import Dict, Optional
from gnm.shape.semantic_sampler import ExpressionSampler, Expression
import tensorflow as tf

tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class _SilentExpressionSampler(ExpressionSampler):
    def __init__(self, decoder_model_path=None, verbose=False):
        if decoder_model_path is None:
            from gnm.shape.semantic_sampler import _EXPRESSION_DECODER_PATH
            decoder_model_path = _EXPRESSION_DECODER_PATH
        self._decoder = tf.keras.models.load_model(str(decoder_model_path), compile=False)
        self._expression_names = tuple(member.name.lower() for member in Expression)
        self._num_classes = self._decoder.inputs[1].shape[-1]
        self._latent_dim = self._decoder.inputs[0].shape[-1]

class EmotionBlender:
    def __init__(self):
        self.sampler = _SilentExpressionSampler()

    def get_emotion_coefficients(self, emotion_name: Optional[str], intensity: float = 1.0) -> np.ndarray:
        if not emotion_name:
            return np.zeros(383, dtype=np.float32)
        try:
            name = emotion_name.upper().strip()
            if name == "SAD":
                name = "CORNERS_DOWN"
            if name == "FEAR":
                name = "SURPRISE"
            if name == "ANGRY":
                name = "SNARL"
            enum_val = Expression[name]
        except KeyError:
            return np.zeros(383, dtype=np.float32)
        raw_coeffs = self.sampler.sample_expression(enum_val).squeeze(0)
        return raw_coeffs * intensity

    def blend(self, speech_coeffs: np.ndarray, emotion_coeffs: np.ndarray) -> np.ndarray:
        assert len(speech_coeffs) == 182, f"Speech coefficients must be 182-dimensional, got {len(speech_coeffs)}"
        assert len(emotion_coeffs) == 383, f"Emotion coefficients must be 383-dimensional, got {len(emotion_coeffs)}"
        final_coeffs = np.zeros(383, dtype=np.float32)
        final_coeffs[0:200] = emotion_coeffs[0:200]
        final_coeffs[200:350] = speech_coeffs[0:150] + 0.3 * emotion_coeffs[200:350]
        final_coeffs[350:382] = speech_coeffs[150:182]
        final_coeffs[382] = emotion_coeffs[382]
        return final_coeffs

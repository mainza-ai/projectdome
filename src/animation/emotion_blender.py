import os
import warnings
import logging
import numpy as np
from typing import Dict, Optional
from gnm.shape.semantic_sampler import ExpressionSampler, Expression
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
tf.get_logger().setLevel('ERROR')
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)

warnings.filterwarnings('ignore', message='No training configuration found')

class _SilentExpressionSampler(ExpressionSampler):
    def __init__(self, decoder_model_path=None, verbose=False):
        if decoder_model_path is None:
            from gnm.shape.semantic_sampler import _EXPRESSION_DECODER_PATH
            decoder_model_path = _EXPRESSION_DECODER_PATH
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self._decoder = tf.keras.models.load_model(str(decoder_model_path), compile=False)
        self._expression_names = tuple(member.name.lower() for member in Expression)
        self._num_classes = self._decoder.inputs[1].shape[-1]
        self._latent_dim = self._decoder.inputs[0].shape[-1]

class _SilentIdentitySampler:
    def __init__(self, decoder_model_path=None):
        if decoder_model_path is None:
            from gnm.shape.semantic_sampler import _IDENTITY_DECODER_PATH
            decoder_model_path = _IDENTITY_DECODER_PATH
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self._decoder = tf.keras.models.load_model(str(decoder_model_path), compile=False)
        self._condition_dim = self._decoder.inputs[1].shape[-1]
        self._LATENT_DIM = 64
        self._NUM_GENDER_CLASSES = 2
        self._NUM_ETHNICITIES_CLASSES = 4
        self._GENDER_LABEL_MAP = {0: 'Female', 1: 'Male'}
        self._ETHNICITY_LABEL_MAP = {0: 'Middle Eastern', 1: 'Asian', 2: 'White', 3: 'Black'}

    def sample_identity(self, gender_class, ethnicity_class, num_samples=1, rng=None, verbose=False):
        import numpy as np
        from gnm.shape.semantic_sampler import _create_combined_one_hot_labels
        raw_label_combo = np.array([[int(gender_class), int(ethnicity_class)]])
        combined_ohe_label = _create_combined_one_hot_labels(
            raw_label_combo, self._NUM_GENDER_CLASSES, self._NUM_ETHNICITIES_CLASSES
        )
        labels_for_decoder = np.repeat(combined_ohe_label, num_samples, axis=0)
        rng = rng if rng is not None else np.random.default_rng()
        z_sample = rng.normal(size=(num_samples, self._LATENT_DIM)).astype('float32')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            generated_vectors = self._decoder.predict([z_sample, labels_for_decoder], verbose=0)
        return generated_vectors

    def explain_classes(self):
        return {'gender': self._GENDER_LABEL_MAP, 'ethnicity': self._ETHNICITY_LABEL_MAP}

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
        speech_lower = speech_coeffs[0:150]
        emotion_lower = emotion_coeffs[200:350] * 0.15
        speech_mag = np.abs(speech_lower)
        emotion_mag = np.abs(emotion_lower)
        final_coeffs[200:350] = np.where(speech_mag > emotion_mag, speech_lower, emotion_lower)
        final_coeffs[350:382] = speech_coeffs[150:182]
        final_coeffs[382] = emotion_coeffs[382]
        return final_coeffs

import os
import hashlib
import urllib.request
import logging
import numpy as np
from piper.voice import PiperVoice
from src.voice.provider import VoiceProvider, VoiceResult

log = logging.getLogger('dome')

PIPER_CHECKSUMS = {
    'en_US-lessac-medium.onnx': '3e7c0c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c',
    'en_US-lessac-medium.onnx.json': 'd4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9',
}

class PiperProvider(VoiceProvider):
    def __init__(self, model_dir: str = "data/voice", voice_name: str = "en_US-lessac-medium"):
        self.model_path = os.path.join(model_dir, f"{voice_name}.onnx")
        self.config_path = os.path.join(model_dir, f"{voice_name}.onnx.json")
        self._ensure_downloaded(model_dir, voice_name)
        log.info(f"Loading Piper voice model from {self.model_path}...")
        self.voice = PiperVoice.load(self.model_path)
        log.info("Piper model loaded.")
        self._cache = {}
        self._cache_max = 64

    def _ensure_downloaded(self, model_dir, voice_name):
        if os.path.exists(self.model_path) and os.path.exists(self.config_path):
            onnx_size = os.path.getsize(self.model_path)
            if onnx_size > 1000000:
                return
        log.info(f"Downloading Piper model {voice_name} to {model_dir}...")
        base_url = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/"
        os.makedirs(model_dir, exist_ok=True)
        try:
            urllib.request.urlretrieve(base_url + f"{voice_name}.onnx", self.model_path)
            urllib.request.urlretrieve(base_url + f"{voice_name}.onnx.json", self.config_path)
            onnx_size = os.path.getsize(self.model_path)
            if onnx_size < 1000000:
                raise RuntimeError(f"Downloaded file too small ({onnx_size} bytes), likely corrupted")
            log.info(f"Downloaded {onnx_size/1024/1024:.0f} MB")
        except Exception as e:
            log.error(f"Download failed: {e}")
            raise

    def synthesize(self, text: str) -> VoiceResult:
        if text in self._cache:
            self._cache[text]['count'] = self._cache[text].get('count', 0) + 1
            return self._cache[text]['result']

        chunks = list(self.voice.synthesize(text))
        audio_arrays = [c.audio_float_array for c in chunks if c.audio_float_array is not None]
        if not audio_arrays:
            result = VoiceResult(audio=np.zeros(0, dtype=np.float32), sample_rate=self.voice.config.sample_rate, phoneme_events=[])
        else:
            result = VoiceResult(audio=np.concatenate(audio_arrays), sample_rate=self.voice.config.sample_rate, phoneme_events=[])

        if len(text) < 200:
            self._cache[text] = {'result': result, 'count': 1}
            if len(self._cache) > self._cache_max:
                oldest = min(self._cache.keys(), key=lambda k: self._cache[k].get('count', 0))
                del self._cache[oldest]

        return result

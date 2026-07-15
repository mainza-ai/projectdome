import os
import urllib.request
import numpy as np
from typing import List
from piper.voice import PiperVoice
from src.voice.provider import VoiceProvider, VoiceResult, PhonemeEvent

class PiperProvider(VoiceProvider):
    def __init__(self, model_dir: str = "data/voice", voice_name: str = "en_US-lessac-medium"):
        self.model_path = os.path.join(model_dir, f"{voice_name}.onnx")
        self.config_path = os.path.join(model_dir, f"{voice_name}.onnx.json")
        
        # Download models if they don't exist
        if not os.path.exists(self.model_path) or not os.path.exists(self.config_path):
            print(f"Piper model not found locally. Downloading to {model_dir}...")
            base_url = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/"
            os.makedirs(model_dir, exist_ok=True)
            
            urllib.request.urlretrieve(base_url + f"{voice_name}.onnx", self.model_path)
            urllib.request.urlretrieve(base_url + f"{voice_name}.onnx.json", self.config_path)
            print("Download completed.")

        print(f"Loading Piper voice model from {self.model_path}...")
        self.voice = PiperVoice.load(self.model_path)
        print("Piper model loaded.")

    def synthesize(self, text: str) -> VoiceResult:
        """Synthesize text into raw float32 audio."""
        chunks = list(self.voice.synthesize(text))
        
        # Concatenate audio chunks from all sentences
        audio_arrays = [chunk.audio_float_array for chunk in chunks if chunk.audio_float_array is not None]
        if not audio_arrays:
            return VoiceResult(audio=np.zeros(0, dtype=np.float32), sample_rate=self.voice.config.sample_rate, phoneme_events=[])
        
        full_audio = np.concatenate(audio_arrays)
        return VoiceResult(
            audio=full_audio,
            sample_rate=self.voice.config.sample_rate,
            phoneme_events=[]  # Will be populated by forced alignment
        )

import numpy as np
from typing import List, Tuple
from src.voice.piper_provider import PiperProvider
from src.alignment.mfa_aligner import Wav2TextGridAligner
from src.alignment.viseme_mapper import VisemeMapper, VisemeEvent

class AcousticPipeline:
    def __init__(self):
        self.voice = PiperProvider()
        self.aligner = Wav2TextGridAligner()
        self.mapper = VisemeMapper()

    def process(self, text: str) -> Tuple[np.ndarray, int, List[VisemeEvent]]:
        """Process text to return audio waveform, sample rate, and viseme timeline."""
        # 1. Synthesize speech audio
        voice_result = self.voice.synthesize(text)
        
        # 2. Get phoneme timestamps via forced alignment
        phonemes = self.aligner.align(voice_result.audio, voice_result.sample_rate, text)
        
        # 3. Reduce phonemes to visemes
        visemes = self.mapper.map_phonemes(phonemes)
        
        return voice_result.audio, voice_result.sample_rate, visemes

if __name__ == "__main__":
    import argparse
    import os
    import soundfile as sf

    parser = argparse.ArgumentParser(description="Test Acoustic Pipeline (Piper + Wav2TextGrid)")
    parser.add_argument("--text", type=str, default="Hello world, welcome to Milimo Quantum.", help="Text to speak")
    parser.add_argument("--out", type=str, default="output/test_speech.wav", help="Output WAV file path")
    args = parser.parse_args()

    print(f"Initializing pipeline with text: '{args.text}'")
    pipeline = AcousticPipeline()
    audio, sr, visemes = pipeline.process(args.text)

    # Save audio
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    sf.write(args.out, audio, sr)
    print(f"Saved synthesized audio to {args.out}")

    # Print viseme timeline
    print("\nGenerated Viseme Timeline:")
    print("-" * 50)
    for v in visemes:
        print(f"{v.name:<8} | Start: {v.start_time:6.3f}s | End: {v.end_time:6.3f}s | Duration: {v.end_time - v.start_time:6.3f}s")
    print("-" * 50)

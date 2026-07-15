import os
import torch
import torchaudio
import soundfile as sf
import numpy as np
from Wav2TextGrid.aligner_core.xvec_extractor import xVecExtractor
from Wav2TextGrid.aligner_core.aligner import xVecSAT_forced_aligner
from src.voice.provider import PhonemeEvent

class Wav2TextGridAligner:
    def __init__(self):
        print("Loading Wav2TextGrid alignment model...")
        self.xvx = xVecExtractor(method='xvector')
        self.aligner = xVecSAT_forced_aligner('pkadambi/Wav2TextGrid', satvector_size=512)
        print("Wav2TextGrid loaded successfully.")

    def align(self, audio_data: np.ndarray, sample_rate: int, text: str) -> list[PhonemeEvent]:
        """Align 1D PCM float32 audio against transcription text."""
        temp_wav = "output/temp_alignment.wav"
        os.makedirs("output", exist_ok=True)
        
        # Resample to 16000 if needed (Wav2TextGrid requires exactly 16kHz)
        if sample_rate != 16000:
            audio_tensor = torch.tensor(audio_data, dtype=torch.float32).unsqueeze(0)
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            resampled_tensor = resampler(audio_tensor)
            resampled_audio = resampled_tensor.squeeze(0).numpy()
        else:
            resampled_audio = audio_data

        # Save to temporary 16kHz WAV file
        sf.write(temp_wav, resampled_audio, 16000)

        # Extract x-vector
        xvector = self.xvx.extract_xvector(temp_wav)
        xvector = xvector[0][0].view(1, -1)
        if torch.cuda.is_available():
            xvector = xvector.cuda()

        # Perform alignment
        # phones is a list of tuples: (start_time, end_time, phone)
        phones, _ = self.aligner.align(temp_wav, text, xvector)
        
        # Cleanup
        if os.path.exists(temp_wav):
            try:
                os.remove(temp_wav)
            except OSError:
                pass

        # Map to PhonemeEvent
        events = []
        for start, end, phone in phones:
            events.append(PhonemeEvent(phoneme=phone, start_time=start, end_time=end))
        return events

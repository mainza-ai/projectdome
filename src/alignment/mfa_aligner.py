import os
import io
import tempfile
import torch
import torchaudio
import soundfile as sf
import numpy as np
from Wav2TextGrid.aligner_core.xvec_extractor import xVecExtractor
from Wav2TextGrid.aligner_core.aligner import xVecSAT_forced_aligner
from src.voice.provider import PhonemeEvent
import logging

log = logging.getLogger('dome')

_aligner_instance = None
_xvx_instance = None

def get_aligner():
    global _aligner_instance, _xvx_instance
    if _aligner_instance is None:
        log.info("Loading Wav2TextGrid alignment model...")
        _xvx_instance = xVecExtractor(method='xvector')
        _aligner_instance = xVecSAT_forced_aligner('pkadambi/Wav2TextGrid', satvector_size=512)
        log.info("Wav2TextGrid loaded.")
    return _xvx_instance, _aligner_instance

class Wav2TextGridAligner:
    def __init__(self):
        self.xvx, self.aligner = get_aligner()

    def align(self, audio_data: np.ndarray, sample_rate: int, text: str) -> list[PhonemeEvent]:
        if sample_rate != 16000:
            audio_tensor = torch.tensor(audio_data, dtype=torch.float32).unsqueeze(0)
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            audio_tensor = resampler(audio_tensor)
            audio_16k = audio_tensor.squeeze(0).numpy()
        else:
            audio_16k = audio_data

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path = tmp.name
            sf.write(tmp_path, audio_16k, 16000)

        try:
            with torch.no_grad():
                xvector = self.xvx.extract_xvector(tmp_path)
                xvector = xvector[0][0].view(1, -1)
                if torch.cuda.is_available():
                    xvector = xvector.cuda()
                phones, _ = self.aligner.align(tmp_path, text, xvector)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        return [PhonemeEvent(phoneme=p, start_time=s, end_time=e) for s, e, p in phones]

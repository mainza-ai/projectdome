from typing import Protocol, List
from dataclasses import dataclass
import numpy as np

@dataclass
class PhonemeEvent:
    phoneme: str
    start_time: float  # in seconds
    end_time: float    # in seconds

@dataclass
class VoiceResult:
    audio: np.ndarray  # PCM float32
    sample_rate: int
    phoneme_events: List[PhonemeEvent]

class VoiceProvider(Protocol):
    def synthesize(self, text: str) -> VoiceResult:
        """Synthesize speech and return audio + phoneme timing."""
        ...

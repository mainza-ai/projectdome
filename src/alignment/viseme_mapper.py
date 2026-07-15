from dataclasses import dataclass
from typing import List
from src.voice.provider import PhonemeEvent

@dataclass
class VisemeEvent:
    name: str
    start_time: float  # seconds
    end_time: float    # seconds

# Phoneme-to-viseme reduction map (keys in UPPERCASE)
PHONEME_TO_VISEME = {
    # Silence/Idle
    "": "IDLE", "SIL": "IDLE", "SP": "IDLE", "SILENCE": "IDLE", "NONE": "IDLE",
    
    # Bilabial: P, B, M
    "P": "PP", "B": "PP", "M": "PP",
    
    # Labiodental: F, V
    "F": "FF", "V": "FF",
    
    # Dental: TH, DH
    "TH": "TH", "DH": "TH",
    
    # Alveolar: T, D, N, L
    "T": "DD", "D": "DD", "N": "DD", "L": "DD",
    
    # Postalveolar: SH, ZH, CH, JH
    "SH": "CH", "ZH": "CH", "CH": "CH", "JH": "CH",
    
    # Velar/Glottal: K, G, NG, HH
    "K": "kk", "G": "kk", "NG": "kk", "HH": "kk",
    
    # Alveolar sibilant: S, Z
    "S": "SS", "Z": "SS",
    
    # Retroflex/Semi-vowel: R, ER
    "R": "RR", "ER": "RR", "W": "OO", "Y": "EE",
    
    # Open Vowels: AA, AE, AH, AO
    "AA": "aa", "AE": "aa", "AH": "aa", "AO": "aa",
    
    # Front Vowels: EH, IH, IY, AY, EY
    "EH": "EE", "IH": "EE", "IY": "EE", "AY": "EE", "EY": "EE",
    
    # Rounded Vowels: UW, UH, OW, OY, AW
    "UW": "OO", "UH": "OO", "OW": "OO", "OY": "OO", "AW": "OO",
    
    # Neutral/Schwa
    "AX": "schwa", "AH0": "schwa", "AXR": "schwa", "SCHWA": "schwa",
}

class VisemeMapper:
    def map_phonemes(self, phonemes: List[PhonemeEvent]) -> List[VisemeEvent]:
        """Convert a list of phoneme events to mapped viseme events, merging duplicates."""
        visemes = []
        for p in phonemes:
            # Clean phoneme name (convert to uppercase, strip stress digits)
            clean_phone = p.phoneme.upper().strip()
            clean_phone = ''.join([c for c in clean_phone if not c.isdigit()])
            
            # Map to viseme
            viseme_name = PHONEME_TO_VISEME.get(clean_phone, "IDLE")
            
            visemes.append(VisemeEvent(
                name=viseme_name,
                start_time=p.start_time,
                end_time=p.end_time
            ))
            
        # Merge consecutive identical visemes
        merged_visemes = []
        for v in visemes:
            if not merged_visemes:
                merged_visemes.append(v)
            else:
                prev = merged_visemes[-1]
                if prev.name == v.name:
                    # Extend end time
                    prev.end_time = v.end_time
                else:
                    merged_visemes.append(v)
        return merged_visemes

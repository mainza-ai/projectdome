import os
import json
import numpy as np
from typing import Dict

VISEMES = [
    "IDLE",   # Silence / rest
    "PP",     # Bilabial: P, B, M
    "FF",     # Labiodental: F, V
    "TH",     # Dental: TH, DH
    "DD",     # Alveolar: T, D, N, L
    "CH",     # Postalveolar: SH, ZH, CH, JH
    "kk",     # Velar/Glottal: K, G, NG, HH
    "SS",     # Sibilant: S, Z
    "RR",     # Retroflex: R, ER
    "aa",     # Open vowels: AA, AE, AH
    "EE",     # Front vowels: EH, IH, IY, AY, EY
    "OO",     # Rounded vowels: UW, UH, OW, OY, AW
    "schwa"   # Neutral: AX, AH0
]

class VisemeTable:
    def __init__(self, filepath: str = "data/viseme_table.json"):
        self.filepath = filepath
        self.table: Dict[str, np.ndarray] = {}
        self.initialize_defaults()
        if os.path.exists(self.filepath):
            self.load()

    def initialize_defaults(self):
        """Initialize default 182-dimensional coefficients for each viseme."""
        # 182 dims: 150 for lower_face_region, 32 for tongue
        for name in VISEMES:
            self.table[name] = np.zeros(182, dtype=np.float32)

        # Set simple PCA-based placeholders for visible deformations
        # In GNM:
        # Index 0-149 of our 182 vector maps to lower_face_region_000 to lower_face_region_149.
        # Index 150-181 maps to tongue_mean + tongue_000 to tongue_030.
        
        # open mouth / jaw drop
        self.table["aa"][0] = 1.2    # Lower face component 0
        self.table["aa"][1] = -0.5
        
        # pucker / rounded lips
        self.table["OO"][0] = 0.5
        self.table["OO"][1] = 1.5    # Lower face component 1
        
        # wide smile/lips spread
        self.table["EE"][2] = 1.5    # Lower face component 2
        
        # closed lips pressed (PP)
        self.table["PP"][0] = -0.5
        self.table["PP"][3] = 1.0    # Lower face component 3
        
        # dental (TH) - tongue forward
        self.table["TH"][0] = 0.3
        self.table["TH"][150] = 1.0  # tongue_mean index

    def get_coefficients(self, name: str) -> np.ndarray:
        """Return 182-dim coefficient vector for the given viseme name."""
        return self.table.get(name, self.table["IDLE"])

    def set_coefficients(self, name: str, coeffs: np.ndarray):
        assert len(coeffs) == 182, "Coefficients must be 182-dimensional"
        self.table[name] = np.array(coeffs, dtype=np.float32)

    def load(self):
        try:
            with open(self.filepath, 'r') as f:
                data = json.load(f)
                for name, coeffs in data.items():
                    if name in self.table:
                        self.table[name] = np.array(coeffs, dtype=np.float32)
            print(f"Loaded viseme table from {self.filepath}")
        except Exception as e:
            print(f"Error loading viseme table from {self.filepath}: {e}")

    def save(self):
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        try:
            # Convert numpy arrays to lists for JSON serialization
            data = {name: coeffs.tolist() for name, coeffs in self.table.items()}
            with open(self.filepath, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Saved viseme table to {self.filepath}")
        except Exception as e:
            print(f"Error saving viseme table to {self.filepath}: {e}")

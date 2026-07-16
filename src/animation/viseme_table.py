import os
import json
import numpy as np
from typing import Dict

VISEMES = [
    "IDLE",
    "PP",
    "FF",
    "TH",
    "DD",
    "CH",
    "kk",
    "SS",
    "RR",
    "aa",
    "EE",
    "OO",
    "schwa"
]

TONGUE_OFFSET = 150

class VisemeTable:
    def __init__(self, filepath: str = "data/viseme_table.json"):
        self.filepath = filepath
        self.table: Dict[str, np.ndarray] = {}
        self.initialize_defaults()
        if os.path.exists(self.filepath):
            self.load()

    def initialize_defaults(self):
        for name in VISEMES:
            self.table[name] = np.zeros(182, dtype=np.float32)

        # GNM expression basis mapping (verified against model):
        # [0]=lower_face_region_000 (lip raise), [1]=lower_face_region_001 (jaw open/lip down),
        # [2]=lower_face_region_002 (lip spread/narrow), [3-149]=other lower face components
        # [150-181]=tongue components

        self.table["aa"][1] = 2.5
        self.table["aa"][0] = -0.5
        self.table["aa"][2] = 0.5

        self.table["EE"][2] = 2.0
        self.table["EE"][1] = 1.0
        self.table["EE"][3] = 0.3

        self.table["OO"][1] = -1.0
        self.table["OO"][2] = -0.8
        self.table["OO"][TONGUE_OFFSET] = -0.3

        self.table["PP"][1] = -1.5
        self.table["PP"][0] = -0.5
        self.table["PP"][3] = 0.5

        self.table["FF"][1] = -0.3
        self.table["FF"][3] = 0.5
        self.table["FF"][TONGUE_OFFSET + 2] = 0.3

        self.table["TH"][1] = 0.3
        self.table["TH"][TONGUE_OFFSET] = 1.0
        self.table["TH"][TONGUE_OFFSET + 1] = 0.8
        self.table["TH"][TONGUE_OFFSET + 2] = 0.3

        self.table["DD"][1] = 0.5
        self.table["DD"][TONGUE_OFFSET] = 0.6
        self.table["DD"][TONGUE_OFFSET + 1] = 0.7

        self.table["CH"][TONGUE_OFFSET] = 0.5
        self.table["CH"][TONGUE_OFFSET + 2] = 0.6
        self.table["CH"][TONGUE_OFFSET + 3] = 0.4

        self.table["kk"][2] = -0.3
        self.table["kk"][TONGUE_OFFSET + 1] = 0.7
        self.table["kk"][TONGUE_OFFSET + 2] = 0.5

        self.table["SS"][TONGUE_OFFSET] = 0.3
        self.table["SS"][TONGUE_OFFSET + 1] = 0.5
        self.table["SS"][TONGUE_OFFSET + 2] = 0.3

        self.table["RR"][TONGUE_OFFSET] = 0.6
        self.table["RR"][TONGUE_OFFSET + 1] = 0.4
        self.table["RR"][TONGUE_OFFSET + 2] = 0.5
        self.table["RR"][TONGUE_OFFSET + 3] = 0.3

        self.table["schwa"][1] = 0.5
        self.table["schwa"][TONGUE_OFFSET] = 0.3

    def get_coefficients(self, name: str) -> np.ndarray:
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
            data = {name: coeffs.tolist() for name, coeffs in self.table.items()}
            with open(self.filepath, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Saved viseme table to {self.filepath}")
        except Exception as e:
            print(f"Error saving viseme table to {self.filepath}: {e}")

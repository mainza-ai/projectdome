import os
import numpy as np
from gnm.shape.gnm_numpy import GNM
from gnm.shape.data.versions.gnm_specs import GNMMajorVersion, GNMVariant

class GNMDriver:
    def __init__(self):
        print("Loading GNM shape driver...")
        self.model = GNM.from_local(GNMMajorVersion.V3, GNMVariant.HEAD)
        print(f"GNM driver ready. Vertices: {self.model.num_vertices}")

    def evaluate(self, identity_coeffs: np.ndarray, expression_coeffs: np.ndarray) -> np.ndarray:
        """Run GNM forward pass to produce (17821, 3) vertex positions.
        
        identity_coeffs: (253,) array or None (defaults to zeros)
        expression_coeffs: (383,) array (contains merged speech and emotion weights)
        """
        if identity_coeffs is None:
            identity_coeffs = np.zeros(self.model.identity_dim, dtype=np.float32)
            
        # Joint rotations and global translation are zero (neutral skeleton pose)
        rotations = np.zeros((self.model.num_joints, 3), dtype=np.float32)
        translation = np.zeros(3, dtype=np.float32)
        
        # model expects (V, 3)
        vertices = self.model(identity_coeffs, expression_coeffs, rotations, translation)
        return vertices

    def save_mesh(self, vertices: np.ndarray, filepath: str):
        """Export the deformed vertex buffer to an OBJ file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            for v in vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            for t in self.model.triangles:
                f.write(f"f {t[0]+1} {t[1]+1} {t[2]+1}\n")
